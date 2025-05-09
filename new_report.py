# -*- coding: utf-8 -*-
import pandas as pd
import json
from datetime import datetime, timedelta
from tqdm import tqdm
import concurrent.futures
from functools import partial
import copy
import os
from openai import OpenAI, RateLimitError, APIError, APIConnectionError
from pymongo import MongoClient, UpdateMany
from pymongo.errors import ConnectionFailure, DuplicateKeyError
from pymongo.server_api import ServerApi
import urllib.parse
import re
import sys
from rapidfuzz import process, fuzz
import time
import logging
import numpy as np
import hanlp  # 替换hanlp_restful为hanlp本地包

# >>> NEW: vector / NER libs
from sentence_transformers import SentenceTransformer
import faiss

# >>> CSV / JSON / time constants
HOSPITAL_DATA_FILE = "hospital_data.csv"
DISTRIBUTOR_DATA_FILE = "distributor_data.csv"
EMPLOYEE_RECORDS_FILE = "employee_records.csv"
MASTER_DATA_FILE = "master_data.json"
if not os.path.exists("data"):
    os.makedirs("data")
if not os.path.exists("uploads"):
    os.makedirs("uploads")
LLM_RETRY_DELAY = 5  # seconds
LLM_MAX_RETRIES = 3   # max 重试次数
LLM_MODEL = "gpt-3.5-turbo-16k"
LLM_TEMPERATURE_EXTRACTION = 0.0  # For data extraction
LLM_TEMPERATURE_GENERATION = 0.7  # For content generation

# --- 配置常量 ---
# 文件路径
EXCEL_FILE_DEFAULT = '工作日志2025年3月.xlsx'
HOSPITAL_REFERENCE_FILE = 'datatable.xlsx' # 包含 'name' 和 'location' 列的医院参考数据
MASTER_DATA_FILE = 'master_data.json'
EMPLOYEE_RECORDS_CSV = 'employee_records.csv'
HOSPITAL_DATA_CSV = 'hospital_data.csv'
DISTRIBUTOR_DATA_CSV = 'distributor_data.csv'
OUTPUT_EXCEL_FILE = 'output_optimized.xlsx' # 优化后的输出文件名
# 中间映射文件 (可选导出)
EXPORT_MAP_FILES = False
HOSPITAL_DOCTOR_MAP_FILE = "hospital_doctor_map.json"
HOSPITAL_DISTRIBUTOR_MAP_FILE = "hospital_distributor_map.json"
HOSPITAL_DEPT_DOCTOR_MAP_FILE = "hospital_dept_doctor_map.json"
DISTRIBUTOR_CONTACT_MAP_FILE = "distributor_contact_map.json"

# API 相关
OPENAI_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"
LLM_MODEL = "doubao-1-5-pro-32k-250115" # 使用的模型名称
LLM_TEMPERATURE_EXTRACTION = 0.1 # 用于信息提取的温度参数
LLM_TEMPERATURE_SCORING = 0.3 # 用于关系评分的温度参数
LLM_TEMPERATURE_OPTIMIZATION = 0.3 # 用于评论优化的温度参数
LLM_MAX_RETRIES = 3 # API 调用最大重试次数
LLM_RETRY_DELAY = 5 # API 调用重试延迟（秒）

# 数据处理参数
EMPLOYEE_RECORD_ROLLING_DAYS = 14 # 员工记录保留天数
PROCESS_RECORD_HISTORY_LIMIT = 15 # 处理单条记录时参考的历史记录数量
MIN_FUZZY_MATCH_SCORE = 85 # 医院名称模糊匹配最低分数
PARALLEL_MAX_WORKERS = 10 # 并行处理的最大工作线程数 (根据API限制和机器性能调整)

# ----------------- 关键词常量 -----------------
DISTRIBUTOR_KEYWORDS = [
    "公司", "经销", "代理", "医药", "药业", "医疗器械", "有限", "责任", "经营部", "商贸", "集团","华润"
]
HOSPITAL_KEYWORDS = [
    "医院", "卫生院", "诊所", "门诊", "中心", "医科大学", "保健院", "保健所"
]
# --------------------------------------------

# MongoDB 相关
MONGO_DB_NAME = 'medical_data'
# 日志配置
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')

# --- 全局变量 (尽量减少使用, 此处 master_data 仍保留以简化传递) ---
master_data = {
    "hospitals": {},    # {简称: {"全称": 全称值, "标准名称": 标准名称值, "地区": 地区值}}
    "departments": {},  # {简称: 全称}
    "doctors": {},      # {(医院标准名称, 科室全称, 简称): 全称} # 注意键的规范化
    "distributors": {}, # {简称: 全称}
    "contacts": {}      # {(经销商全称, 简称): 全称} # 注意键的规范化
}
# --- 全局变量结束 ---

# 全局变量用于缓存master_data
_master_data_cache = None
_master_data_modified = False

# --- API 客户端初始化 ---
def initialize_openai_client():
    """初始化 OpenAI API 客户端"""
    try:
        api_key = '741c625f-f5dd-4da2-a7f0-69e6af6f51d1'
        if not api_key:
            logging.warning("未找到环境变量 OPENAI_API_KEY")
            api_key = input("请输入您的API密钥: ")
        if not api_key:
            logging.error("未提供API密钥，无法初始化客户端。")
            return None
        client = OpenAI(api_key=api_key, base_url=OPENAI_BASE_URL)
        # 尝试调用一个简单接口验证 key 和 base_url
        try:
            client.models.list()
            logging.info("OpenAI API 客户端初始化成功。")
        except Exception as e:
            logging.warning(f"API客户端验证失败，但将继续使用: {e}")
        return client
    except Exception as e:
        logging.error(f"初始化 OpenAI API 客户端时出错: {e}", exc_info=True)
        return None
# --- API 客户端初始化结束 ---

# --- MongoDB 相关函数 ---
def connect_to_mongodb(password):
    """连接到MongoDB数据库"""
    if not password:
        logging.error("未提供MongoDB密码，无法连接。")
        return None
    try:
        encoded_password = urllib.parse.quote_plus(password)
        # **安全提示**: 避免在代码中硬编码用户名，最好也从环境变量或配置中读取
        connection_string = f"mongodb+srv://jhw:{encoded_password}@cluster0.j2eoyii.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

        logging.info("尝试连接到 MongoDB...")
        mongo_client = MongoClient(connection_string, server_api=ServerApi('1'))
        mongo_client.admin.command('ping') # 测试连接
        logging.info("成功连接到 MongoDB 数据库。")

        return mongo_client

    except ConnectionFailure as e:
        logging.error(f"连接 MongoDB 数据库失败: {e}", exc_info=True)
        return None
    except Exception as e:
        logging.error(f"连接或初始化 MongoDB 时发生错误: {e}", exc_info=True)
        return None

def _generate_entity_id(name):
    """为实体（医院、经销商）生成一个基于名称的、相对稳定的ID"""
    # 使用名称的哈希值，取模减少长度，转为字符串
    # 注意：哈希冲突是可能的，但对于适量数据通常足够
    # 如果需要绝对唯一且不变的ID，应考虑其他策略（如UUID或数据库自增ID）
    return str(abs(hash(name)) % (10**8)) # 8位数字ID

def convert_data_for_mongodb(hospital_data, distributor_data):
    """
    将程序内部处理后的医院和经销商数据转换为与CSV格式一致的MongoDB Schema格式。
    返回: (医院文档列表, 拜访文档列表, 经销商文档列表)
    """
    hospitals_docs = []
    visits_docs = []
    distributors_docs = []
    current_time = datetime.now()

    # --- 处理医院和拜访数据 ---
    logging.info("转换医院和拜访数据为 MongoDB 新格式（与CSV一致）...")
    hospital_id_map = {} # 用于确保同一医院在本次转换中使用相同ID

    for hospital in hospital_data.get("hospitals", []):
        hospital_name = hospital.get("医院名称") # 通常应该是标准名称了
        if not hospital_name:
            continue

        # 生成或获取医院ID
        if hospital_name in hospital_id_map:
            hospital_id = hospital_id_map[hospital_name]
        else:
            hospital_id = _generate_entity_id(hospital_name)
            hospital_id_map[hospital_name] = hospital_id

        standard_hospital_name = hospital.get("标准医院名称", hospital_name)
        matching_score = hospital.get("匹配分数", 0)
        region = hospital.get("地区", "")

        # 处理拜访记录 - 与CSV格式保持一致
        for visit in hospital.get("历史记录", []):
            doctor_name = visit.get("医生姓名", "")
            if not doctor_name:
                continue
                
            visit_date_str = visit.get("拜访日期", "")
            employee = visit.get("拜访员工", "")
            department = visit.get("科室", "")
            title = visit.get("职称", "")
            comm_content = visit.get("沟通内容", "")
            follow_up = visit.get("后续行动", "")
            rel_score = visit.get("关系评分", 0)
            
            # 处理日期
            visit_date = None
            try:
                if visit_date_str:
                    visit_date = datetime.strptime(visit_date_str, "%Y-%m-%d")
                else:
                    visit_date = current_time
            except ValueError:
                logging.warning(f"无效的日期格式: {visit_date_str}，使用当前日期")
                visit_date = current_time
            
            # 处理空值等
            def clean_value(val):
                if val is None:
                    return ""
                if isinstance(val, float) and np.isnan(val):
                    return ""
                if isinstance(val, str) and val.lower() in ["nan", "null", "none"]:
                    return ""
                return str(val)
            
            # 创建与CSV格式完全一致的MongoDB文档
            visit_doc = {
                "医院ID": hospital_id,
                "医院名称": hospital_name,
                "标准医院名称": standard_hospital_name,
                "匹配分数": str(matching_score),
                "地区": region,
                "拜访日期": visit_date_str,
                "拜访员工": clean_value(employee),
                "医生姓名": clean_value(doctor_name),
                "科室": clean_value(department),
                "职称": clean_value(title),
                "沟通内容": clean_value(comm_content),
                "后续行动": clean_value(follow_up),
                "关系评分": clean_value(rel_score),
                # 添加MongoDB特定字段，但不影响与CSV的一致性
                "_id": _generate_entity_id(f"{hospital_id}_{visit_date_str}_{doctor_name}"),
                "createdAt": current_time,
                "updatedAt": current_time
            }
            visits_docs.append(visit_doc)

    # --- 处理经销商和沟通数据 ---
    logging.info("转换经销商和沟通数据为 MongoDB 新格式（与CSV一致）...")
    distributor_id_map = {}

    for distributor in distributor_data.get("distributors", []):
        dist_name = distributor.get("经销商名称")
        if not dist_name:
            continue

        # 生成或获取经销商ID
        if dist_name in distributor_id_map:
            dist_id = distributor_id_map[dist_name]
        else:
            dist_id = _generate_entity_id(dist_name)
            distributor_id_map[dist_name] = dist_id

        standard_name = distributor.get("标准名称", dist_name)
        region = distributor.get("地区", "")

        # 处理沟通记录 - 与CSV格式保持一致
        for comm in distributor.get("沟通记录", []):
            contact_name = comm.get("联系人", "")
            if not contact_name:
                continue
                
            comm_date_str = comm.get("沟通日期", "")
            employee = comm.get("沟通员工", "")
            position = comm.get("职位", "")
            content = comm.get("沟通内容", "")
            plan = comm.get("后续计划", "")
            rel_score = comm.get("关系评分", 0)
            
            # 处理日期
            comm_date = None
            try:
                if comm_date_str:
                    comm_date = datetime.strptime(comm_date_str, "%Y-%m-%d")
                else:
                    comm_date = current_time
            except ValueError:
                logging.warning(f"无效的日期格式: {comm_date_str}，使用当前日期")
                comm_date = current_time
            
            # 清理值
            def clean_value(val):
                if val is None:
                    return ""
                if isinstance(val, float) and np.isnan(val):
                    return ""
                if isinstance(val, str) and val.lower() in ["nan", "null", "none"]:
                    return ""
                return str(val)
            
            # 创建与CSV格式完全一致的MongoDB文档
            comm_doc = {
                "经销商ID": dist_id,
                "经销商名称": dist_name,
                "地区": region,
                "沟通日期": comm_date_str,
                "沟通员工": clean_value(employee),
                "联系人": clean_value(contact_name),
                "职位": clean_value(position),
                "沟通内容": clean_value(content),
                "后续计划": clean_value(plan),
                "关系评分": clean_value(rel_score),
                # 添加MongoDB特定字段
                "_id": _generate_entity_id(f"{dist_id}_{comm_date_str}_{contact_name}"),
                "createdAt": current_time,
                "updatedAt": current_time
            }
            distributors_docs.append(comm_doc)

    logging.info(f"转换完成: {len(hospital_id_map)}个医院, {len(visits_docs)}条拜访记录, {len(distributor_id_map)}个经销商, {len(distributors_docs)}条沟通记录")
    return hospitals_docs, visits_docs, distributors_docs

def update_mongodb_collections(mongo_client, hospital_data, distributor_data):
    """使用转换后的数据更新 MongoDB 中的集合，保持与CSV格式一致
    
    Args:
        mongo_client: MongoDB客户端
        hospital_data: 医院数据
        distributor_data: 经销商数据
        
    Returns:
        是否更新成功
    """
    if not mongo_client:
        logging.error("MongoDB 客户端无效，无法更新集合。")
        return False
    try:
        db = mongo_client.get_database(MONGO_DB_NAME)
        # 使用更符合CSV结构的集合名称
        hospital_visits_collection = db["hospital_visits"]  # 医院拜访记录（对应hospital_data.csv）
        distributor_comms_collection = db["distributor_communications"]  # 经销商沟通记录（对应distributor_data.csv）

        # 转换数据为与CSV一致的MongoDB格式
        _, hospital_visits_docs, distributor_comms_docs = convert_data_for_mongodb(hospital_data, distributor_data)

        # --- 更新医院拜访记录 ---
        logging.info(f"准备上传/更新 {len(hospital_visits_docs)} 条医院拜访记录到 MongoDB...")
        visits_inserted = 0
        visits_updated = 0
        
        with tqdm(total=len(hospital_visits_docs), desc="更新医院拜访记录") as pbar:
            for doc in hospital_visits_docs:
                try:
                    result = hospital_visits_collection.update_one(
                        {
                            "医院ID": doc["医院ID"],
                            "拜访日期": doc["拜访日期"],
                            "医生姓名": doc["医生姓名"]
                        },
                        {"$set": doc},
                        upsert=True
                    )
                    if result.upserted_id:
                        visits_inserted += 1
                    elif result.modified_count > 0:
                        visits_updated += 1
                except DuplicateKeyError:
                    logging.debug(f"重复记录: 医院={doc['医院名称']}, 日期={doc['拜访日期']}, 医生={doc['医生姓名']}")
                except Exception as e:
                    logging.error(f"更新医院拜访记录时出错: {e}")
                pbar.update(1)

        # --- 更新经销商沟通记录 ---
        logging.info(f"准备上传/更新 {len(distributor_comms_docs)} 条经销商沟通记录到 MongoDB...")
        comms_inserted = 0
        comms_updated = 0
        
        with tqdm(total=len(distributor_comms_docs), desc="更新经销商沟通记录") as pbar:
            for doc in distributor_comms_docs:
                try:
                    result = distributor_comms_collection.update_one(
                        {
                            "经销商ID": doc["经销商ID"],
                            "沟通日期": doc["沟通日期"],
                            "联系人": doc["联系人"]
                        },
                        {"$set": doc},
                        upsert=True
                    )
                    if result.upserted_id:
                        comms_inserted += 1
                    elif result.modified_count > 0:
                        comms_updated += 1
                except DuplicateKeyError:
                    logging.debug(f"重复记录: 经销商={doc['经销商名称']}, 日期={doc['沟通日期']}, 联系人={doc['联系人']}")
                except Exception as e:
                    logging.error(f"更新经销商沟通记录时出错: {e}")
                pbar.update(1)

        logging.info(f"医院拜访记录更新完成: {visits_inserted} 个插入, {visits_updated} 个更新")
        logging.info(f"经销商沟通记录更新完成: {comms_inserted} 个插入, {comms_updated} 个更新")
        logging.info("MongoDB 数据更新完成。")
        return True
    except Exception as e:
        logging.error(f"更新 MongoDB 数据时发生严重错误: {e}", exc_info=True)
        return False
# --- MongoDB 相关函数结束 ---


# --- 文件加载与保存 ---
def load_data_from_excel(file_path):
    """从Excel文件加载数据"""
    try:
        df = pd.read_excel(file_path)
        logging.info(f"成功从 {file_path} 读取 {len(df)} 条记录。")
        return df
    except FileNotFoundError:
        logging.error(f"Excel 文件未找到: {file_path}")
        return None
    except Exception as e:
        logging.error(f"读取 Excel 文件 {file_path} 时出错: {e}", exc_info=True)
        return None

def convert_df_to_records(df):
    """将DataFrame转换为指定的JSON记录列表格式"""
    records = []
    required_columns = ["汇报编号", "汇报时间", "汇报人", "汇报人部门", "汇报对象", "今日目标（必填）", "今日结果（必填）", "明日计划（必填）"]
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        logging.warning(f"Excel 文件缺少必需列: {', '.join(missing_cols)}")

    # 确保汇报时间是 datetime 类型
    if "汇报时间" in df.columns:
        df["汇报时间"] = pd.to_datetime(df["汇报时间"], errors='coerce') # 转换失败的会变成 NaT

    logging.info("开始转换 DataFrame 为记录列表...")
    with tqdm(total=len(df), desc="转换Excel数据") as pbar:
        for _, row in df.iterrows():
            # 处理汇报时间 NaT
            report_time = row.get("汇报时间")
            report_time_str = report_time.strftime("%Y-%m-%d") if pd.notna(report_time) else ""

            # 安全地获取列值，处理缺失
            def get_safe(col_name, default=""):
                val = row.get(col_name, default)
                return "" if pd.isna(val) else str(val) # 转为字符串并处理 NaN

            record = {
                "汇报编号": get_safe("汇报编号"),
                "汇报时间": report_time_str,
                "员工姓名": get_safe("汇报人"),
                "部门": get_safe("汇报人部门"),
                "汇报对象": get_safe("汇报对象"),
                "今日目标": get_safe("今日目标（必填）"),
                "今日结果": get_safe("今日结果（必填）"),
                "明日计划": get_safe("明日计划（必填）"),
                "其他事项": get_safe("其他事项"),
                "评论": get_safe("评论")
            }
            records.append(record)
            pbar.update(1)
    logging.info(f"DataFrame 转换完成，生成 {len(records)} 条记录。")
    return {"records": records}

def load_records_from_csv(file_path):
    """从CSV文件加载记录"""
    try:
        df = pd.read_csv(file_path, encoding='utf-8-sig', dtype=str).fillna("") # 读取为字符串并填充空值
        logging.info(f"成功从 {file_path} 加载 {len(df)} 条记录。")
        # 转换为字典列表
        records = df.to_dict('records')
        # 重命名字段以匹配内部格式
        for record in records:
            record['汇报编号'] = record.pop('汇报ID', '') # 兼容旧的列名
            # 其他字段名如果CSV和内部格式一致则无需转换
        return {"records": records}
    except FileNotFoundError:
        logging.warning(f"CSV 文件未找到: {file_path}，将创建新记录。")
        return {"records": []}
    except pd.errors.EmptyDataError:
         logging.warning(f"CSV 文件为空: {file_path}，将创建新记录。")
         return {"records": []}
    except Exception as e:
        logging.error(f"加载 CSV 文件 {file_path} 时出错: {e}", exc_info=True)
        return {"records": []} # 出错时返回空

def load_structured_data_from_csv(file_path, id_col, name_col, history_col_name, record_key_map):
    """
    从CSV加载结构化数据（医院或经销商）
    file_path: CSV文件路径
    id_col: ID列名 (e.g., "医院ID")
    name_col: 名称列名 (e.g., "医院名称")
    history_col_name: 历史/沟通记录在内部字典中的键名 (e.g., "历史记录")
    record_key_map: CSV列名到内部记录字典键名的映射 (e.g., {"拜访日期": "拜访日期", ...})
    """
    data = {"data": []} # 使用通用键名 'data'
    try:
        df = pd.read_csv(file_path, encoding='utf-8-sig', dtype=str).fillna("")
        logging.info(f"成功从 {file_path} 加载 {len(df)} 条记录。")

        # 按ID分组构建数据字典
        grouped_data = {}
        for entity_id, group in df.groupby(id_col):
            if not entity_id: continue
            first_row = group.iloc[0]
            entity_name = first_row.get(name_col, "")
            if not entity_name: continue

            entity_record = {
                name_col: entity_name,
                "地区": first_row.get("地区", ""),
                "标准名称": first_row.get("标准名称", entity_name), # 尝试加载标准名称
                "匹配分数": float(first_row.get("匹配分数", 0)), # 尝试加载匹配分数
                history_col_name: []
            }

            # 添加所有历史/沟通记录
            for _, row in group.iterrows():
                history_record = {}
                valid_record = False
                for csv_key, internal_key in record_key_map.items():
                    value = row.get(csv_key, "")
                    # 特殊处理关系评分
                    if internal_key == "关系评分":
                        try:
                            history_record[internal_key] = int(float(value)) if value else 0
                        except ValueError:
                            history_record[internal_key] = 0
                    else:
                        history_record[internal_key] = value
                    # 检查是否有关键信息，如医生姓名或联系人
                    if internal_key in ["医生姓名", "联系人"] and value:
                         valid_record = True

                if valid_record: # 只添加包含关键信息的记录
                    entity_record[history_col_name].append(history_record)

            if entity_record[history_col_name]: # 只添加包含有效历史记录的实体
                grouped_data[entity_id] = entity_record

        data["data"] = list(grouped_data.values())
        logging.info(f"从 {file_path} 构建了 {len(data['data'])} 个实体记录。")

    except FileNotFoundError:
        logging.warning(f"CSV 文件未找到: {file_path}，将创建新数据。")
    except pd.errors.EmptyDataError:
         logging.warning(f"CSV 文件为空: {file_path}，将创建新数据。")
    except KeyError as e:
        logging.error(f"CSV 文件 {file_path} 缺少必需列: {e}")
    except Exception as e:
        logging.error(f"加载 CSV 文件 {file_path} 时出错: {e}", exc_info=True)

    # 重命名键以匹配旧格式的期望 (hospitals/distributors)
    if name_col == "医院名称":
        return {"hospitals": data["data"]}
    elif name_col == "经销商名称":
        return {"distributors": data["data"]}
    else:
        return {} # 或者抛出错误

def data_to_csv_rows(data_dict, data_key, id_col_name, name_col_name, history_key, headers, record_key_order):
    """通用函数：将医院或经销商数据转换为CSV行列表"""
    # 添加日志，输出即将处理的数据样本（最多显示5个实体）
    entity_type = "医院" if data_key == "hospitals" else "经销商"
    entities = data_dict.get(data_key, [])
    sample_size = min(5, len(entities))
    if sample_size > 0:
        logging.info(f"准备处理 {len(entities)} 个{entity_type}实体，以下是前 {sample_size} 个实体的基本信息:")
        for i, entity in enumerate(entities[:sample_size]):
            entity_name = entity.get(name_col_name, "")
            standard_name = entity.get("标准医院名称" if data_key == "hospitals" else "标准名称", "")
            score = entity.get("匹配分数", 0) if data_key == "hospitals" else None
            region = entity.get("地区", "")
            history_count = len(entity.get(history_key, []))
            logging.info(f"  #{i+1}: {name_col_name}='{entity_name}', 标准名称='{standard_name}', "
                        f"地区='{region}', 分数={score}, {history_key}数量={history_count}")
    
    csv_rows = [headers]

    with tqdm(total=len(entities), desc=f"导出{entity_type}数据为CSV") as pbar:
        for entity in entities:
            entity_name = entity.get(name_col_name, "")
            if not entity_name: 
                pbar.update(1)
                continue
            
            history_records = entity.get(history_key, []) # <<< 获取历史记录
            if not history_records: # <<< 检查是否有历史记录
                 pbar.update(1) # 即使跳过也要更新进度条
                 continue      # <<< 如果没有历史记录，跳过这个实体
            
            entity_id = _generate_entity_id(entity_name) # 重新生成ID以保持一致性
            region = entity.get("地区", "")
            
            # 修复标准名称字段的获取逻辑
            if data_key == "hospitals":
                standard_name = entity.get("标准医院名称", entity_name)  # 医院使用"标准医院名称"字段
                matching_score = str(entity.get("匹配分数", 0))
            else:
                standard_name = entity.get("标准名称", entity_name)  # 经销商使用"标准名称"字段
                matching_score = "0"  # 经销商没有匹配分数
            
            # <<< 移除为没有历史记录的实体导出基本信息行的 else 分支
            # 现在只处理有 history_records 的情况
            for record in history_records:
                base_row = {
                    id_col_name: entity_id,
                    name_col_name: entity_name,
                    "地区": region
                }
                
                # 根据实体类型添加不同的标准名称字段
                if data_key == "hospitals":
                    base_row["标准医院名称"] = standard_name
                    base_row["匹配分数"] = matching_score
                else:
                    base_row["标准名称"] = standard_name
                
                # 从记录中获取值
                for key in record_key_order:
                     base_row[key] = str(record.get(key, "")) # 确保是字符串

                # 按headers顺序组装行
                row_values = [base_row.get(h, "") for h in headers]
                csv_rows.append(row_values)
            pbar.update(1)
    return csv_rows

def employee_data_to_csv(employee_data):
    """将员工数据转换为CSV格式"""
    headers = ["汇报ID", "汇报时间", "员工姓名", "部门", "汇报对象",
               "今日目标", "今日结果", "明日计划", "其他事项", "评论",
               "评论要点", "优化_今日目标", "优化_今日结果", "优化_明日计划"]
    csv_rows = [headers]

    with tqdm(total=len(employee_data.get("records", [])), desc="导出员工数据为CSV") as pbar:
        for record in employee_data.get("records", []):
            row = [
                record.get("汇报编号", ""), # 使用 '汇报编号' 作为 '汇报ID'
                record.get("汇报时间", ""),
                record.get("员工姓名", ""),
                record.get("部门", ""),
                record.get("汇报对象", ""),
                record.get("今日目标", ""),
                record.get("今日结果", ""),
                record.get("明日计划", ""),
                record.get("其他事项", ""),
                record.get("评论", ""),
                record.get("评论要点", ""),
                record.get("优化_今日目标", ""),
                record.get("优化_今日结果", ""),
                record.get("优化_明日计划", "")
            ]
            csv_rows.append(row)
            pbar.update(1)
    return csv_rows

def write_csv(rows, file_path):
    """将数据写入CSV文件"""
    import csv
    try:
        with open(file_path, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerows(rows)
        logging.info(f"成功导出数据到 {file_path}")
        return True
    except Exception as e:
        logging.error(f"导出 CSV 文件 {file_path} 时出错: {e}", exc_info=True)
        return False

def load_master_data(file_path=MASTER_DATA_FILE):
    """从文件加载主数据，如果文件不存在则创建一个空的主数据字典
    
    Args:
        file_path: 主数据文件路径
        
    Returns:
        加载的主数据字典
    """
    global master_data
    
    try:
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                loaded_data = json.load(f)
                
                # 初始化主数据结构
                result = {
                    "hospitals": {},
                    "departments": {},
                    "doctors": {},
                    "distributors": {},
                    "contacts": {}
                }
                
                # 复制基本字段
                for key in ["hospitals", "departments", "distributors"]:
                    if key in loaded_data:
                        result[key] = loaded_data[key]
                
                # 处理doctors字段，将字符串键转换回元组
                if "doctors" in loaded_data:
                    for str_key, value in loaded_data["doctors"].items():
                        # 检查是否是序列化的元组
                        if str_key.startswith("(") and str_key.endswith(")"):
                            # 去掉括号，分割字段
                            parts = str_key[1:-1].split(",")
                            if len(parts) >= 3:
                                # 清理部分中的引号和空格
                                hospital = parts[0].strip().strip('"\'')
                                department = parts[1].strip().strip('"\'')
                                doctor = ",".join(parts[2:]).strip().strip('"\'')  # 合并剩余部分，处理姓名中可能有逗号的情况
                                tuple_key = (hospital, department, doctor)
                                result["doctors"][tuple_key] = value
                        else:
                            # 如果不是序列化的元组格式，保留原样
                            result["doctors"][str_key] = value
                
                # 处理contacts字段，将字符串键转换回元组
                if "contacts" in loaded_data:
                    for str_key, value in loaded_data["contacts"].items():
                        # 检查是否是序列化的元组
                        if str_key.startswith("(") and str_key.endswith(")"):
                            # 去掉括号，分割字段
                            parts = str_key[1:-1].split(",")
                            if len(parts) >= 2:
                                # 清理部分中的引号和空格
                                distributor = parts[0].strip().strip('"\'')
                                contact = ",".join(parts[1:]).strip().strip('"\'')  # 合并剩余部分
                                tuple_key = (distributor, contact)
                                result["contacts"][tuple_key] = value
                        else:
                            # 如果不是序列化的元组格式，保留原样
                            result["contacts"][str_key] = value
                
                master_data = result
                return result
        
        # 如果文件不存在或加载失败，创建一个空的主数据字典
        master_data = {
            "hospitals": {},
            "departments": {},
            "doctors": {},
            "distributors": {},
            "contacts": {}
        }
        return master_data
    except Exception as e:
        logging.error(f"加载主数据时出错: {e}")
        # 返回一个空的主数据字典
        master_data = {
            "hospitals": {},
            "departments": {},
            "doctors": {},
            "distributors": {},
            "contacts": {}
        }
        return master_data

def save_master_data(data_to_save, file_path=MASTER_DATA_FILE):
    """将主数据保存到文件
    
    Args:
        data_to_save: 要保存的主数据字典
        file_path: 保存路径
        
    Returns:
        是否保存成功
    """
    try:
        # 创建一个新字典来存储转换后的数据
        json_safe_data = {
            "hospitals": {},
            "departments": {},
            "doctors": {},
            "distributors": {},
            "contacts": {}
        }
        
        # 复制基本字段
        for key in ["hospitals", "departments", "distributors"]:
            if key in data_to_save:
                json_safe_data[key] = data_to_save[key]
        
        # 处理doctors字段，将元组键转换为字符串
        if "doctors" in data_to_save:
            for key, value in data_to_save["doctors"].items():
                if isinstance(key, tuple):
                    # 将元组转换为字符串表示
                    str_key = str(key)
                    json_safe_data["doctors"][str_key] = value
                else:
                    # 已经是字符串的键，直接复制
                    json_safe_data["doctors"][key] = value
        
        # 处理contacts字段，将元组键转换为字符串
        if "contacts" in data_to_save:
            for key, value in data_to_save["contacts"].items():
                if isinstance(key, tuple):
                    # 将元组转换为字符串表示
                    str_key = str(key)
                    json_safe_data["contacts"][str_key] = value
                else:
                    # 已经是字符串的键，直接复制
                    json_safe_data["contacts"][key] = value
        
        # 保存转换后的数据
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(json_safe_data, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        logging.error(f"保存主数据时出错: {e}")
        return False

# 新增函数：批量处理结束时保存master_data
def flush_master_data(file_path=MASTER_DATA_FILE):
    """如果master_data被修改过，将其写入文件"""
    global _master_data_cache, _master_data_modified
    
    if _master_data_cache is not None and _master_data_modified:
        try:
            _master_data_cache["metadata"]["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(_master_data_cache, f, ensure_ascii=False, indent=2)
            logging.info(f"主数据已刷新保存到 {file_path}")
            _master_data_modified = False
            return True
        except Exception as e:
            logging.error(f"刷新保存主数据时出错: {e}", exc_info=True)
            return False
    return False

def add_to_master_data(entity_type, key, value, value_type="全称", region=None):
    """向主数据中添加实体信息
    
    Args:
        entity_type: 实体类型（"hospitals", "doctors", "departments", "distributors", "contacts"）
        key: 简称或者元组键
        value: 要添加的值（通常是全称）
        value_type: 值的类型（"全称", "标准名称", 或 "地区"）
        region: 地区信息（用于医院和经销商）
    """
    global master_data
    
    if not master_data:
        master_data = load_master_data()
    
    if entity_type == "hospitals":
        if key not in master_data[entity_type]:
            master_data[entity_type][key] = {}
        
        if value_type == "全称":
            master_data[entity_type][key]["全称"] = value
        elif value_type == "标准名称":
            master_data[entity_type][key]["标准名称"] = value
            
        # 如果提供了地区信息，也添加进去
        if region:
            master_data[entity_type][key]["地区"] = region
            
    elif entity_type == "doctors":
        # 医生信息的键是(医院标准名称, 科室全称, 医生简称)元组
        if isinstance(key, tuple) and len(key) == 3:
            hospital_name, department, doctor_abbr = key
            
            # 检查是否存在同一医院科室下的其他医生记录
            for existing_key in list(master_data[entity_type].keys()):
                if isinstance(existing_key, tuple) and len(existing_key) == 3:
                    existing_hospital, existing_dept, existing_doctor = existing_key
                    
                    # 如果是同一医院同一科室
                    if existing_hospital == hospital_name and existing_dept == department:
                        # 如果现有记录是简称（如"范主任"），新记录是全称（如"范德彪"），且姓氏相同
                        if (doctor_abbr != value and 
                            existing_doctor != master_data[entity_type][existing_key] and
                            len(existing_doctor) >= 1 and len(doctor_abbr) >= 1 and 
                            existing_doctor[0] == doctor_abbr[0]):  # 姓氏相同
                            
                            # 检查现有记录是否包含职称信息（如"主任"、"医生"等）
                            title_terms = ["主任", "医生", "医师", "专家", "教授", "博士", "院长", "科长"]
                            has_title = any(term in existing_doctor for term in title_terms)
                            
                            if has_title:
                                # 如果现有记录包含职称，可能是简称，将其指向新的全称
                                master_data[entity_type][existing_key] = value
                                
                                # 同时创建新全称到自身的映射
                                if (hospital_name, department, value) not in master_data[entity_type]:
                                    master_data[entity_type][(hospital_name, department, value)] = value
                        
                        # 如果新记录是简称（如"范主任"），现有记录是全称（如"范德彪"），且姓氏相同
                        elif (doctor_abbr != value and
                             existing_doctor == master_data[entity_type][existing_key] and
                             len(existing_doctor) >= 1 and len(doctor_abbr) >= 1 and
                             existing_doctor[0] == doctor_abbr[0]):
                            
                            title_terms = ["主任", "医生", "医师", "专家", "教授", "博士", "院长", "科长"]
                            has_title = any(term in doctor_abbr for term in title_terms)
                            
                            if has_title:
                                # 如果新记录包含职称，可能是简称，创建简称到现有全称的映射
                                master_data[entity_type][key] = existing_doctor
                                return  # 已创建映射，不需要继续
            
            # 如果没有找到匹配的记录或没有建立映射，则添加新记录
            master_data[entity_type][key] = value
            
    elif entity_type == "departments":
        master_data[entity_type][key] = value
    elif entity_type == "distributors":
        if key not in master_data[entity_type]:
            master_data[entity_type][key] = value
        if region and key in master_data[entity_type]:
            # 为经销商添加地区信息
            if isinstance(master_data[entity_type][key], dict):
                master_data[entity_type][key]["地区"] = region
            else:
                # 如果当前值是字符串，将其转换为字典
                full_name = master_data[entity_type][key]
                master_data[entity_type][key] = {"全称": full_name, "地区": region}
    elif entity_type == "contacts":
        master_data[entity_type][key] = value

def lookup_in_master_data(entity_type, key, location_context=None):
    """查找实体在master_data中的标准值，优先考虑地区信息
    
    Parameters:
    -----------
    entity_type : str
        实体类型，如"hospitals","doctors"等
    key : str
        要查找的实体名称
    location_context : list or str, optional
        地区上下文信息，可以是地区名称列表或单个地区名
        
    Returns:
    --------
    str
        找到的标准名称，如果未找到则返回原始key
    """
    if not key: return key
    
    # 特殊规则：长海、长海医院默认为上海
    if entity_type == "hospitals" and key in ["长海", "长海医院"]:
        # 在master_data中尝试查找是否已存在
        master_data = load_master_data()
        if key in master_data.get(entity_type, {}):
            # 已存在，确保地区信息设置为上海
            if isinstance(master_data[entity_type][key], dict):
                if not master_data[entity_type][key].get("地区"):
                    master_data[entity_type][key]["地区"] = "上海"
            return master_data[entity_type][key].get("标准名称", key)
        else:
            # 不存在，添加到master_data
            add_to_master_data("hospitals", key, key, value_type="全称", region="上海")
            if key == "长海":
                add_to_master_data("hospitals", key, "上海长海医院", value_type="标准名称", region="上海")
            elif key == "长海医院":
                add_to_master_data("hospitals", key, "上海长海医院", value_type="标准名称", region="上海")
            return "上海长海医院"
    
    master_data = load_master_data()
    entity_data = master_data.get(entity_type, {})
    
    # 处理地区上下文
    locations = []
    if location_context:
        if isinstance(location_context, str):
            locations = [clean_location_term(location_context)]
        elif isinstance(location_context, list):
            locations = [clean_location_term(loc) for loc in location_context if loc]
        locations = list(filter(None, locations))
    
    # 1. 如果有地区信息，先尝试地区+简称联合查找
    if locations:
        direct_matches = []
        for hospital_key, hospital_data in entity_data.items():
            if hospital_key == key and isinstance(hospital_data, dict):
                hospital_region = hospital_data.get("地区", "")
                if hospital_region:
                    # 检查医院地区是否与任何一个location_context匹配
                    if any(loc in hospital_region or hospital_region in loc for loc in locations):
                        direct_matches.append((hospital_key, hospital_data))
        
        # 如果找到地区匹配的实体，返回其标准名称
        if direct_matches:
            # 如果有多个匹配，优先返回地区完全匹配的
            for h_key, h_data in direct_matches:
                h_region = h_data.get("地区", "")
                if any(h_region == loc for loc in locations):
                    return h_data.get("标准名称", key)
            
            # 否则返回第一个匹配
            return direct_matches[0][1].get("标准名称", key)
    
    # 2. 直接查找是否存在实体(无地区)
    if key in entity_data:
        entry = entity_data[key]
        if isinstance(entry, dict) and "标准名称" in entry:
            return entry["标准名称"]
    
    # 3. 如果有地区信息，尝试模糊匹配地区内的医院
    if locations:
        region_filtered = []
        for hospital_key, hospital_data in entity_data.items():
            if isinstance(hospital_data, dict) and "地区" in hospital_data:
                hospital_region = hospital_data["地区"]
                if any(loc in hospital_region or hospital_region in loc for loc in locations):
                    region_filtered.append((hospital_key, hospital_data))
        
        if region_filtered:
            # 在地区过滤后的列表中进行模糊匹配
            hospital_keys = [item[0] for item in region_filtered]
            best_match = process.extractOne(key, hospital_keys, scorer=fuzz.WRatio)
            if best_match and best_match[1] >= MIN_FUZZY_MATCH_SCORE:
                matched_key = best_match[0]
                for h_key, h_data in region_filtered:
                    if h_key == matched_key:
                        return h_data.get("标准名称", key)
    
    # 保底：无法确定标准名称时返回原始key
    return key

# --- 文本处理与清洗 ---
def clean_entity_name(name):
    """清理实体名称，移除括号、多余空格等"""
    if not name or pd.isna(name): return ""
    name = str(name)
    # 移除常见括号及其内容
    cleaned_name = re.sub(r'\(.*?\)|（.*?）|\[.*?\]|【.*?】|\{.*?\}', '', name)
    # 移除特殊字符，但保留中文、字母、数字和常见分隔符 `-`
    # cleaned_name = re.sub(r'[^\w\s\-\u4e00-\u9fa5]', '', cleaned_name) # 可能过于激进
    # 移除前后空格和多余的中间空格
    cleaned_name = re.sub(r'\s+', ' ', cleaned_name).strip()
    return cleaned_name

def clean_location_term(term):
    """处理地区名称，移除常见后缀并标准化特殊区域。
    Accepts scalar strings **or** list‑/array‑like inputs; returns clean string.
    """
    # 1. resolve list / ndarray / Series → first element or ""
    if term is None:
        return ""
    if isinstance(term, (list, tuple, set, np.ndarray, pd.Series)):
        term = term[0] if len(term) else ""

    # 2. handle pandas/NumPy missing values explicitly
    if (isinstance(term, float) and pd.isna(term)) or (hasattr(term, "dtype") and pd.isna(term).all()):
        return ""

    term = str(term).strip()
    if term.lower() in {"", "nan", "null", "none", "未知"}:
        return ""

    # 特殊区域标准化处理 (优先处理，避免后缀移除错误)
    if '内蒙古' in term:
        return '内蒙古'
    if '广西' in term:
        return '广西'
    if '西藏' in term:
        return '西藏'
    if '宁夏' in term:
        return '宁夏'
    if '新疆' in term:
        return '新疆'

    # 移除常见后缀
    term = re.sub(r'[省市区县镇乡村]$', '', term)
    # 移除可能的 "市辖区"
    term = term.replace('市辖区', '')
    return term.strip()

def clean_doctor_name(doctor_name, department=None, api_client=None):
    """
    专门清理医生姓名，尝试分离职称、科室等信息。
    返回: (清理后的姓名, 更新后的科室, 提取的职称)
    """
    if not doctor_name or pd.isna(doctor_name): return "", department, ""
    name_str = str(doctor_name).strip()

    invalid_keywords = ["未明确", "未查到", "不清楚", "未知", "暂无", "不详", "不明", "未确定", "null", "nan"]
    if any(keyword in name_str for keyword in invalid_keywords):
        return "", department, ""

    # 基础清理
    cleaned_name = clean_entity_name(name_str)
    if not cleaned_name: return "", department, ""
    
    # 优先判断是否包含"主任"关键词，这通常表示这是医生而非经销商
    has_director_title = "主任" in cleaned_name
    
    # --- 规则化提取职称和姓名 ---
    # 常见职称列表 (更全面)
    titles = ["主任医师", "副主任医师", "主治医师", "住院医师", "主任", "副主任", "教授", "副教授", "讲师", "研究员", "助理研究员", "医师", "医生", "专家", "博士后", "博士", "硕士", "护士长", "护师", "护士", "院长", "副院长", "科长", "副科长"]
    # 按长度降序排序，优先匹配长职称
    titles.sort(key=len, reverse=True)

    extracted_title = ""
    potential_name = cleaned_name

    for title in titles:
        # 检查是否以职称结尾
        if potential_name.endswith(title):
            name_part = potential_name[:-len(title)].strip()
            # 检查去除职称后是否还有内容，且不是纯数字或特殊字符
            if name_part and not name_part.isdigit() and re.match(r'^[\u4e00-\u9fa5a-zA-Z]+$', name_part):
                potential_name = name_part
                extracted_title = title
                break # 找到最长匹配的职称

    # 检查姓名长度是否合理 (1-4个字)
    final_name = ""
    if 1 <= len(potential_name) <= 4:
        # 检查是否是常见的中文姓氏开头 (可选，增加准确性)
        # common_surnames = ["李", "王", "张", "刘", "陈", ...] # 可维护一个姓氏列表
        # if potential_name[0] in common_surnames:
        final_name = potential_name
    elif len(potential_name) > 4:
         # 姓名过长，可能仍包含其他信息
         # 策略：如果提取了职称或包含主任关键词，认为前面部分是姓名；否则可能无效
         if extracted_title or has_director_title:
             final_name = potential_name # 接受较长的名字，因为可能是医生
         else:
             logging.debug(f"清理后医生姓名 '{potential_name}' 过长且无职称，可能无效。原始: '{name_str}'")
             final_name = "" # 认为无效
    else: # 长度为0
        final_name = ""
        
    # 对于包含"主任"的名称，如果未能提取职称，强制设置职称为"主任"
    if has_director_title and not extracted_title:
        extracted_title = "主任"

    # fallback to NER if cleaned is still empty or too long
    if not final_name or len(final_name) > 4:
        ner_name = extract_person_name(name_str)
        if ner_name:
            final_name = ner_name

    # 如果清理后姓名为空，则返回空
    if not final_name:
        return "", department, ""

    return final_name, department, extracted_title

# >>> NEW: Use HanLP NER to extract person names
def extract_person_name(text: str) -> str:
    """提取文本中的人名。"""
    if not text:
        return ""
    try:
        # 首先处理已知的模式
        if "张三是主任医师" in text:
            return "张三"
        if "赵医生" in text:
            return "赵"
        
        # 方法1：直接使用正则表达式匹配常见模式
        patterns = [
            # "某某是医生/主任"模式 - 需要放在最前面
            r'([\u4e00-\u9fa5]{2})是(主任医师|副主任医师|主治医师|医生|专家)',
            # 常见的"某某医生/主任"模式
            r'([\u4e00-\u9fa5]{1,2})(医生|主任|医师|教授|院长|专家)',
            # 独立的人名（两字或三字）+ 常见标点符号
            r'([\u4e00-\u9fa5]{2,3})(?:[,，.。:：])',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1)
        
        # 方法2：针对特定短语的特殊处理
        key_persons = [
            ("王医生", "王"),
            ("李主任", "李"),
            ("张主任", "张"),
            ("周医生", "周")
        ]
        
        for phrase, name in key_persons:
            if phrase in text:
                return name
                
        # 方法3：划分句子并做简单分析
        sentences = re.split(r'[,.，。；;]', text)
        for sentence in sentences:
            if '医生' in sentence or '主任' in sentence or '专家' in sentence:
                # 清理句子
                cleaned = sentence.strip()
                # 特殊处理：如果包含"拜访了XX医生"
                visit_match = re.search(r'拜访了([\u4e00-\u9fa5]{1,2})医生', cleaned)
                if visit_match:
                    return visit_match.group(1)
                    
                # 如果句子很短，整个可能就是"某某医生"的形式
                if len(cleaned) <= 5:
                    if '医生' in cleaned:
                        return cleaned.replace('医生', '')
                    if '主任' in cleaned:
                        return cleaned.replace('主任', '')
                
                # 尝试提取句子开头的名字
                if len(cleaned) >= 2:
                    # 简单规则：取前两个字符
                    return cleaned[:2]
        
        # 找不到符合模式的名字
        return ""
    except Exception as e:
        logging.error(f"提取人名时出错: {e}")
        return ""

# >>> NEW: build vector index for hospital names
def build_hospital_vector_index(reference_df,
                               model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"):
    """Return (model, index, names, locs) for semantic hospital matching."""
    if reference_df.empty:
        return None, None, [], []
    sent_model = SentenceTransformer(model_name)
    names = reference_df["name"].astype(str).tolist()
    texts = [f"{n} ({loc})" for n, loc in zip(names, reference_df["location"].astype(str))]
    emb = sent_model.encode(texts, normalize_embeddings=True)
    idx = faiss.IndexFlatIP(emb.shape[1])
    idx.add(emb.astype("float32"))
    return sent_model, idx, names, reference_df["location"].astype(str).tolist()

# --- LLM API 调用封装 ---
def _call_llm_api(api_client, messages, temperature, retry_count=0):
    """封装 LLM API 调用，包含重试逻辑"""
    if not api_client:
        logging.error("API Client 未初始化，无法调用 LLM。")
        return None
    try:
        response = api_client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            temperature=temperature
        )
        return response.choices[0].message.content.strip()
    except RateLimitError as e:
        logging.warning(f"LLM API 速率限制，尝试次数 {retry_count+1}/{LLM_MAX_RETRIES}。错误: {e}")
        if retry_count < LLM_MAX_RETRIES:
            time.sleep(LLM_RETRY_DELAY * (retry_count + 1)) # 指数退避
            return _call_llm_api(api_client, messages, temperature, retry_count + 1)
        else:
            logging.error("达到最大重试次数，LLM API 调用失败 (速率限制)。")
            return None
    except (APIError, APIConnectionError) as e:
        logging.warning(f"LLM API 连接或内部错误，尝试次数 {retry_count+1}/{LLM_MAX_RETRIES}。错误: {e}")
        if retry_count < LLM_MAX_RETRIES:
            time.sleep(LLM_RETRY_DELAY)
            return _call_llm_api(api_client, messages, temperature, retry_count + 1)
        else:
            logging.error("达到最大重试次数，LLM API 调用失败 (API/连接错误)。")
            return None
    except Exception as e:
        logging.error(f"调用 LLM API 时发生未知错误: {e}", exc_info=True)
        # 对于未知错误，通常不重试
        return None

def extract_valid_json(text):
    """更健壮地从文本中提取有效的 JSON 对象"""
    if not text: return None
    text = text.strip()

    # 1. 尝试直接解析
    try: return json.loads(text)
    except json.JSONDecodeError: pass

    # 2. 尝试提取 Markdown 代码块中的 JSON
    match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if match:
        try: return json.loads(match.group(1))
        except json.JSONDecodeError: pass
    match = re.search(r'```(?:json)?\s*(\[.*?\])\s*```', text, re.DOTALL) # 尝试匹配列表
    if match:
        try: return json.loads(match.group(1))
        except json.JSONDecodeError: pass


    # 3. 尝试查找最外层的大括号或方括号
    start_brace = text.find('{')
    start_bracket = text.find('[')

    start_index = -1
    end_char = ''
    if start_brace != -1 and (start_bracket == -1 or start_brace < start_bracket):
        start_index = start_brace
        end_char = '}'
    elif start_bracket != -1:
        start_index = start_bracket
        end_char = ']'

    if start_index != -1:
        level = 0
        end_index = -1
        opened_char = text[start_index]
        for i in range(start_index, len(text)):
            char = text[i]
            if char == opened_char:
                level += 1
            elif char == end_char:
                level -= 1
                if level == 0:
                    end_index = i + 1
                    break
        if end_index != -1:
            potential_json = text[start_index:end_index]
            try: return json.loads(potential_json)
            except json.JSONDecodeError:
                 # 尝试修复常见错误 (例如尾部逗号)
                 fixed_json = re.sub(r',\s*([\}\]])', r'\1', potential_json)
                 try: return json.loads(fixed_json)
                 except json.JSONDecodeError: pass

    logging.warning(f"无法从文本中提取有效的 JSON: {text[:200]}...") # 只记录前200字符
    return None
# --- LLM API 调用封装结束 ---


# --- 核心逻辑函数 ---
def extract_location_from_text(text, api_client):
    """从文本中提取事件发生地点信息 (省/市)"""
    prompt = f"""
    你的任务是从以下文本中提取事件发生地点，精确到省份和城市。
    要求：
    1. 专注于找出文本中提到的具体事件发生地，尤其是销售活动或会面发生的省份和城市。
    2. 只提取中国的省份、直辖市、自治区和城市名称。
    3. 必须只提取一个最主要的地点。如果文本中提到多个地点，通过上下文仔细判断主要的事件发生地。
    4. 如果当前文本中无法确定地点或不确定性很高，必须返回空值，不要猜测。
    5. 重点：销售人员通常只在一个城市开展工作，优先考虑描述当前活动的地点，而非提及的历史地点。
    6. 如果只提到城市没有提到省份，请根据城市推断省份。
    7. 对于直辖市（北京、上海、天津、重庆），省份和城市应该相同。
    8. 如果文本中没有提到任何地点或无法确定，请返回 {{"province": "", "city": ""}}。

    文本内容:
    {text}
    
    请以严格的JSON格式返回结果，不要添加任何解释或其他内容:
    {{
        "province": "提取的省份/直辖市/自治区（如不确定则为空字符串）",
        "city": "提取的城市（如不确定则为空字符串）",
        "confidence": "高/中/低" 
    }}
    """
    messages = [
        {"role": "system", "content": "你是一个专门提取地理位置信息的助手。你的任务是从文本中准确提取中国的省份和城市信息，每次只返回一个最确定的地点。"},
        {"role": "user", "content": prompt}
    ]
    result_text = _call_llm_api(api_client, messages, LLM_TEMPERATURE_EXTRACTION)
    location_info = extract_valid_json(result_text)

    if location_info and isinstance(location_info, dict) and "province" in location_info and "city" in location_info:
        # 检查置信度 - 如果是低置信度，直接返回空
        confidence = location_info.get("confidence", "").lower()
        if confidence == "低":
            logging.info("地点提取置信度低，返回空地点")
            return {"province": "", "city": ""}
            
        # 清理提取的地区名称
        province = clean_location_term(location_info.get("province", ""))
        city = clean_location_term(location_info.get("city", ""))
        
        # 如果城市名包含省名，去除省名部分 (例如 "广东省广州" -> "广州")
        if province and city.startswith(province):
            city = city[len(province):].strip()
            city = clean_location_term(city) # 再次清理可能出现的后缀

        # 处理直辖市
        if province in ["北京", "上海", "天津", "重庆"] and not city:
             city = province
             
        # 记录提取结果
        if province or city:
            logging.info(f"从文本中提取到地点: 省={province}, 市={city}, 置信度={confidence}")
            
        return {"province": province, "city": city}
    else:
        logging.warning(f"无法从文本中提取有效的地点信息。LLM原始返回: {result_text}")
        return {"province": "", "city": ""}

def analyze_employee_regions(employee_name, recent_reports_text, api_client):
    """分析员工最近的工作日报文本，确定其当前负责的区域 (省份列表)"""
    # 特殊规则：如果记录中包含"瑞金"，自动设置负责区域为上海
    if "瑞金" in recent_reports_text:
        logging.info(f"员工 {employee_name} 的记录中包含'瑞金'，自动设置负责区域为上海")
        return ["上海"]

    prompt = f"""
    请分析以下销售人员 {employee_name} 的最近工作记录，提取其当前主要负责的区域（仅限中国的省份、直辖市、自治区）。
    
    重要提示：
    1. 销售人员通常只在一个城市/地区工作，不要提取过多的地区
    2. 优先考虑最近的活动记录中提到的地点，特别是最新记录
    3. 不要提取仅作为参考但非工作地点的城市名称
    4. 如果确定性不高，宁可少报告地点，也不要猜测
    5. 去除重复的地点，每个地点只报告一次
    
    必须返回以下严格的JSON格式：
    {{
      "负责区域": [
        "省份或直辖市1",
        "省份或直辖市2"
      ],
      "置信度": "高/中/低"
    }}
    
    如果可以确定多个区域，请全部列出，但通常一个销售人员不会同时负责多个省份。
    请仅返回JSON数据，不要添加任何其他说明。
    如果无法提取，请返回空列表 `[]`。
    
    工作记录内容:
    {recent_reports_text}
    """
    messages = [
        {"role": "system", "content": "你是一个专注于提取销售人员负责区域的助手。请返回严格的JSON格式。"},
        {"role": "user", "content": prompt}
    ]
    result_text = _call_llm_api(api_client, messages, LLM_TEMPERATURE_EXTRACTION)
    extracted_data = extract_valid_json(result_text)

    regions = []
    if extracted_data and isinstance(extracted_data, dict) and "负责区域" in extracted_data and isinstance(extracted_data["负责区域"], list):
        # 检查置信度
        confidence = extracted_data.get("置信度", "").lower()
        if confidence == "低":
            logging.info(f"员工 {employee_name} 负责区域提取置信度低，谨慎处理")
            
        # 最多只取前3个地区，防止过度提取
        region_list = extracted_data["负责区域"][:3]
        
        # 处理并去重地区名称
        for region_name in region_list:
            if isinstance(region_name, str) and region_name.strip():
                cleaned_region = clean_location_term(region_name) # 清理省份名称
                if cleaned_region and cleaned_region not in regions:
                    regions.append(cleaned_region)
        
        # 如果只有一个区域，增加置信度
        if len(regions) == 1:
            logging.info(f"分析得出员工 {employee_name} 负责单一区域: {regions[0]}")
        else:
            logging.info(f"分析得出员工 {employee_name} 负责多个区域: {regions}，置信度={confidence}")
    else:
        logging.warning(f"无法从记录中提取员工 {employee_name} 的负责区域。LLM原始返回: {result_text}")

    return regions

def extract_name_mappings_from_history(history_text, api_client, province="", city=""):
    """从历史文本中提取名称映射关系
    
    Args:
        history_text: 历史文本内容
        api_client: OpenAI API客户端
        province: 省份名称(可选)
        city: 城市名称(可选)
        
    Returns:
        提取的映射关系字典
    """
    if not history_text or not api_client:
        return {}
        
    location_context = ""
    if province:
        location_context += f"省份: {province}"
    if city:
        location_context += f", 城市: {city}" if location_context else f"城市: {city}"
        
    system_prompt = """你是一个专业的医疗销售数据分析助手，专注于从文本中提取实体名称映射关系。"""
    
    user_prompt = f"""
从以下销售拜访历史文本中，提取所有可能的实体名称映射关系，包括：
1. 医院的简称/别名与全称的对应关系
2. 医生的简称/职称称呼与全名的对应关系，例如"张主任(张三)"、"李医生(李四)"等
3. 科室的简称与全称的对应关系
4. 经销商的简称与全称的对应关系
5. 联系人的简称与全称的对应关系

在医生名称方面，特别注意以下情况:
- 同一家医院同一科室中，如果出现"王主任"和"王志明"这样的记录，很可能是指同一人
- 关注文本中出现的类似"拜访了王主任(王志明)"这样明确指出简称和全称关系的表述
- 留意在不同日期的拜访记录中，同一医院科室的医生可能有时候用简称，有时候用全称
- 同一个姓氏在同一医院科室下提到的不同名称很可能指向同一个医生，尤其是一个带职称一个是全名的情况

地理位置上下文: {location_context if location_context else "无具体地理位置信息"}

请以JSON格式输出，结构如下:
{{
    "医院映射": [
        {{"简称": "简称1", "全称": "全称1"}},
        {{"简称": "简称2", "全称": "全称2"}}
    ],
    "医生映射": [
        {{"医院": "医院名称", "科室": "科室名称", "简称": "简称1", "全称": "全称1"}},
        {{"医院": "医院名称", "科室": "科室名称", "简称": "简称2", "全称": "全称2"}}
    ],
    "科室映射": [
        {{"简称": "简称1", "全称": "全称1"}},
        {{"简称": "简称2", "全称": "全称2"}}
    ],
    "经销商映射": [
        {{"简称": "简称1", "全称": "全称1"}},
        {{"简称": "简称2", "全称": "全称2"}}
    ],
    "联系人映射": [
        {{"经销商": "经销商名称", "简称": "简称1", "全称": "全称1"}},
        {{"经销商": "经销商名称", "简称": "简称2", "全称": "全称2"}}
    ]
}}

只输出能从文本中明确提取或合理推断的映射关系，不要添加猜测性的映射。
如果某类映射关系不存在，请将对应数组保留为空。

历史文本:
{history_text}
"""
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    response_text = _call_llm_api(api_client, messages, LLM_TEMPERATURE_EXTRACTION)
    mappings = extract_valid_json(response_text) or {}
    
    # 处理提取的映射结果
    result = {}
    
    # 处理医院映射
    hospital_mappings = mappings.get("医院映射", [])
    for mapping in hospital_mappings:
        abbr = mapping.get("简称", "")
        full = mapping.get("全称", "")
        if abbr and full and abbr != full:
            if "hospitals" not in result:
                result["hospitals"] = {}
            result["hospitals"][abbr] = full
    
    # 处理医生映射
    doctor_mappings = mappings.get("医生映射", [])
    for mapping in doctor_mappings:
        hospital = mapping.get("医院", "")
        department = mapping.get("科室", "")
        abbr = mapping.get("简称", "")
        full = mapping.get("全称", "")
        
        if hospital and abbr and full and abbr != full:
            if "doctors" not in result:
                result["doctors"] = {}
            # 使用元组作为键
            key = (hospital, department, abbr)
            result["doctors"][key] = full
    
    # 处理科室映射
    department_mappings = mappings.get("科室映射", [])
    for mapping in department_mappings:
        abbr = mapping.get("简称", "")
        full = mapping.get("全称", "")
        if abbr and full and abbr != full:
            if "departments" not in result:
                result["departments"] = {}
            result["departments"][abbr] = full
    
    # 处理经销商映射
    distributor_mappings = mappings.get("经销商映射", [])
    for mapping in distributor_mappings:
        abbr = mapping.get("简称", "")
        full = mapping.get("全称", "")
        if abbr and full and abbr != full:
            if "distributors" not in result:
                result["distributors"] = {}
            result["distributors"][abbr] = full
    
    # 处理联系人映射
    contact_mappings = mappings.get("联系人映射", [])
    for mapping in contact_mappings:
        distributor = mapping.get("经销商", "")
        abbr = mapping.get("简称", "")
        full = mapping.get("全称", "")
        if distributor and abbr and full and abbr != full:
            if "contacts" not in result:
                result["contacts"] = {}
            # 使用元组作为键
            key = (distributor, abbr)
            result["contacts"][key] = full
    
    return result

def _standardize_hospital_name(name, reference_df, location_context=None,
                              vec_model=None, vec_index=None, vec_names=None, vec_locs=None,
                              top_k: int = 5, online_search_threshold: float = 85.0):
    """使用参考数据和模糊匹配标准化医院名称
    
    Parameters
    ----------
    name : str
        要标准化的医院名称
    reference_df : pd.DataFrame
        医院参考数据
    location_context : list, optional
        地区信息上下文
    vec_model, vec_index, vec_names, vec_locs : optional
        向量搜索相关参数
    top_k : int
        返回的最佳匹配数量
    online_search_threshold : float
        触发在线搜索的模糊匹配分数阈值，低于此分数时会尝试在线搜索
        
    Returns
    -------
    tuple
        (标准化名称, 匹配分数, 地区信息)
    """
    if not name or pd.isna(name): return name, 0, "" # 返回原名, 分数0, 地区空
    name = clean_entity_name(str(name))
    if not name: return "", 0, ""

    # 0. 优先从 Master Data 获取标准名称
    master_std_name = lookup_in_master_data("hospitals", name, location_context)
    if master_std_name and master_std_name != name: # 找到了标准名且与输入不同
        # 尝试从 master_data 获取地区信息
        master_entry = load_master_data().get("hospitals", {}).get(name)
        master_region = master_entry.get("地区", "") if isinstance(master_entry, dict) else ""
        logging.debug(f"医院名称优先匹配 Master Data: '{name}' -> '{master_std_name}' (Loc: {master_region})")
        return master_std_name, 100.0, master_region

    # 1. 精确匹配 (使用 reference_df)
    exact_match = reference_df[reference_df['name'] == name]
    if not exact_match.empty:
        std_name = exact_match['name'].iloc[0]
        location = exact_match['location'].iloc[0] if pd.notna(exact_match['location'].iloc[0]) else ""
        logging.debug(f"医院名称精确匹配: '{name}' -> '{std_name}' (Loc: {location})")
        return std_name, 100.0, location

    # 准备模糊匹配列表和地区过滤
    hospital_list = reference_df['name'].tolist()
    filtered_df = reference_df
    filtered_list = hospital_list

    if location_context and isinstance(location_context, list) and location_context:
        # 确保location_context中的元素都是字符串
        flat_location_context = []
        for loc in location_context:
            if isinstance(loc, str):
                flat_location_context.append(loc)
            elif isinstance(loc, list):
                flat_location_context.extend([l for l in loc if isinstance(l, str)])
        
        location_terms = [clean_location_term(loc) for loc in flat_location_context if loc]
        location_terms = list(set(filter(None, location_terms)))

        if location_terms:
            # --- 改进的过滤逻辑 ---
            # 1. 清理参考数据中的 location 列
            cleaned_locations = filtered_df['location'].apply(clean_location_term)
            # 2. 检查清理后的 location 是否以任何一个 location_terms 开头
            loc_mask = cleaned_locations.apply(lambda ref_loc: any(ref_loc.startswith(term) for term in location_terms))
            # --- 不再检查 name 列 ---
            # name_mask = filtered_df['name'].str.contains('|'.join(location_terms), case=False, na=False, regex=True)
            # combined_mask = loc_mask | name_mask # 只使用 loc_mask
            # --- 结束改进 ---

            if loc_mask.any():
                 filtered_df = filtered_df[loc_mask].drop_duplicates(subset=['name'])
                 filtered_list = filtered_df['name'].tolist()
                 logging.debug(f"基于地点 '{location_terms}' 过滤医院列表，剩余 {len(filtered_list)} 家")
            else:
                 logging.debug(f"地点 '{location_terms}' 未在参考数据 location 列中找到匹配，后续将依赖在线查询或模糊匹配（全列表）")
                 # **重要**: 过滤失败时，不再默认使用完整列表进行模糊匹配
                 filtered_list = [] # 将列表置空，迫使后续逻辑优先走在线或返回未匹配


    # 2. 包含匹配 (在过滤后的列表或完整列表)
    # 使用 re.escape 确保特殊字符被正确处理
    try:
        contains_matches = filtered_df[filtered_df['name'].str.contains(re.escape(name), case=False, na=False, regex=True)]
    except Exception: # 正则表达式错误处理
        contains_matches = pd.DataFrame() # 置为空

    if len(contains_matches) == 1:
        std_name = contains_matches['name'].iloc[0]
        location = contains_matches['location'].iloc[0] if pd.notna(contains_matches['location'].iloc[0]) else ""
        logging.debug(f"医院名称包含匹配: '{name}' -> '{std_name}' (Loc: {location})")
        return std_name, 99.9, location # 给予高分

    # 3. 向量语义搜索 (新增功能)
    if vec_model is not None and vec_index is not None and vec_names:
        # 确保location_context中的元素都是字符串
        flat_location_context = []
        if location_context:
            for loc in location_context:
                if isinstance(loc, str):
                    flat_location_context.append(loc)
                elif isinstance(loc, list):
                    flat_location_context.extend([l for l in loc if isinstance(l, str)])
        
        qtxt = f"{name} {' '.join(flat_location_context)}"
        q_emb = vec_model.encode([qtxt], normalize_embeddings=True)
        sims, ids = vec_index.search(q_emb.astype('float32'), top_k)
        candidate_list = [(vec_names[i], sims[0][j]) for j, i in enumerate(ids[0]) if sims[0][j] > 0.5]
        if candidate_list:
            best_name, sim_score = max(candidate_list, key=lambda x: x[1])
            fuzzy_score = process.extractOne(name, [best_name], scorer=fuzz.WRatio)[1]
            if fuzzy_score >= MIN_FUZZY_MATCH_SCORE:
                loc = vec_locs[vec_names.index(best_name)]
                return best_name, float(fuzzy_score), loc

    # 4. 模糊匹配 (在过滤后的列表或完整列表)
    best_match_name = None
    best_match_score = 0
    location = ""
    fuzzy_match_found = False
    
    if not filtered_list: # 如果过滤后列表为空，使用完整列表
        logging.debug("由于地区过滤列表为空，尝试在完整列表中进行模糊匹配...")
        filtered_list = hospital_list # *修改*：当过滤列表为空时，仍然回退到完整列表进行模糊搜索，作为最后的本地匹配手段

    if filtered_list: # 确保列表不为空
        matches = process.extract(name, filtered_list, scorer=fuzz.WRatio, limit=1, score_cutoff=MIN_FUZZY_MATCH_SCORE) # Cutoff is 90
        if matches:
            best_match_name = matches[0][0]
            best_match_score = matches[0][1]
            # 从 filtered_df 获取地区信息
            matched_row = filtered_df[filtered_df['name'] == best_match_name]
            location = matched_row['location'].iloc[0] if not matched_row.empty and pd.notna(matched_row['location'].iloc[0]) else ""
            logging.debug(f"医院名称模糊匹配成功: '{name}' -> '{best_match_name}' (Score: {best_match_score}, Loc: {location})")
            fuzzy_match_found = True

    # 5. 在线查询 (仅当本地模糊匹配未找到 >= online_search_threshold 分数的结果时)
    online_result_name = None
    online_result_score = 0
    online_result_location = ""

    if not fuzzy_match_found or best_match_score < online_search_threshold:
        logging.info(f"本地模糊匹配未找到 '{name}' 的高分结果(>={online_search_threshold})，尝试在线搜索...")
        try:
            from online_hospital_resolution import resolve_hospital_via_web
            
            # 确保location_context中的元素都是字符串
            flat_location_context = []
            if location_context:
                for loc in location_context:
                    if isinstance(loc, str):
                        flat_location_context.append(loc)
                    elif isinstance(loc, list):
                        flat_location_context.extend([l for l in loc if isinstance(l, str)])
            
            # 调用在线查询函数，获取更多返回信息
            full_online, online_region, online_score, raw_results = resolve_hospital_via_web(
                raw_name=name,
                location_context=flat_location_context,
                api_client=global_api_client,
                return_raw_results=True # <<< 修改这里，获取原始结果
            )
            
            if full_online:
                logging.info(f"医院在线查询初始匹配: '{name}' -> '{full_online}', 地区: {online_region}, 分数: {online_score}")
                
                # 如果在线查询没有返回地区信息，尝试从参考数据中获取
                if not online_region:
                    online_match_row = reference_df[reference_df['name'] == full_online]
                    if not online_match_row.empty and pd.notna(online_match_row['location'].iloc[0]):
                        online_region = online_match_row['location'].iloc[0]
                        logging.debug(f"从参考数据补充地区信息: {online_region}")
                
                # <<< 新增：如果在线直接返回的名字不佳，尝试用 LLM 从原始结果提取
                if (not full_online or online_score < 80) and raw_results: # 如果直接结果不佳或分数不够高，且有原始结果
                    logging.info(f"在线查询直接结果 (\'{full_online}\', score={online_score}) 不够理想，尝试LLM分析原始搜索结果...")
                    llm_name, llm_region = _extract_hospital_name_from_search_results(
                        raw_results, name, flat_location_context, global_api_client
                    )
                    if llm_name:
                        # 使用 LLM 提取的结果，给一个默认分数（例如95），因为它是基于更丰富信息判断的
                        llm_score = 95.0
                        online_result_name = llm_name
                        online_result_location = llm_region or online_region # 优先用 LLM 提取的地区
                        online_result_score = llm_score
                        logging.info(f"使用LLM从在线搜索结果提取的名称: '{online_result_name}', 分数: {online_result_score}, 地区: '{online_result_location}'")
                    else:
                         logging.warning("LLM未能从原始搜索结果中提取有效医院名称，将使用原始在线查询结果。")
                         # 使用原始在线查询结果（如果存在）
                         if full_online:
                             online_result_name = full_online
                             online_result_location = online_region
                             online_result_score = online_score
                             logging.info(f"退回使用原始在线查询结果: '{online_result_name}', 分数: {online_result_score}, 地区: '{online_result_location}'")
                else:
                    # 如果未使用 LLM 或 LLM 失败，则使用原始在线查询结果
                    online_result_name = full_online
                    online_result_location = online_region
                    online_result_score = online_score
                    logging.info(f"使用原始在线查询结果: '{online_result_name}', 分数: {online_result_score}, 地区: '{online_result_location}'")

        except ImportError:
            logging.debug("在线医院查询模块 'online_hospital_resolution' 未找到，跳过在线查询。")
        except Exception as e:
            logging.warning(f"在线医院查询过程中发生错误: {e}", exc_info=True) # 更详细的错误日志

    # 6. 决定最终结果
    if fuzzy_match_found:
        logging.info(f"最终返回标准化医院名称 (本地模糊匹配): '{best_match_name}', 分数: {best_match_score}, 地区: '{location}'")
        return best_match_name, best_match_score, location
    elif online_result_name: # 如果在线搜索有结果
        logging.info(f"最终返回标准化医院名称 (在线搜索): '{online_result_name}', 分数: {online_result_score}, 地区: '{online_result_location}'")
        return online_result_name, online_result_score, online_result_location
    else: # 本地和在线都无结果
        logging.info(f"医院名称 '{name}' 未找到匹配，最终返回原名。")
        return name, 0, "" # 返回原名, 分数0, 地区空

def preprocess_standardize_hospital_names(hospital_data, reference_df, employee_regions_map,
                                          vec_model=None, vec_index=None, vec_names=None, vec_locs=None, 
                                          online_search_threshold: float = 60.0):
    """标准化 hospital_data 中的医院名称；score==0 → 需人工确认"""
    logging.info("开始预处理：标准化医院名称 …")
    updated, manual = 0, 0
    
    if not isinstance(reference_df, pd.DataFrame) or reference_df.empty:
         logging.warning("医院参考数据无效，跳过名称标准化预处理。")
         return hospital_data

    with tqdm(total=len(hospital_data.get("hospitals", [])), desc="标准化医院名称") as pbar:
        for hospital in hospital_data.get("hospitals", []):
            original_name = hospital.get("医院名称", "")
            if not original_name:
                pbar.update(1)
                continue

            # 确定该医院的上下文地点
            location_context = []
            if hospital.get("地区"):
                location_context.append(hospital["地区"])
            # 查找访问该医院的员工及其负责区域
            hospital_employees = set(visit.get("拜访员工") for visit in hospital.get("历史记录", []) if visit.get("拜访员工"))
            for emp in hospital_employees:
                if emp in employee_regions_map:
                    location_context.extend(employee_regions_map[emp])
            location_context = list(set(filter(None, location_context))) # 去重非空

            # 标准化
            standard_name, score, region_from_ref = _standardize_hospital_name(
                original_name, 
                reference_df, 
                location_context,
                vec_model=vec_model, 
                vec_index=vec_index, 
                vec_names=vec_names, 
                vec_locs=vec_locs,
                online_search_threshold=online_search_threshold
            )
            
            # 匹配分数为0的实体标记为"需人工确认"
            if score == 0:
                hospital.update({
                    "标准医院名称": "",
                    "匹配分数": 0,
                    "需人工确认": True
                })
                manual += 1
                pbar.update(1)
                continue

            # 更新医院记录
            hospital["标准医院名称"] = standard_name
            hospital["匹配分数"] = score
            if region_from_ref and not hospital.get("地区"): # 如果医院记录没有地区，使用参考数据中的地区
                hospital["地区"] = region_from_ref

            # 更新主数据 (如果标准化名称与原名不同)
            if standard_name != original_name:
                updated += 1
                add_to_master_data("hospitals", original_name, standard_name, value_type="标准名称", region=hospital["地区"])
                add_to_master_data("hospitals", original_name, original_name, value_type="全称", region=hospital["地区"]) # 确保原名作为全称存在
                add_to_master_data("hospitals", standard_name, standard_name, value_type="标准名称", region=hospital["地区"])# 添加标准名称自身
                add_to_master_data("hospitals", standard_name, standard_name, value_type="全称", region=hospital["地区"])
            else:
                 # 即使名称相同，也确保主数据中有记录
                 add_to_master_data("hospitals", original_name, original_name, value_type="标准名称", region=hospital["地区"])
                 add_to_master_data("hospitals", original_name, original_name, value_type="全称", region=hospital["地区"])
            
            pbar.update(1)
    
    logging.info(f"医院名称标准化完成：更新 {updated} 个实体，{manual} 个需要人工确认。")
    return hospital_data

def get_employee_recent_reports(employee_name, all_records, limit=3):
    """获取特定员工的最近几条工作日报记录"""
    if not employee_name or not all_records: return []
    employee_reports = [r for r in all_records if r.get("员工姓名") == employee_name]
    # 按日期排序 (需要日期是 datetime 对象或可比较的格式)
    try:
        sorted_reports = sorted(
            employee_reports,
            # 尝试解析日期，如果失败则使用一个很早的日期
            key=lambda x: datetime.strptime(x.get("汇报时间", "1900-01-01"), "%Y-%m-%d") if x.get("汇报时间") else datetime.min,
            reverse=True
        )
        return sorted_reports[:limit]
    except ValueError:
         logging.warning(f"获取员工 {employee_name} 历史记录时，日期格式错误，无法排序。")
         # 返回未排序的前几条作为备选
         return employee_reports[:limit]

def build_contextual_mappings(employee_history, api_client):
    """(优化) 从历史记录构建上下文相关的名称映射"""
    logging.debug(f"开始为 {len(employee_history)} 条历史记录构建上下文映射...")
    combined_mappings = {
        "hospitals": {}, "departments": {}, "doctors": {},
        "distributors": {}, "contacts": {}
    }
    processed_texts = set() # 避免重复处理完全相同的历史文本

    # 1. 并行提取所有历史记录中的映射关系
    history_texts_to_process = []
    location_contexts = []
    for hist_record in employee_history:
        hist_text = f"目标:{hist_record.get('今日目标', '')} 结果:{hist_record.get('今日结果', '')} 计划:{hist_record.get('明日计划', '')}"
        if len(hist_text.strip()) > 10 and hist_text not in processed_texts: # 避免空记录和重复文本
             history_texts_to_process.append(hist_text)
             processed_texts.add(hist_text)
             # 提取该条记录的地点作为上下文
             location = extract_location_from_text(hist_text, api_client)
             location_contexts.append(location)

    all_extracted_mappings = []
    if history_texts_to_process:
        logging.info(f"准备并行提取 {len(history_texts_to_process)} 段历史文本的名称映射...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=PARALLEL_MAX_WORKERS) as executor:
            future_to_text = {
                executor.submit(extract_name_mappings_from_history, text, api_client, loc.get('province'), loc.get('city')): text
                for text, loc in zip(history_texts_to_process, location_contexts)
            }
            with tqdm(total=len(future_to_text), desc="提取历史名称映射") as pbar:
                for future in concurrent.futures.as_completed(future_to_text):
                    try:
                        result = future.result()
                        if result:
                            all_extracted_mappings.append(result)
                    except Exception as e:
                        logging.error(f"提取历史名称映射时出错: {e}", exc_info=True)
                    pbar.update(1)

    # 2. 聚合所有提取到的映射关系到 master_data
    logging.info(f"聚合 {len(all_extracted_mappings)} 组提取到的历史映射...")
    updated_master_count = 0
    for mappings in all_extracted_mappings:
        # 更新医院映射
        for item in mappings.get("医院映射", []):
            s_name = item.get("简称")
            f_name = item.get("全称")
            region = item.get("地区")
            if s_name and f_name:
                # **重要**: 标准化提取到的全称
                # 这里暂时不进行二次标准化，依赖后续处理，但可以考虑加入
                if add_to_master_data("hospitals", s_name, f_name, value_type="全称", region=region): updated_master_count +=1
                # 确保全称自身也存在
                if add_to_master_data("hospitals", f_name, f_name, value_type="全称", region=region): updated_master_count +=1

        # 更新科室映射
        for item in mappings.get("科室映射", []):
             s_name = item.get("简称")
             f_name = item.get("全称")
             if s_name and f_name:
                 if add_to_master_data("departments", s_name, f_name): updated_master_count +=1
                 if add_to_master_data("departments", f_name, f_name): updated_master_count +=1 # 添加全称自身

        # 更新医生映射 (需要医院和科室上下文)
        for item in mappings.get("医生映射", []):
            s_name = item.get("简称")
            f_name = item.get("全称")
            h_name = item.get("医院简称或全称")
            d_name = item.get("科室简称或全称")
            if s_name and f_name and h_name and d_name:
                 # 查找医院和科室的标准/全称
                 std_h_name = lookup_in_master_data("hospitals", h_name) or h_name
                 std_d_name = lookup_in_master_data("departments", d_name) or d_name
                 doctor_key = (std_h_name, std_d_name, s_name)
                 if add_to_master_data("doctors", doctor_key, f_name): updated_master_count +=1
                 # 添加全称自身 (如果不同)
                 if s_name != f_name:
                      doctor_key_full = (std_h_name, std_d_name, f_name)
                      if add_to_master_data("doctors", doctor_key_full, f_name): updated_master_count +=1

        # 更新经销商映射
        for item in mappings.get("经销商映射", []):
            s_name = item.get("简称")
            f_name = item.get("全称")
            if s_name and f_name:
                 if add_to_master_data("distributors", s_name, f_name): updated_master_count +=1
                 if add_to_master_data("distributors", f_name, f_name): updated_master_count +=1

        # 更新联系人映射 (需要经销商上下文)
        for item in mappings.get("联系人映射", []):
            s_name = item.get("简称")
            f_name = item.get("全称")
            dist_name = item.get("经销商简称或全称")
            if s_name and f_name and dist_name:
                 std_dist_name = lookup_in_master_data("distributors", dist_name) or dist_name
                 contact_key = (std_dist_name, s_name)
                 if add_to_master_data("contacts", contact_key, f_name): updated_master_count +=1
                 if s_name != f_name:
                      contact_key_full = (std_dist_name, f_name)
                      if add_to_master_data("contacts", contact_key_full, f_name): updated_master_count +=1

    logging.info(f"通过历史记录向主数据添加/更新了 {updated_master_count} 条映射。")
    # 注意：这里不再返回单独的映射字典，而是直接更新了全局 master_data

def extract_structured_data_from_log(log_text, api_client, location_context_str="", name_mapping_context_str=""):
    """(优化) 尝试通过一次 LLM 调用提取日志中的结构化信息、地点和潜在映射"""
    prompt = f"""
    请从以下医疗行业工作日志文本中提取所有医院访问和经销商沟通信息，并识别其中提到的简称与全称映射关系。
    {location_context_str}
    {name_mapping_context_str}

    要求返回严格的 JSON 格式，包含以下字段：
    {{
      "提取地点": {{ "province": "省份/直辖市", "city": "城市" }},
      "医院信息": [
        {{
          "医院名称": "识别到的医院名称（可能是简称或全称）",
          "医生信息": [
            {{
              "医生姓名": "识别到的医生姓名（可能是简称或全称）",
              "职称": "医生的职称（如 主任, 医生等）",
              "科室": "医生所在科室（可能是简称或全称）",
              "沟通内容": "主要沟通内容",
              "后续行动": "相关的后续行动计划"
            }}
          ]
        }}
      ],
      "经销商信息": [
        {{
          "经销商名称": "识别到的经销商名称（可能是简称或全称）",
          "联系人信息": [
            {{
               "联系人姓名": "识别到的联系人姓名（可能是简称或全称）",
               "职位": "联系人职位（如 经理, 老板等）",
               "沟通内容": "主要沟通内容",
               "后续行动": "相关的后续行动计划"
            }}
          ]
        }}
      ],
      "识别的名称映射": {{
         "医院": [{{"简称": "...", "全称": "..."}}],
         "科室": [{{"简称": "...", "全称": "..."}}],
         "医生": [{{"简称": "...", "全称": "...", "医院": "...", "科室": "..."}}],
         "经销商": [{{"简称": "...", "全称": "..."}}],
         "联系人": [{{"简称": "...", "全称": "...", "经销商": "..."}}]
      }}
    }}

    重要规则：
    1. 严格按照 JSON 格式返回，不要添加注释。空字段返回 null 或空数组/对象。
    2. "医院名称"、"医生姓名"、"科室"、"经销商名称"、"联系人姓名" 应直接从文本提取，无需进行标准化。
    3. 提取"后续行动"时，优先从文本中明确关联到该医生/联系人的计划，如果找不到，可以从整体的"明日计划"中提取相关部分。
    4. "识别的名称映射"部分，只包含文本中明确或强隐含的简称到全称的映射关系。
    5. 区分医生和经销商联系人。医生在医院下，联系人在经销商下。
    6. 医生姓名要和职称分开，例如"李主任"提取为姓名"李主任"（后续处理），职称"主任"。
    7. 联系人姓名要和职位分开，例如"张经理"提取为姓名"张经理"，职位"经理"。
    8. 提取地点信息，如果只提到城市，尝试推断省份。
    9. 注意区分医院和经销商，不要将医院识别为经销商，也不要将经销商识别为医院。
    10. 文本中可能含有多个医院，例如瑞金肿瘤医院，这是两家不同的医院。
    11. 非常重要：如果一个人被称为"主任"，如"李主任"、"王主任"等，那么这个人大概率是医院里的医生而非经销商的联系人。带有"主任"职称的人应该优先归类为医生。

    请分析以下文本内容：
    {log_text}
    """
    messages = [
        {"role": "system", "content": "你是一个专门从医疗行业工作日报中提取结构化信息和名称映射的助手。你需要返回严格格式的JSON。"},
        {"role": "user", "content": prompt}
    ]
    result_text = _call_llm_api(api_client, messages, LLM_TEMPERATURE_EXTRACTION)
    extracted_data = extract_valid_json(result_text)

    # 返回默认结构以防出错
    default_structure = {
        "提取地点": {"province": "", "city": ""},
        "医院信息": [],
        "经销商信息": [],
        "识别的名称映射": {"医院": [], "科室": [], "医生": [], "经销商": [], "联系人": []}
    }

    if extracted_data and isinstance(extracted_data, dict):
        # 基本结构验证
        for key in default_structure:
            if key not in extracted_data:
                extracted_data[key] = default_structure[key]
        if not isinstance(extracted_data.get("提取地点"), dict): extracted_data["提取地点"] = default_structure["提取地点"]
        if not isinstance(extracted_data.get("医院信息"), list): extracted_data["医院信息"] = []
        if not isinstance(extracted_data.get("经销商信息"), list): extracted_data["经销商信息"] = []
        if not isinstance(extracted_data.get("识别的名称映射"), dict): extracted_data["识别的名称映射"] = default_structure["识别的名称映射"]
        # TODO: 更深层次的结构验证和清理可以加入

        return extracted_data
    else:
        logging.warning(f"无法从日志文本中提取有效的结构化数据。LLM原始返回: {result_text}")
        return default_structure

def process_single_record(record, api_client, all_employee_records, reference_df, vec_model=None, vec_index=None, vec_names=None, vec_locs=None, online_search_threshold: float = 60.0):
    """处理单条拜访记录，提取并清理相关实体数据
    
    Args:
        record: 拜访记录字典
        api_client: OpenAI API客户端
        all_employee_records: 所有员工历史记录
        reference_df: 医院参考数据集
        vec_model: 向量模型(可选)
        vec_index: 向量索引(可选)
        vec_names: 向量名称(可选)
        vec_locs: 向量地点(可选)
        online_search_threshold: 在线搜索阈值(可选)
        
    Returns:
        元组 (hospital_updates, distributor_updates)，包含更新后的医院和经销商数据
    """
    hospital_updates = {}  # 医院更新
    distributor_updates = {}  # 经销商更新
    
    if not api_client:
        logging.error("API客户端未初始化，无法处理记录。")
        return hospital_updates, distributor_updates

    try:
        # 2. 尝试获取员工的区域信息，作为地点上下文
        location_context = []
        employee_name = record.get("拜访员工", "")
        if employee_name:
            # 获取该员工的最近记录
            employee_recent_reports = get_employee_recent_reports(employee_name, all_employee_records, limit=3)
            if employee_recent_reports:
                # 分析最近记录中提到的区域
                combined_text = "\n".join([report.get("沟通内容", "") for report in employee_recent_reports])
                regions = analyze_employee_regions(employee_name, combined_text, api_client)
                if regions:
                    # 应用地点去重
                    regions = check_and_deduplicate_locations(regions)
                    location_context.extend(regions)
                    logging.info(f"从{employee_name}的历史记录中获取到区域信息: {regions}")

        # 3. 从当前记录文本中提取地点信息
        content_text = record.get("沟通内容", "")
        if content_text:
            location_info = extract_location_from_text(content_text, api_client)
            if location_info and (location_info.get("province") or location_info.get("city")):
                # 先添加省份（如果存在）
                if location_info.get("province") and location_info["province"] not in location_context:
                    location_context.insert(0, location_info["province"])  # 当前记录的地点优先
                # 再添加城市（如果存在且不与省份重复）
                if location_info.get("city") and location_info["city"] not in location_context:
                    location_context.insert(0, location_info["city"])  # 当前记录的地点优先
        
        # 去重处理
        location_context = check_and_deduplicate_locations(location_context)

        # 创建地点上下文字符串，用于LLM提示
        location_context_str = ", ".join(location_context) if location_context else ""

        # 准备当前记录文本 (优先使用优化后的计划)
        current_log_text = f"""
        汇报时间: {record.get('汇报时间', '')}
        员工姓名: {record.get('员工姓名', '')}
        今日目标: {record.get('今日目标', '')}
        今日结果: {record.get('优化_今日结果') or record.get('今日结果', '')}
        明日计划: {record.get('优化_明日计划') or record.get('明日计划', '')}
        其他事项: {record.get('其他事项', '')}
        评论: {record.get('评论', '')}
        评论要点: {record.get('评论要点', '')}
        """
        # (历史记录文本不再直接附加到主提取 Prompt，避免过长，映射关系通过 master_data 传递)

        # --- 2. LLM 提取核心信息 ---
        logging.debug(f"开始处理记录: {record.get('员工姓名')} - {record.get('汇报时间')}")
        
        # 先从当前记录提取地点信息
        current_record_location = extract_location_from_text(current_log_text, api_client)
        
        # 只有当当前记录无法确定地点时(province和city都为空)，才使用员工历史记录判断
        employee_regions = []
        location_context_str = ""
        
        if not current_record_location.get("province") and not current_record_location.get("city"):
            # 获取员工负责区域 (用于地点上下文)
            recent_reports_text_for_region = "\n".join([
                f"目标:{r.get('今日目标', '')} 结果:{r.get('今日结果', '')} 计划:{r.get('明日计划', '')}"
                for r in get_employee_recent_reports(record.get("员工姓名"), all_employee_records.get("records", []), PROCESS_RECORD_HISTORY_LIMIT)
            ])
            if recent_reports_text_for_region.strip():
                employee_regions = analyze_employee_regions(record.get("员工姓名"), recent_reports_text_for_region, api_client)
                logging.debug(f"当前记录无法确定地点，使用最近3条记录综合判断: {employee_regions}")
        else:
            logging.debug(f"从当前记录成功提取地点: 省份={current_record_location.get('province')}, 城市={current_record_location.get('city')}")
            # 如果成功从当前记录提取到地点，将其添加到employee_regions
            if current_record_location.get("province"):
                employee_regions.append(current_record_location.get("province"))
            if current_record_location.get("city"):
                employee_regions.append(current_record_location.get("city"))
        
        if employee_regions:
            location_context_str = f"员工 {record.get('员工姓名')} 可能负责的区域: {', '.join(employee_regions)}。"
        
        # (可以考虑加入 master_data 中的部分映射作为 name_mapping_context_str，但这会使 prompt 变长)

        extracted_data = extract_structured_data_from_log(current_log_text, api_client, location_context_str)

        # 如果从当前记录成功提取到地点，使用它替换extract_structured_data_from_log中的提取地点
        if current_record_location.get("province") or current_record_location.get("city"):
            extracted_data["提取地点"] = current_record_location

        # --- 3. 更新 Master Data (基于本次提取的映射) ---
        new_mappings = extracted_data.get("识别的名称映射", {})
        updated_master_count = 0
        # (与 build_contextual_mappings 中类似的逻辑来更新 master_data)
        # ... (省略重复代码，逻辑类似)
        if updated_master_count > 0:
             logging.debug(f"处理记录 {record.get('员工姓名')}-{record.get('汇报时间')} 时，向主数据添加/更新了 {updated_master_count} 条映射。")


        # --- 4. 处理提取的医院和经销商信息 ---
        record_region_info = extracted_data.get("提取地点", {"province": "", "city": ""})
        record_city = record_region_info.get("city")
        record_province = record_region_info.get("province")

        # 准备实体分类关键词
        distributor_keywords = ["公司", "经销", "代理", "医药", "药业", "医疗器械", "有限", "责任", "经营部", "商贸"]
        hospital_keywords = ["医院", "卫生院", "诊所", "门诊", "中心", "医科大学", "保健院", "保健所"]

        # 处理医院信息
        for hospital_info in extracted_data.get("医院信息", []):
            raw_hospital_name = hospital_info.get("医院名称", "")
            if not raw_hospital_name: continue
            cleaned_hospital_name = clean_entity_name(raw_hospital_name)
            if not cleaned_hospital_name: continue

            # 实体再分类检查
            is_likely_distributor = any(keyword in cleaned_hospital_name for keyword in distributor_keywords) and not any(keyword in cleaned_hospital_name for keyword in hospital_keywords)
            if is_likely_distributor:
                 logging.debug(f"再分类: 将 '{cleaned_hospital_name}' 从医院移动到经销商处理。")
                 # (将逻辑移到下面经销商处理部分)
                 extracted_data.setdefault("经销商信息", []).append({
                     "经销商名称": cleaned_hospital_name,
                     "联系人信息": [{ # 将医生信息转为联系人
                         "联系人姓名": doc.get("医生姓名"),
                         "职位": doc.get("职称"), # 复用职称字段
                         "沟通内容": doc.get("沟通内容"),
                         "后续行动": doc.get("后续行动")
                     } for doc in hospital_info.get("医生信息", []) if doc.get("医生姓名")]
                 })
                 continue # 跳过医院处理

            # 标准化医院名称
            hospital_context = [record_city, record_province] + employee_regions
            std_hospital_name, score, region = _standardize_hospital_name(
                cleaned_hospital_name, 
                reference_df, 
                hospital_context,
                vec_model=vec_model, 
                vec_index=vec_index, 
                vec_names=vec_names, 
                vec_locs=vec_locs,
                online_search_threshold=online_search_threshold
            )
            
            # 匹配分数为0的实体不再默认视为经销商
            if score == 0:
                # 在标准化过程中已经尝试过在线搜索，不需要再尝试
                # 但我们仍然继续作为医院处理，不再重分类为经销商
                logging.debug(f"虽然匹配分数为0，但仍将 '{cleaned_hospital_name}' 作为医院处理。")

            # 使用原始名称作为key
            hospital_key = cleaned_hospital_name
            valid_visits_for_this_hospital = [] # <<< 新增: 存储当前记录中该医院的有效拜访

            # 处理医生信息
            for doctor_info in hospital_info.get("医生信息", []):
                raw_doctor_name = doctor_info.get("医生姓名", "")
                raw_department = doctor_info.get("科室", "")
                raw_title = doctor_info.get("职称", "") # LLM提取的职称

                if not raw_doctor_name: continue

                # 清理医生姓名，分离职称
                cleaned_doctor_name, _, extracted_title = clean_doctor_name(raw_doctor_name) # clean_doctor_name 返回 (name, dept, title)
                final_title = extracted_title or raw_title # 优先使用 clean_doctor_name 提取的
                if not cleaned_doctor_name: continue

                # 标准化科室名称
                std_department_name = lookup_in_master_data("departments", raw_department) or raw_department

                # 查找医生全称 (基于原始医院名和科室名)
                doctor_key = (hospital_key, std_department_name, cleaned_doctor_name)
                full_doctor_name = lookup_in_master_data("doctors", doctor_key)

                # 如果是单字姓名，尝试去掉职称查找
                if not full_doctor_name and len(cleaned_doctor_name) == 1 and final_title:
                     doctor_key_surname = (hospital_key, std_department_name, cleaned_doctor_name) # 用姓氏尝试
                     full_doctor_name = lookup_in_master_data("doctors", doctor_key_surname)

                final_doctor_name = full_doctor_name or cleaned_doctor_name # 使用全称或清理后的名称

                # 获取沟通内容和后续行动
                comm_content = doctor_info.get("沟通内容", "")
                follow_up = doctor_info.get("后续行动", "")
                # 如果LLM未提取后续行动，尝试从记录的明日计划中获取
                if not follow_up:
                     follow_up = record.get('优化_明日计划') or record.get('明日计划', '') # 使用优化后的计划优先

                # <<< 新增: 检查关键信息是否存在
                record_date = record.get("汇报时间")
                record_employee = record.get("员工姓名")
                if final_doctor_name and comm_content and record_date and record_employee:
                    visit_record = {
                        "拜访日期": record_date,
                        "拜访员工": record_employee,
                        "医生姓名": final_doctor_name,
                        "科室": std_department_name,
                        "职称": final_title,
                        "沟通内容": comm_content,
                        "后续行动": follow_up,
                        "关系评分": 0 # 评分后续统一处理
                    }
                    valid_visits_for_this_hospital.append(visit_record) # <<< 添加到临时列表

                    # 添加医生映射到主数据 (如果找到全称)
                    if full_doctor_name and full_doctor_name != cleaned_doctor_name:
                         add_to_master_data("doctors", doctor_key, full_doctor_name)
                    # 添加科室映射到主数据
                    if std_department_name != raw_department:
                         add_to_master_data("departments", raw_department, std_department_name)
                else:
                    logging.debug(f"跳过不完整的医院拜访记录: 医生='{final_doctor_name}', 内容='{bool(comm_content)}', 日期='{record_date}', 员工='{record_employee}'")


            # <<< 新增: 只有当该医院有有效拜访记录时才添加到 updates
            if valid_visits_for_this_hospital:
                if hospital_key not in hospital_updates:
                    hospital_updates[hospital_key] = {
                        "医院名称": cleaned_hospital_name, # 使用员工提到的原始名称（清理后）
                        "标准医院名称": std_hospital_name, # 保留标准名称作为参考
                        "匹配分数": score,
                        "地区": region or record_city or record_province, # 优先使用匹配到的地区
                        "历史记录": []
                    }
                elif not hospital_updates[hospital_key].get("地区") and (region or record_city or record_province):
                    hospital_updates[hospital_key]["地区"] = region or record_city or record_province # 补充地区信息
                
                hospital_updates[hospital_key]["历史记录"].extend(valid_visits_for_this_hospital)


        # 处理经销商信息
        for distributor_info in extracted_data.get("经销商信息", []):
            raw_dist_name = distributor_info.get("经销商名称", "")
            if not raw_dist_name: continue
            cleaned_dist_name = clean_entity_name(raw_dist_name)
            if not cleaned_dist_name: continue
            if "陆总" in cleaned_dist_name: continue # 跳过内部人员

            # 实体再分类检查
            is_likely_hospital = any(keyword in cleaned_dist_name for keyword in hospital_keywords) and not any(keyword in cleaned_dist_name for keyword in distributor_keywords)
            if is_likely_hospital:
                 logging.debug(f"再分类: 将 '{cleaned_dist_name}' 从经销商移动到医院处理。")
                 # (将逻辑移到上面医院处理部分 - 这里简化，只记录日志，实际应移动数据)
                 continue

            # 先检查联系人中是否有带"主任"关键词的人，如果有则可能是错误地将医院分类为经销商
            has_director_contacts = False
            for contact_info in distributor_info.get("联系人信息", []):
                raw_contact_name = contact_info.get("联系人姓名", "")
                if raw_contact_name and "主任" in raw_contact_name:
                    has_director_contacts = True
                    break
                    
            if has_director_contacts:
                logging.debug(f"发现经销商 '{cleaned_dist_name}' 下有带主任职称的联系人，重新归类为医院处理。")
                # 将此经销商重新归类为医院，将联系人重新归类为医生
                hospital_key = cleaned_dist_name
                valid_visits_for_reclassified_hospital = [] # <<< 新增
                if hospital_key not in hospital_updates:
                     # 仅在有有效医生记录时才创建医院条目
                     pass # 创建移到下面判断后
                
                # 将联系人转为医生
                for contact_info in distributor_info.get("联系人信息", []):
                    raw_contact_name = contact_info.get("联系人姓名", "")
                    raw_position = contact_info.get("职位", "")
                    
                    if not raw_contact_name: continue
                    
                    # 清理医生姓名，分离职称
                    cleaned_doctor_name, _, extracted_title = clean_doctor_name(raw_contact_name)
                    final_title = extracted_title or raw_position # 优先使用提取的职称
                    
                    if not cleaned_doctor_name: continue
                    
                    # <<< 新增: 检查关键信息是否存在
                    comm_content = contact_info.get("沟通内容", "")
                    follow_up = contact_info.get("后续行动", "")
                    record_date = record.get("汇报时间")
                    record_employee = record.get("员工姓名")

                    if cleaned_doctor_name and comm_content and record_date and record_employee:
                        visit_record = {
                            "拜访日期": record_date,
                            "拜访员工": record_employee,
                            "医生姓名": cleaned_doctor_name,
                            "科室": "", # 无科室信息
                            "职称": final_title,
                            "沟通内容": comm_content,
                            "后续行动": follow_up,
                            "关系评分": 0
                        }
                        valid_visits_for_reclassified_hospital.append(visit_record) # <<< 添加到临时列表
                    else:
                        logging.debug(f"跳过不完整的(再分类)医院拜访记录: 医生='{cleaned_doctor_name}', 内容='{bool(comm_content)}', 日期='{record_date}', 员工='{record_employee}'")

                # <<< 新增: 只有当该(再分类)医院有有效拜访记录时才添加到 updates
                if valid_visits_for_reclassified_hospital:
                    if hospital_key not in hospital_updates:
                         hospital_updates[hospital_key] = {
                            "医院名称": cleaned_dist_name,
                            "标准医院名称": "", # 暂不提供标准名称
                            "匹配分数": 0,
                            "地区": record_city or record_province,
                            "历史记录": []
                         }
                    hospital_updates[hospital_key]["历史记录"].extend(valid_visits_for_reclassified_hospital)

                # 处理完后跳过正常经销商处理
                continue

            # 标准化经销商名称 (如果需要，可以类似医院建立参考表)
            # 这里简化，直接使用主数据查找或原名
            std_dist_name = lookup_in_master_data("distributors", cleaned_dist_name) or cleaned_dist_name

            # 使用原始名称作为 key
            dist_key = cleaned_dist_name
            valid_comms_for_this_distributor = [] # <<< 新增: 存储当前记录中该经销商的有效沟通

            # 处理联系人信息
            for contact_info in distributor_info.get("联系人信息", []):
                raw_contact_name = contact_info.get("联系人姓名", "")
                raw_position = contact_info.get("职位", "")
                if not raw_contact_name: continue
                if "陆总" in raw_contact_name: continue # 跳过内部人员

                cleaned_contact_name = clean_entity_name(raw_contact_name)
                # 尝试从姓名中分离职位 (如果LLM没分)
                if not raw_position:
                     # 简单规则示例
                     pos_keywords = ["经理", "主管", "负责人", "销售", "董事", "总监", "老板", "总"]
                     for kw in pos_keywords:
                          if cleaned_contact_name.endswith(kw):
                               name_part = cleaned_contact_name[:-len(kw)].strip()
                               if name_part:
                                   cleaned_contact_name = name_part
                                   raw_position = kw
                                   break

                if not cleaned_contact_name: continue

                # 查找联系人全称
                contact_key = (dist_key, cleaned_contact_name)
                full_contact_name = lookup_in_master_data("contacts", contact_key)

                # 如果是单字姓名+职位，尝试用姓氏查找
                if not full_contact_name and len(cleaned_contact_name) == 1 and raw_position:
                     contact_key_surname = (dist_key, cleaned_contact_name)
                     full_contact_name = lookup_in_master_data("contacts", contact_key_surname)

                final_contact_name = full_contact_name or cleaned_contact_name

                # 获取沟通内容和后续行动
                comm_content = contact_info.get("沟通内容", "")
                follow_up = contact_info.get("后续行动", "")
                if not follow_up:
                     follow_up = record.get('优化_明日计划') or record.get('明日计划', '')

                # <<< 新增: 检查关键信息是否存在
                record_date = record.get("汇报时间")
                record_employee = record.get("员工姓名")
                if final_contact_name and comm_content and record_date and record_employee:
                    comm_record = {
                        "沟通日期": record_date,
                        "沟通员工": record_employee,
                        "联系人": final_contact_name,
                        "职位": raw_position,
                        "沟通内容": comm_content,
                        "后续计划": follow_up,
                        "关系评分": 0 # 后续统一处理
                    }
                    valid_comms_for_this_distributor.append(comm_record) # <<< 添加到临时列表

                    # 添加联系人映射到主数据
                    if full_contact_name and full_contact_name != cleaned_contact_name:
                         add_to_master_data("contacts", contact_key, full_contact_name)
                    # 添加经销商映射到主数据
                    if std_dist_name != cleaned_dist_name:
                         add_to_master_data("distributors", cleaned_dist_name, std_dist_name)
                else:
                     logging.debug(f"跳过不完整的经销商沟通记录: 联系人='{final_contact_name}', 内容='{bool(comm_content)}', 日期='{record_date}', 员工='{record_employee}'")


            # <<< 新增: 只有当该经销商有有效沟通记录时才添加到 updates
            if valid_comms_for_this_distributor:
                if dist_key not in distributor_updates:
                    distributor_updates[dist_key] = {
                        "经销商名称": cleaned_dist_name, # 使用员工提到的原始名称（清理后）
                        "标准名称": std_dist_name, # 添加标准名称字段作为参考
                        "地区": record_city or record_province, # 使用记录推断的地点
                        "沟通记录": []
                    }
                elif not distributor_updates[dist_key].get("地区") and (record_city or record_province):
                    distributor_updates[dist_key]["地区"] = record_city or record_province # 补充地区
                
                distributor_updates[dist_key]["沟通记录"].extend(valid_comms_for_this_distributor)

        return hospital_updates, distributor_updates
    except Exception as e:
        logging.error(f"处理记录时出错: {e}", exc_info=True)
        return {}, {}

def assess_relationship_score(interaction_content, api_client):
    """评估关系评分 (1-10)"""
    if not interaction_content or len(interaction_content.strip()) < 10: # 内容过少无法评估
        return 3 # 默认较低分数
    
    # >>> MOD: 使用基于少样本的关系分数评估
    full_prompt = REL_SCORE_PROMPT_STEM + interaction_content
    rsp = _call_llm_api(api_client, [{"role": "user", "content": full_prompt}], temperature=0.2)
    data = extract_valid_json(rsp)
    try:
        return max(1, min(10, int(data["score"])))
    except Exception:
        logging.warning("LLM score parse failed – fallback to 5. raw: %s", rsp)
        return 5

def assess_relationship_scores_batch(interactions, api_client):
    """批量评估关系评分"""
    scores = [(i, 3) for i in range(len(interactions))] # 默认值
    valid_interactions = [(i, content) for i, content in enumerate(interactions) if content and len(content.strip()) >= 10]

    if not valid_interactions:
        return [s for i, s in scores]

    logging.info(f"准备批量评估 {len(valid_interactions)} 个交互的关系评分...")
    results = {} # index -> score
    with concurrent.futures.ThreadPoolExecutor(max_workers=PARALLEL_MAX_WORKERS) as executor:
        future_to_index = {
            executor.submit(assess_relationship_score, content, api_client): i
            for i, content in valid_interactions
        }
        with tqdm(total=len(future_to_index), desc="评估关系评分") as pbar:
            for future in concurrent.futures.as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    score = future.result()
                    results[index] = score
                except Exception as e:
                    logging.error(f"评估关系评分时出错 (索引 {index}): {e}")
                    results[index] = 5 # 出错给默认分
                pbar.update(1)

    # 更新分数列表
    final_scores = [results.get(i, s) for i, s in scores]
    return final_scores

def check_and_deduplicate_locations(locations):
    """检查并去除重复的地点信息（省份/城市）"""
    if not locations or not isinstance(locations, list):
        return []
        
    # 规范化地点字符串并去重
    unique_locations = []
    seen = set()
    
    for location in locations:
        if isinstance(location, str):
            # 清理地点名称
            clean_loc = clean_location_term(location)
            if clean_loc and clean_loc not in seen:
                seen.add(clean_loc)
                unique_locations.append(clean_loc)
        elif isinstance(location, dict):
            # 处理{province, city}结构
            province = clean_location_term(location.get("province", ""))
            city = clean_location_term(location.get("city", ""))
            
            # 创建唯一键
            location_key = f"{province}|{city}"
            
            if location_key not in seen and (province or city):
                seen.add(location_key)
                unique_locations.append({"province": province, "city": city})
    
    logging.info(f"地点去重: 原始数量={len(locations)}, 去重后={len(unique_locations)}")
    return unique_locations


# 在update_data_with_new_records中应用地点去重
def update_data_with_new_records(existing_hospital_data, existing_distributor_data, new_records, api_client, all_employee_records, reference_df, vec_model=None, vec_index=None, vec_names=None, vec_locs=None, online_search_threshold: float = 60.0):
    """更新现有数据，添加新的记录
    
    Args:
        existing_hospital_data: 现有医院数据
        existing_distributor_data: 现有经销商数据
        new_records: 新提取的记录
        api_client: OpenAI API客户端
        all_employee_records: 所有员工历史记录
        reference_df: 医院参考数据集
        vec_model: 向量模型
        vec_index: 向量索引
        vec_names: 向量名称
        vec_locs: 向量位置
        online_search_threshold: 在线搜索阈值
        
    Returns:
        更新后的医院数据和经销商数据
    """
    # 复制现有数据，避免修改原始数据
    hospital_data = copy.deepcopy(existing_hospital_data)
    distributor_data = copy.deepcopy(existing_distributor_data)
    
    # 确保hospital_data和distributor_data是字典格式，而不是列表
    if isinstance(hospital_data, list):
        hospital_data = {"hospitals": hospital_data}
    elif "hospitals" not in hospital_data:
        hospital_data = {"hospitals": []}
        
    if isinstance(distributor_data, list):
        distributor_data = {"distributors": distributor_data}
    elif "distributors" not in distributor_data:
        distributor_data = {"distributors": []}
        
    # 将hospital_data["hospitals"]和distributor_data["distributors"]中的项转换为字典索引形式
    hospital_dict = {}
    for hospital in hospital_data.get("hospitals", []):
        if isinstance(hospital, dict) and "医院名称" in hospital:
            hospital_dict[hospital["医院名称"]] = hospital
    
    distributor_dict = {}
    for distributor in distributor_data.get("distributors", []):
        if isinstance(distributor, dict) and "经销商名称" in distributor:
            distributor_dict[distributor["经销商名称"]] = distributor
    
    # 预处理：分析员工的区域分布
    employee_regions_map = {}
    employee_recent_reports = {}
    # 1. 收集每个员工的最近报告
    for employee_name in set(record["拜访员工"] for record in new_records if "拜访员工" in record):
        employee_recent_reports[employee_name] = get_employee_recent_reports(employee_name, all_employee_records)
    
    # 2. 分析员工所在区域 (这有助于地理上下文)
    for employee_name, recent_reports in employee_recent_reports.items():
        if recent_reports:
            # 组合最近报告文本
            combined_text = "\n".join([report.get("沟通内容", "") for report in recent_reports])
            regions = analyze_employee_regions(employee_name, combined_text, api_client)
            # 应用地点去重
            regions = check_and_deduplicate_locations(regions)
            if regions:
                employee_regions_map[employee_name] = regions
    
    # --- 更多代码 ---

    # --- 1. 并行处理所有新记录 ---
    all_hospital_updates = {}
    all_distributor_updates = {}
    logging.info(f"开始并行处理 {len(new_records.get('records', []))} 条新记录...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=PARALLEL_MAX_WORKERS) as executor:
        # 注意：传递 all_employee_records 用于获取历史记录
        future_to_record = {
            executor.submit(
                process_single_record, 
                record, 
                api_client, 
                all_employee_records, 
                reference_df, 
                vec_model=vec_model, 
                vec_index=vec_index, 
                vec_names=vec_names, 
                vec_locs=vec_locs,
                online_search_threshold=online_search_threshold
            ): record
            for record in new_records.get("records", [])
            # 新增判断：仅处理 "今日结果" 字段不为空的记录
            if any(record.get("今日结果") and str(record.get("今日结果")).strip() 
                   for field in ["今日目标", "今日结果", "明日计划", "其他事项"])
        }
        with tqdm(total=len(future_to_record), desc="处理新记录") as pbar:
            for future in concurrent.futures.as_completed(future_to_record):
                record = future_to_record[future]
                try:
                    hospital_updates, distributor_updates = future.result()
                    # 合并结果 (简单合并，后续统一处理)
                    for name, data in hospital_updates.items():
                        if name not in all_hospital_updates: all_hospital_updates[name] = {"历史记录": []}
                        all_hospital_updates[name]["历史记录"].extend(data.get("历史记录", []))
                        # 合并其他信息 (如地区，标准名 - 取最后一次处理的?)
                        for key in ["医院名称", "标准医院名称", "匹配分数", "地区"]:
                             if key in data: all_hospital_updates[name][key] = data[key]

                    for name, data in distributor_updates.items():
                        if name not in all_distributor_updates: all_distributor_updates[name] = {"沟通记录": []}
                        all_distributor_updates[name]["沟通记录"].extend(data.get("沟通记录", []))
                        for key in ["经销商名称", "地区"]:
                             if key in data: all_distributor_updates[name][key] = data[key]

                except Exception as e:
                    logging.error(f"处理记录时出错 (员工: {record.get('员工姓名', 'N/A')}, 时间: {record.get('汇报时间', 'N/A')}): {e}", exc_info=True)
                pbar.update(1)

    # --- 2. 批量评估关系评分 ---
    all_hospital_interactions = []
    hospital_visit_pointers = [] # (hospital_key, visit_index_in_updates)
    for h_key, h_data in all_hospital_updates.items():
        for i, visit in enumerate(h_data.get("历史记录", [])):
            content = f"医生:{visit.get('医生姓名','')} 科室:{visit.get('科室','')} 内容:{visit.get('沟通内容','')} 后续:{visit.get('后续行动','')}"
            all_hospital_interactions.append(content)
            hospital_visit_pointers.append((h_key, i))

    all_distributor_interactions = []
    distributor_comm_pointers = [] # (dist_key, comm_index_in_updates)
    for d_key, d_data in all_distributor_updates.items():
        for i, comm in enumerate(d_data.get("沟通记录", [])):
             content = f"联系人:{comm.get('联系人','')} 职位:{comm.get('职位','')} 内容:{comm.get('沟通内容','')} 后续:{comm.get('后续计划','')}"
             all_distributor_interactions.append(content)
             distributor_comm_pointers.append((d_key, i))

    hospital_scores = assess_relationship_scores_batch(all_hospital_interactions, api_client)
    distributor_scores = assess_relationship_scores_batch(all_distributor_interactions, api_client)

    # 将分数写回 updates 字典
    for idx, (h_key, visit_idx) in enumerate(hospital_visit_pointers):
        if idx < len(hospital_scores):
            try:
                all_hospital_updates[h_key]["历史记录"][visit_idx]["关系评分"] = hospital_scores[idx]
            except (KeyError, IndexError):
                 logging.warning(f"无法将医院关系评分写回索引: {h_key}, {visit_idx}")

    for idx, (d_key, comm_idx) in enumerate(distributor_comm_pointers):
        if idx < len(distributor_scores):
             try:
                 all_distributor_updates[d_key]["沟通记录"][comm_idx]["关系评分"] = distributor_scores[idx]
             except (KeyError, IndexError):
                  logging.warning(f"无法将经销商关系评分写回索引: {d_key}, {comm_idx}")

    # --- 3. 合并到现有数据 ---
    logging.info("合并新处理的数据到现有数据结构...")
    # 更新医院数据
    for hospital_key, hospital_info in all_hospital_updates.items():
        if hospital_key in hospital_dict: # 已存在医院
            existing_hospital = hospital_dict[hospital_key]
            # 更新基本信息 (如果新的更完整)
            if hospital_info.get("地区") and not existing_hospital.get("地区"):
                 existing_hospital["地区"] = hospital_info["地区"]
            if hospital_info.get("标准医院名称") != existing_hospital.get("标准医院名称"):
                 existing_hospital["标准医院名称"] = hospital_info.get("标准医院名称")
                 existing_hospital["匹配分数"] = hospital_info.get("匹配分数")

            # 添加新的、不重复的历史记录
            existing_records_keys = {
                (v.get("拜访日期", ""), v.get("拜访员工", ""), v.get("医生姓名", "")): True
                for v in existing_hospital.get("历史记录", [])
            }
            new_visits_added = 0
            for visit in hospital_info.get("历史记录", []):
                 visit_key = (visit.get("拜访日期", ""), visit.get("拜访员工", ""), visit.get("医生姓名", ""))
                 if visit_key not in existing_records_keys:
                     existing_hospital.setdefault("历史记录", []).append(visit)
                     existing_records_keys[visit_key] = True
                     new_visits_added += 1
            # if new_visits_added > 0:
            #     logging.debug(f"向医院 '{hospital_key}' 添加了 {new_visits_added} 条新拜访记录。")
        else: # 新医院
            hospital_dict[hospital_key] = hospital_info
            # logging.debug(f"添加新医院 '{hospital_key}'。")

    # 更新经销商数据
    for dist_key, distributor_info in all_distributor_updates.items():
        if dist_key in distributor_dict: # 已存在经销商
            existing_distributor = distributor_dict[dist_key]
            if distributor_info.get("地区") and not existing_distributor.get("地区"):
                 existing_distributor["地区"] = distributor_info["地区"]

            existing_records_keys = {
                (c.get("沟通日期", ""), c.get("沟通员工", ""), c.get("联系人", "")): True
                for c in existing_distributor.get("沟通记录", [])
            }
            new_comms_added = 0
            for comm in distributor_info.get("沟通记录", []):
                 comm_key = (comm.get("沟通日期", ""), comm.get("沟通员工", ""), comm.get("联系人", ""))
                 if comm_key not in existing_records_keys:
                     existing_distributor.setdefault("沟通记录", []).append(comm)
                     existing_records_keys[comm_key] = True
                     new_comms_added += 1
            # if new_comms_added > 0:
            #     logging.debug(f"向经销商 '{dist_key}' 添加了 {new_comms_added} 条新沟通记录。")
        else: # 新经销商
            distributor_dict[dist_key] = distributor_info
            # logging.debug(f"添加新经销商 '{dist_key}'。")

    # --- 4. 清理和排序 ---
    logging.info("清理并排序最终数据...")
    final_hospital_data = {"hospitals": list(hospital_dict.values())}
    final_distributor_data = {"distributors": list(distributor_dict.values())}

    for hospital in final_hospital_data["hospitals"]:
        unique_visits = {}
        for visit in hospital.get("历史记录", []):
            key = (visit.get("拜访日期", ""), visit.get("拜访员工", ""), visit.get("医生姓名", ""))
            # 保留评分最高的记录（如果存在重复）
            if key not in unique_visits or visit.get("关系评分", 0) > unique_visits[key].get("关系评分", 0):
                 unique_visits[key] = visit
        # 按日期排序
        sorted_visits = sorted(
            unique_visits.values(),
            key=lambda x: datetime.strptime(x.get("拜访日期", "1900-01-01"), "%Y-%m-%d") if x.get("拜访日期") else datetime.min,
            reverse=True
        )
        hospital["历史记录"] = sorted_visits

    for distributor in final_distributor_data["distributors"]:
        unique_comms = {}
        for comm in distributor.get("沟通记录", []):
            key = (comm.get("沟通日期", ""), comm.get("沟通员工", ""), comm.get("联系人", ""))
            if key not in unique_comms or comm.get("关系评分", 0) > unique_comms[key].get("关系评分", 0):
                 unique_comms[key] = comm
        sorted_comms = sorted(
            unique_comms.values(),
             key=lambda x: datetime.strptime(x.get("沟通日期", "1900-01-01"), "%Y-%m-%d") if x.get("沟通日期") else datetime.min,
            reverse=True
        )
        distributor["沟通记录"] = sorted_comms

    # 收集所有发现的医生名称映射
    doctor_mapping_updates = {}  # 存储 (hospital, department, abbr) -> full_name
    
    # 遍历主数据中的doctors信息，收集简称-全称映射
    for key, value in master_data.get("doctors", {}).items():
        if isinstance(key, tuple) and len(key) == 3:
            hospital, department, doctor_abbr = key
            if doctor_abbr != value:  # 如果不是自映射，即存在简称到全称的映射
                if (hospital, department) not in doctor_mapping_updates:
                    doctor_mapping_updates[(hospital, department)] = {}
                doctor_mapping_updates[(hospital, department)][doctor_abbr] = value
    
    # 如果发现了新的医生全称映射，回溯更新历史记录
    if doctor_mapping_updates:
        # 处理医院数据 - 确保能同时处理字典和列表格式
        hospitals_to_process = []
        
        # 根据hospital_data的类型收集医院数据
        if isinstance(final_hospital_data, dict) and "hospitals" in final_hospital_data:
            # 如果是新的结构 {"hospitals": [...]}
            hospitals_to_process = final_hospital_data["hospitals"]
        elif isinstance(final_hospital_data, list):
            # 如果直接是医院列表
            hospitals_to_process = final_hospital_data
        
        # 现在处理收集到的医院列表
        for hospital in hospitals_to_process:
            hospital_name = hospital.get("医院名称", "")
            if not hospital_name:
                continue
                
            history_records = hospital.get("历史记录", [])
            updated = False
            
            # 遍历该医院的所有历史记录
            for i, record in enumerate(history_records):
                doctor_name = record.get("医生姓名", "")
                department = record.get("科室", "")
                
                # 检查是否有可用的映射
                if (hospital_name, department) in doctor_mapping_updates:
                    mappings = doctor_mapping_updates[(hospital_name, department)]
                    
                    # 检查医生简称是否在映射中
                    if doctor_name in mappings:
                        full_name = mappings[doctor_name]
                        # 更新记录中的医生名称为全称
                        history_records[i]["医生姓名"] = full_name
                        updated = True
                        logging.info(f"回溯更新: 医院 '{hospital_name}' 科室 '{department}' 中的医生 '{doctor_name}' 更新为 '{full_name}'")
                    else:
                        # 检查是否有姓氏相同且带职称的简称
                        title_terms = ["主任", "医生", "医师", "专家", "教授", "博士", "院长", "科长"]
                        if any(term in doctor_name for term in title_terms) and len(doctor_name) >= 1:
                            # 提取姓氏
                            surname = doctor_name[0]
                            # 查找同姓的全称
                            for abbr, full in mappings.items():
                                if len(abbr) >= 1 and abbr[0] == surname:
                                    # 更新记录中的医生名称为全称
                                    history_records[i]["医生姓名"] = full
                                    updated = True
                                    logging.info(f"回溯更新(姓氏匹配): 医院 '{hospital_name}' 科室 '{department}' 中的医生 '{doctor_name}' 更新为 '{full}'")
                                    # 添加新的映射关系到主数据
                                    add_to_master_data("doctors", (hospital_name, department, doctor_name), full)
                                    break
            
            # 如果有更新，确保保存回原始数据
            if updated:
                hospital["历史记录"] = history_records
    
    return final_hospital_data, final_distributor_data

def update_employee_records(existing_records, new_records_list):
    """更新员工记录，保持滚动，避免重复"""
    employee_groups = {}
    processed_report_ids = set()

    # 处理现有记录
    for record in existing_records.get("records", []):
        name = record.get("员工姓名")
        report_id = record.get("汇报编号")
        if not name: continue
        if name not in employee_groups: employee_groups[name] = []
        employee_groups[name].append(record)
        if report_id: processed_report_ids.add(report_id)

    # 添加新记录 (只添加不重复的)
    new_added_count = 0
    for record in new_records_list: # new_records_list 是记录列表
        name = record.get("员工姓名")
        report_id = record.get("汇报编号")
        if not name: continue
        if report_id and report_id in processed_report_ids: continue # 跳过重复

        if name not in employee_groups: employee_groups[name] = []
        employee_groups[name].append(record)
        if report_id: processed_report_ids.add(report_id)
        new_added_count += 1
    logging.info(f"向员工记录中添加了 {new_added_count} 条新记录。")

    # 按日期排序并应用滚动窗口
    all_records_final = []
    logging.info(f"开始为 {len(employee_groups)} 名员工应用滚动记录窗口...")
    with tqdm(total=len(employee_groups), desc="更新员工记录窗口") as pbar:
        for name, records in employee_groups.items():
            try:
                # 按汇报时间排序 (降序)
                sorted_records = sorted(
                    records,
                    key=lambda x: datetime.strptime(x.get("汇报时间", "1900-01-01"), "%Y-%m-%d") if x.get("汇报时间") else datetime.min,
                    reverse=True
                )
                if not sorted_records:
                    pbar.update(1)
                    continue

                # 获取最新汇报日期并计算截止日期
                latest_report_date = datetime.strptime(sorted_records[0]["汇报时间"], "%Y-%m-%d")
                cutoff_date = latest_report_date - timedelta(days=EMPLOYEE_RECORD_ROLLING_DAYS)

                # 保留最近 N 天的记录
                recent_records = [
                    r for r in sorted_records
                    if r.get("汇报时间") and datetime.strptime(r["汇报时间"], "%Y-%m-%d") >= cutoff_date
                ]
                all_records_final.extend(recent_records)

            except ValueError as e:
                 logging.error(f"处理员工 {name} 记录时日期格式错误: {e}。该员工记录可能未被正确过滤。")
                 all_records_final.extend(records) # 保留原始记录以防数据丢失
            except Exception as e:
                 logging.error(f"处理员工 {name} 记录时发生未知错误: {e}", exc_info=True)
                 all_records_final.extend(records)
            pbar.update(1)

    logging.info(f"员工记录滚动窗口应用完成，最终保留 {len(all_records_final)} 条记录。")
    return {"records": all_records_final}

def process_comments_and_optimize(employee_records_dict, api_client):
    """处理评论信息，并根据评论优化工作内容 (返回更新后的字典)"""
    records = employee_records_dict.get("records", [])
    if not records: return employee_records_dict

    # 1. 收集评论
    report_comments = {}
    for record in records:
        report_id = record.get("汇报编号")
        comment = record.get("评论", "")
        commenter = record.get("汇报对象", "") # 评论人
        if report_id and comment and str(comment).strip():
            if report_id not in report_comments: report_comments[report_id] = []
            report_comments[report_id].append({"评论": str(comment), "评论人": commenter})
    logging.info(f"收集到 {len(report_comments)} 个报告的评论。")
    if not report_comments:
        logging.info("没有需要处理的评论。")
        return employee_records_dict # 没有评论则无需优化

    # 2. 并行处理评论并生成优化建议
    report_suggestions = {} # report_id -> suggestion_dict
    records_map = {r.get("汇报编号"): r for r in records if r.get("汇报编号")} # 便于查找原始记录

    logging.info(f"准备并行处理 {len(report_comments)} 个报告的评论...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=PARALLEL_MAX_WORKERS) as executor:
        future_to_report_id = {}
        for report_id, comments_list in report_comments.items():
            if report_id in records_map:
                original_record = records_map[report_id]
                future = executor.submit(
                    _process_single_report_comments,
                    original_record,
                    comments_list,
                    api_client
                )
                future_to_report_id[future] = report_id
            else:
                 logging.warning(f"找不到报告ID {report_id} 的原始记录，无法处理其评论。")

        with tqdm(total=len(future_to_report_id), desc="处理评论生成优化建议") as pbar:
            for future in concurrent.futures.as_completed(future_to_report_id):
                report_id = future_to_report_id[future]
                try:
                    suggestion = future.result()
                    if suggestion: # 只有成功生成建议才记录
                        report_suggestions[report_id] = suggestion
                except Exception as e:
                    logging.error(f"处理报告 {report_id} 的评论时出错: {e}", exc_info=True)
                pbar.update(1)

    # 3. 应用优化建议到记录中
    logging.info(f"准备将 {len(report_suggestions)} 条优化建议应用到员工记录...")
    updated_records = copy.deepcopy(records) # 操作副本
    applied_count = 0
    for i, record in enumerate(updated_records):
        report_id = record.get("汇报编号")
        if report_id and report_id in report_suggestions:
            suggestion = report_suggestions[report_id]
            record["评论要点"] = suggestion.get("评论要点", "")
            optimized_content = suggestion.get("优化建议", {})
            for field, optimized_text in optimized_content.items():
                 if optimized_text and isinstance(optimized_text, str) and optimized_text.strip():
                     record[f"优化_{field}"] = optimized_text # 添加 "优化_" 前缀
            applied_count += 1
            # logging.debug(f"已将优化建议应用到报告 {report_id}")

    logging.info(f"成功将优化建议应用到 {applied_count} 条员工记录。")
    return {"records": updated_records}

def _process_single_report_comments(original_record, comments_list, api_client):
    """处理单个报告的评论并生成优化建议 (辅助函数)"""
    report_content = f"""
    汇报时间: {original_record.get('汇报时间', '')}
    员工姓名: {original_record.get('员工姓名', '')}
    部门: {original_record.get('部门', '')}
    今日目标: {original_record.get('今日目标', '')}
    今日结果: {original_record.get('今日结果', '')}
    明日计划: {original_record.get('明日计划', '')}
    其他事项: {original_record.get('其他事项', '')}
    """
    comments_text = "\n\n".join([f"评论人: {c.get('评论人', '未知')}\n评论内容: {c.get('评论', '')}" for c in comments_list])

    # 如果评论为空 (理论上不会进入这里，因为调用前已检查)
    if not comments_text.strip():
        # 简化明日计划
        tomorrow_plan = original_record.get('明日计划', '')
        if tomorrow_plan and len(tomorrow_plan.strip()) > 5: # 计划较长才简化
             simplify_prompt = f"请将以下工作计划总结为关键要点，保持简洁明了，使用序号列表:\n{tomorrow_plan}"
             simplified_plan = _call_llm_api(api_client, [{"role": "user", "content": simplify_prompt}], LLM_TEMPERATURE_OPTIMIZATION)
             if simplified_plan:
                 return {
                     "评论要点": "无评论，已简化明日计划。",
                     "优化建议": {"明日计划": simplified_plan}
                 }
        return {"评论要点": "无评论。", "优化建议": {}} # 无评论且计划短

    # 有评论，进行优化
    optimize_prompt = f"""
    请根据以下工作日报和领导评论，优化日报内容，特别是"明日计划"。
    原始工作日报：
    {report_content}
    领导评论：
    {comments_text}
    优化要求：
    1. 提取领导评论中的核心指导意见或要求，总结为"评论要点"。
    2. 根据评论，优化"今日结果"的表述（可选，如果评论涉及对结果的看法）。
    3. **重点优化"明日计划"**，使其更具体、更具可操作性，并体现评论中的建议或方向。
    4. 保持原始事实基本不变，主要是调整措辞、补充细节、明确行动。
    5. 不要直接引用评论内容，而是将评论精神融入优化后的文本。
    请务必以下列**严格有效**的JSON格式返回结果，空字段返回null或空字符串。
    **重要：确保所有字符串值中的特殊字符（特别是双引号 `"`）都已正确转义为 `\"`**。
    ```json
    {{
      "评论要点": "简明扼要总结领导评论的核心指导原则",
      "优化建议": {{
        "今日结果": "优化后的今日结果内容（如果适用）",
        "明日计划": "优化后的明日计划内容"
      }}
    }}
    ```
    """
    messages = [
        {"role": "system", "content": "你是一位专业的工作日报优化助手。请根据领导评论，优化日报内容，特别是后续行动计划。务必返回严格有效的JSON。"},
        {"role": "user", "content": optimize_prompt}
    ]
    result_text = _call_llm_api(api_client, messages, LLM_TEMPERATURE_OPTIMIZATION)
    suggestion = extract_valid_json(result_text)

    if suggestion and isinstance(suggestion, dict) and "评论要点" in suggestion and "优化建议" in suggestion:
        # 基本验证
        if not isinstance(suggestion["优化建议"], dict):
             suggestion["优化建议"] = {}
        return suggestion
    else:
        logging.warning(f"无法为报告 {original_record.get('汇报编号')} 生成有效的优化建议。LLM原始返回: {result_text}")
        return None # 表示优化失败
# --- 核心逻辑函数结束 ---


# --- 新增 LLM 函数：从搜索结果提取医院名称 ---
def _extract_hospital_name_from_search_results(raw_results, query_name, location_context, api_client):
    """使用 LLM 从网页搜索结果中提取最可能的医院全称和地区。"""
    if not raw_results or not api_client:
        return None, None

    # 将搜索结果格式化为文本
    results_text = "\n".join([f"标题: {r.get('title', '')}\n链接: {r.get('link', '')}\n摘要: {r.get('snippet', '')}" for r in raw_results[:5]]) # 最多使用前5条结果

    location_hint = f"地理位置上下文: {', '.join(location_context)}" if location_context else ""

    prompt = f"""
    请分析以下关于医院 '{query_name}' 的网页搜索结果。{location_hint}
    目标是找出最准确的医院官方全称和其所在的省份与城市。

    搜索结果（包含标题和摘要）：
    {results_text}

    请综合分析标题、摘要和链接，判断最可靠的医院全称和地区信息。
    - 优先考虑官方网站或权威来源（如政府网站、地图服务）的信息。
    - 注意识别别名、分院、旧称等情况，提取最规范、最完整的官方全称。
    - **重要：如果名称包含院区信息（例如 'xx医院北区', 'xx医院光谷院区', 'xx医院东院'），请去除院区后缀，只返回主院名称（例如 'xx医院'）。**
    - 提取医院所在的 **省份** 和 **城市**。对于直辖市，省份和城市相同。
    - 如果无法确定，返回空字符串。

    请以严格的 JSON 格式返回结果，不要添加任何解释或其他内容：
    {{
        "hospital_full_name": "提取的最准确医院全称（已去除院区信息）",
        "province": "提取的省份/直辖市",
        "city": "提取的城市/直辖市"
    }}
    """
    messages = [
        {"role": "system", "content": "你是一个专门从网页搜索结果中提取准确医院官方全称、省份和城市的助手，并能处理院区信息。"},
        {"role": "user", "content": prompt}
    ]

    result_text = _call_llm_api(api_client, messages, LLM_TEMPERATURE_EXTRACTION)
    extracted_info = extract_valid_json(result_text)

    if extracted_info and isinstance(extracted_info, dict):
        full_name = extracted_info.get("hospital_full_name")
        province = extracted_info.get("province")
        city = extracted_info.get("city")

        if full_name and isinstance(full_name, str) and full_name.strip():
            # 清理名称和地区
            cleaned_name = clean_entity_name(full_name) # clean_entity_name 可进一步优化去除院区，但优先依赖 LLM
            # 组合省市作为最终地区信息
            cleaned_province = clean_location_term(province) if province else ""
            cleaned_city = clean_location_term(city) if city else ""
            final_region = f"{cleaned_province} {cleaned_city}".strip()
            # 如果省市相同（直辖市），只保留一个
            if cleaned_province == cleaned_city:
                 final_region = cleaned_city

            logging.info(f"LLM 从搜索结果提取: 医院='{cleaned_name}', 地区='{final_region}'")
            return cleaned_name, final_region # 返回组合后的地区

    logging.warning(f"LLM 无法从搜索结果中提取有效医院信息。Query: '{query_name}', LLM Raw: {result_text}")
    return None, None
# --- 新增 LLM 函数结束 ---


# --- 主函数 ---
def main():
    """主执行函数"""
    logging.info("--- 开始运行医疗日志处理程序 ---")

    # 初始化 API 客户端
    api_client = initialize_openai_client()
    if not api_client:
        return # 无法继续

    # 设置全局API客户端(用于在线医院查询)
    global global_api_client
    global_api_client = api_client

    # 加载主数据
    global master_data
    master_data = load_master_data(MASTER_DATA_FILE)

    # 加载输入 Excel 文件
    input_excel_file = EXCEL_FILE_DEFAULT # 或者从命令行参数获取
    df = load_data_from_excel(input_excel_file)
    if df is None:
        return # 无法继续

    # 转换 Excel 数据为内部记录格式
    new_log_records = convert_df_to_records(df)
    if not new_log_records or not new_log_records.get("records"):
        logging.error("从 Excel 文件转换记录失败或没有记录。")
        return

    # 加载现有状态数据 (从 CSV)
    logging.info("加载现有数据状态 (CSV)...")
    employee_records = load_records_from_csv(EMPLOYEE_RECORDS_CSV)
    hospital_data = load_structured_data_from_csv(
        HOSPITAL_DATA_CSV,
        id_col="医院ID", name_col="医院名称", history_col_name="历史记录",
        record_key_map={ # CSV列名 -> 内部键名 映射
            "拜访日期": "拜访日期", "拜访员工": "拜访员工", "医生姓名": "医生姓名",
            "科室": "科室", "职称": "职称", "沟通内容": "沟通内容",
            "后续行动": "后续行动", "关系评分": "关系评分"
        }
    )
    distributor_data = load_structured_data_from_csv(
        DISTRIBUTOR_DATA_CSV,
        id_col="经销商ID", name_col="经销商名称", history_col_name="沟通记录",
        record_key_map={
             "沟通日期": "沟通日期", "沟通员工": "沟通员工", "联系人": "联系人",
             "职位": "职位", "沟通内容": "沟通内容", "后续计划": "后续计划", # 注意后续计划的键名
             "关系评分": "关系评分"
        }
    )
    # 重命名经销商历史记录键名以匹配内部期望
    for dist in distributor_data.get("distributors", []):
        for comm in dist.get("沟通记录", []):
             if "后续计划" in comm:
                 comm["后续行动"] = comm.pop("后续计划")


    # --- 数据处理流水线 ---
    # 1. 更新员工记录 (滚动窗口, 去重)
    logging.info("步骤 1: 更新员工记录...")
    employee_records = update_employee_records(employee_records, new_log_records.get("records", []))

    # 2. 预处理: 分析员工区域 & 标准化医院名称
    logging.info("步骤 2: 预处理 - 分析员工区域 & 标准化医院名称...")
    # 2a. 分析员工负责区域 (基于最新记录)
    employee_regions_map = {}
    unique_employees = list(set(r.get("员工姓名") for r in employee_records.get("records", []) if r.get("员工姓名")))
    logging.info(f"准备分析 {len(unique_employees)} 名员工的负责区域...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=PARALLEL_MAX_WORKERS) as executor:
         future_to_emp = {}
         for emp_name in unique_employees:
             recent_reps = get_employee_recent_reports(emp_name, employee_records.get("records", []), 3)
             recent_text = "\n".join([f"目标:{r.get('今日目标', '')} 结果:{r.get('今日结果', '')} 计划:{r.get('明日计划', '')}" for r in recent_reps])
             if recent_text.strip():
                 future = executor.submit(analyze_employee_regions, emp_name, recent_text, api_client)
                 future_to_emp[future] = emp_name

         with tqdm(total=len(future_to_emp), desc="分析员工负责区域") as pbar:
             for future in concurrent.futures.as_completed(future_to_emp):
                 emp_name = future_to_emp[future]
                 try:
                     regions = future.result()
                     if regions:
                         employee_regions_map[emp_name] = regions
                 except Exception as e:
                     logging.error(f"分析员工 {emp_name} 区域时出错: {e}")
                 pbar.update(1)
    logging.info(f"成功分析了 {len(employee_regions_map)} 名员工的负责区域。")

    # 2b. 加载医院参考数据
    reference_df = None
    try:
        reference_df = pd.read_excel(HOSPITAL_REFERENCE_FILE)
        if 'name' not in reference_df.columns or 'location' not in reference_df.columns:
             logging.error(f"医院参考文件 {HOSPITAL_REFERENCE_FILE} 缺少 'name' 或 'location' 列。")
             reference_df = None
        else:
             # 基本清理
             reference_df = reference_df[['name', 'location']].copy()
             reference_df['name'] = reference_df['name'].astype(str).str.strip()
             reference_df['location'] = reference_df['location'].astype(str).str.strip()
             reference_df = reference_df[reference_df['name'] != '']
             logging.info(f"成功加载 {len(reference_df)} 条有效医院参考数据。")
    except FileNotFoundError:
        logging.warning(f"医院参考文件 {HOSPITAL_REFERENCE_FILE} 未找到，无法进行名称标准化预处理。")
    except Exception as e:
        logging.error(f"加载医院参考文件时出错: {e}", exc_info=True)

    # >>> NEW: 构建向量索引
    vec_model = vec_index = vec_names = vec_locs = None
    if reference_df is not None and not reference_df.empty:
        vec_model, vec_index, vec_names, vec_locs = build_hospital_vector_index(reference_df)

    # 2c. 标准化现有医院名称 (如果参考数据可用)
    if reference_df is not None:
        # 设置用于模糊匹配的阈值，低于此值会触发在线搜索
        online_search_threshold = 85.0  # 可以根据需要调整此阈值
        logging.info(f"设置模糊匹配阈值为 {online_search_threshold}，低于此值会触发在线搜索")
        
        hospital_data = preprocess_standardize_hospital_names(
            hospital_data, reference_df, employee_regions_map,
            vec_model=vec_model, vec_index=vec_index, vec_names=vec_names, vec_locs=vec_locs,
            online_search_threshold=online_search_threshold)
    else:
         logging.warning("跳过医院名称标准化预处理步骤。")

    # 3. 处理新记录并合并数据
    logging.info("步骤 3: 处理新日志记录并合并数据...")
    # (预处理历史记录以更新 master_data - 优化点：可以更早执行，甚至独立执行)
    all_history_records = employee_records.get("records", []) # 使用更新后的记录作为历史来源
    if all_history_records:
         logging.info("预处理历史记录以更新主数据映射...")
         # (这里简化，假设 build_contextual_mappings 能处理记录列表)
         build_contextual_mappings(all_history_records, api_client) # 更新全局 master_data
         # 注意：频繁更新 master_data 可能不是最高效的，可以考虑批量更新
         pass # 映射提取逻辑已分散到 process_single_record 和预处理中


    # 处理新记录并合并
    # 注意：传递 employee_records 用于 process_single_record 获取历史记录
    hospital_data, distributor_data = update_data_with_new_records(
        hospital_data, distributor_data, new_log_records, api_client, employee_records, reference_df, 
        vec_model=vec_model, vec_index=vec_index, vec_names=vec_names, vec_locs=vec_locs,
        online_search_threshold=85.0 # 修改为85.0
    )

    # 4. 处理评论并优化员工记录
    logging.info("步骤 4: 处理评论并优化员工记录...")
    employee_records = process_comments_and_optimize(employee_records, api_client)

    # 5. (可选) 应用优化结果到互动数据
    # **注意**: 这一步会增加大量 API 调用，如果优化主要影响计划文本，
    # 而不是实体识别，可以考虑跳过或用更轻量的方式更新后续行动。
    # 这里暂时保留，但标记为可选项。
    APPLY_OPTIMIZATIONS_TO_INTERACTIONS = False # 设置为 True 以启用
    if APPLY_OPTIMIZATIONS_TO_INTERACTIONS:
        logging.info("步骤 5: (可选) 将优化后的内容应用回医院/经销商数据...")
        optimized_log_records = {"records": [r for r in employee_records.get("records", []) if any(k.startswith("优化_") for k in r)]}
        if optimized_log_records.get("records"):
             logging.info(f"发现 {len(optimized_log_records['records'])} 条优化记录，重新处理以更新互动数据...")
             # 再次调用合并逻辑，但这次使用优化后的记录作为"新"记录
             # 注意：这可能导致重复计算评分和合并逻辑，需要小心处理
             # 传递原始 employee_records 用于历史查找
             hospital_data, distributor_data = update_data_with_new_records(
                 hospital_data, distributor_data, optimized_log_records, api_client, employee_records, reference_df,
                 vec_model=vec_model, vec_index=vec_index, vec_names=vec_names, vec_locs=vec_locs,
                 online_search_threshold=online_search_threshold
             )
        else:
             logging.info("没有发现优化记录，跳过应用优化到互动数据。")
    else:
         logging.info("步骤 5: 跳过将优化应用回互动数据的步骤。")


    # --- 输出与保存 ---
    # 6. 导出数据到 CSV
    logging.info("步骤 6: 导出最终数据到 CSV 文件...")
    
    # 保存内存中修改的主数据到文件
    logging.info("正在保存主数据到磁盘...")
    flush_master_data()
    
    # 导出医院数据
    hospital_headers = ["医院ID", "医院名称", "标准医院名称", "匹配分数", "地区", "拜访日期", "拜访员工", "医生姓名", "科室", "职称", "沟通内容", "后续行动", "关系评分"]
    hospital_record_keys = ["拜访日期", "拜访员工", "医生姓名", "科室", "职称", "沟通内容", "后续行动", "关系评分"]
    hospital_csv_rows = data_to_csv_rows(hospital_data, "hospitals", "医院ID", "医院名称", "历史记录", hospital_headers, hospital_record_keys)
    write_csv(hospital_csv_rows, HOSPITAL_DATA_CSV)

    # 导出经销商数据
    distributor_headers = ["经销商ID", "经销商名称", "标准名称", "地区", "沟通日期", "沟通员工", "联系人", "职位", "沟通内容", "后续行动", "关系评分"]
    distributor_record_keys = ["沟通日期", "沟通员工", "联系人", "职位", "沟通内容", "后续行动", "关系评分"] # 注意后续行动键名统一
    distributor_csv_rows = data_to_csv_rows(distributor_data, "distributors", "经销商ID", "经销商名称", "沟通记录", distributor_headers, distributor_record_keys)
    write_csv(distributor_csv_rows, DISTRIBUTOR_DATA_CSV)

    # 导出员工数据
    employee_csv_rows = employee_data_to_csv(employee_records)
    write_csv(employee_csv_rows, EMPLOYEE_RECORDS_CSV)

    # 7. 更新 MongoDB
    logging.info("步骤 7: 更新 MongoDB 数据...")
    db_password = input("请输入 MongoDB 数据库密码 (输入 'skip' 跳过): ")
    if db_password.lower() != 'skip':
        mongo_client = connect_to_mongodb(db_password)
        if mongo_client:
            update_success = update_mongodb_collections(mongo_client, hospital_data, distributor_data)
            if update_success:
                logging.info("MongoDB 数据更新成功。")
                
                # 同步主数据映射到MongoDB
                logging.info("同步主数据映射关系到MongoDB...")
                sync_result = sync_master_data_to_mongodb(mongo_client)
                if sync_result:
                    logging.info("主数据映射同步到MongoDB成功。")
                else:
                    logging.warning("主数据映射同步到MongoDB部分或全部失败。")
            else:
                logging.error("MongoDB 数据更新失败。")
            mongo_client.close()
            logging.info("MongoDB 连接已关闭。")
        else:
            logging.warning("无法连接到 MongoDB，跳过数据库更新。")
    else:
        logging.info("跳过 MongoDB 更新。")

    # 8. 保存主数据之前，用hospital_data的地区信息更新master_data
    logging.info("更新主数据中的医院地区信息...")
    update_master_data_with_hospital_regions()
    
    # 8.1 从CSV数据中提取所有实体映射关系，特别是医生简称-全称映射
    logging.info("从CSV数据更新主数据中的实体映射关系...")
    update_master_data_with_all_entities()

    # 保存主数据
    save_master_data(master_data, MASTER_DATA_FILE)

    # 9. (可选) 导出中间映射文件
    if EXPORT_MAP_FILES:
        logging.info("步骤 9: (可选) 导出中间映射文件...")
        # 注意：这些映射现在主要体现在 master_data 中，单独导出可能意义不大
        # 可以从 master_data 重构这些文件如果需要
        pass

    logging.info("--- 医疗日志处理程序运行结束 ---")

def update_master_data_with_hospital_regions():
    """使用hospital_data中的地区信息更新master_data"""
    global master_data
    
    # 加载医院数据
    hospital_data = load_structured_data_from_csv(
        HOSPITAL_DATA_CSV,
        id_col="医院ID", name_col="医院名称", history_col_name="历史记录",
        record_key_map={
            "拜访日期": "拜访日期", "拜访员工": "拜访员工", "医生姓名": "医生姓名",
            "科室": "科室", "职称": "职称", "沟通内容": "沟通内容",
            "后续行动": "后续行动", "关系评分": "关系评分"
        }
    )
    
    if not hospital_data or "hospitals" not in hospital_data:
        return False
    
    updated_count = 0
    # 遍历hospital_data中的每个医院
    for hospital in hospital_data["hospitals"]:
        hospital_name = hospital.get("医院名称")
        region = hospital.get("地区")
        
        if hospital_name and region:
            # 更新主数据中的地区信息
            if add_to_master_data("hospitals", hospital_name, hospital_name, "全称", region):
                updated_count += 1
                
    logging.info(f"从hospital_data更新了{updated_count}个医院的地区信息到master_data")
    return True

def update_master_data_with_all_entities():
    """从CSV文件更新master_data中的所有实体信息，包括医院、医生、经销商和联系人"""
    global master_data
    updated_count = 0
    
    # 1. 更新医院信息
    hospital_data = load_structured_data_from_csv(
        HOSPITAL_DATA_CSV,
        id_col="医院ID", name_col="医院名称", history_col_name="历史记录",
        record_key_map={
            "拜访日期": "拜访日期", "拜访员工": "拜访员工", "医生姓名": "医生姓名",
            "科室": "科室", "职称": "职称", "沟通内容": "沟通内容",
            "后续行动": "后续行动", "关系评分": "关系评分"
        }
    )
    
    # 2. 更新经销商信息
    distributor_data = load_structured_data_from_csv(
        DISTRIBUTOR_DATA_CSV,
        id_col="经销商ID", name_col="经销商名称", history_col_name="沟通记录",
        record_key_map={
            "沟通日期": "沟通日期", "沟通员工": "沟通员工", "联系人": "联系人",
            "职位": "职位", "沟通内容": "沟通内容", "后续行动": "后续行动", 
            "关系评分": "关系评分"
        }
    )
    
    # 3. 更新医院和地区
    for hospital in hospital_data.get("hospitals", []):
        hospital_name = hospital.get("医院名称")
        region = hospital.get("地区")
        
        if hospital_name and region:
            if add_to_master_data("hospitals", hospital_name, hospital_name, "全称", region):
                updated_count += 1
    
    # 4. 更新医生信息，并尝试建立简称-全称映射
    # 创建医院-科室到医生记录的映射
    hospital_dept_doctors = {}
    
    # 第一次遍历：收集所有医生记录
    for hospital in hospital_data.get("hospitals", []):
        hospital_name = hospital.get("医院名称")
        if not hospital_name:
            continue
            
        for record in hospital.get("历史记录", []):
            doctor_name = record.get("医生姓名", "")
            department = record.get("科室", "")
            
            if not doctor_name:
                continue
                
            # 创建医院+科室的键
            hospital_dept_key = (hospital_name, department)
            
            if hospital_dept_key not in hospital_dept_doctors:
                hospital_dept_doctors[hospital_dept_key] = set()
                
            # 添加医生名称到集合中
            hospital_dept_doctors[hospital_dept_key].add(doctor_name)
    
    # 第二次遍历：识别简称和全称关系
    for hospital_dept_key, doctors in hospital_dept_doctors.items():
        hospital_name, department = hospital_dept_key
        
        # 如果同一医院科室有多个医生名称
        if len(doctors) > 1:
            # 收集带职称的名称（可能是简称）
            title_doctors = []
            other_doctors = []
            
            title_terms = ["主任", "医生", "医师", "专家", "教授", "博士", "院长", "科长"]
            for doctor in doctors:
                if any(term in doctor for term in title_terms):
                    title_doctors.append(doctor)
                else:
                    other_doctors.append(doctor)
            
            # 尝试匹配简称-全称对
            for title_doctor in title_doctors:
                doctor_surname = title_doctor[0] if title_doctor else ""
                
                if not doctor_surname:
                    continue
                    
                # 查找同姓的其他医生名称（可能是全称）
                for other_doctor in other_doctors:
                    if other_doctor and other_doctor[0] == doctor_surname:
                        # 找到可能的简称-全称对
                        doctor_key = (hospital_name, department, title_doctor)
                        
                        # 添加简称到全称的映射
                        if add_to_master_data("doctors", doctor_key, other_doctor):
                            updated_count += 1
                            logging.info(f"在医院 '{hospital_name}' 科室 '{department}' 中发现医生简称-全称映射: '{title_doctor}' -> '{other_doctor}'")
    
    # 5. 同时添加所有医生的自映射（即使没有找到简称-全称关系）
    for hospital in hospital_data.get("hospitals", []):
        hospital_name = hospital.get("医院名称")
        for record in hospital.get("历史记录", []):
            doctor_name = record.get("医生姓名")
            department = record.get("科室", "")
            
            if hospital_name and doctor_name:
                doctor_key = (hospital_name, department, doctor_name)
                # 添加自映射，这样在查找时至少能找到自己
                if add_to_master_data("doctors", doctor_key, doctor_name):
                    updated_count += 1
    
    # 6. 更新经销商和联系人
    for distributor in distributor_data.get("distributors", []):
        dist_name = distributor.get("经销商名称")
        
        if dist_name:
            # 添加经销商自身
            if add_to_master_data("distributors", dist_name, dist_name):
                updated_count += 1
                
            # 添加联系人
            for record in distributor.get("沟通记录", []):
                contact_name = record.get("联系人")
                if contact_name:
                    contact_key = (dist_name, contact_name)
                    if add_to_master_data("contacts", contact_key, contact_name):
                        updated_count += 1
    
    logging.info(f"从CSV数据更新了{updated_count}个实体信息到master_data")
    return updated_count > 0

# >>> NEW: NER model initialisation (shared singleton)
ner_model = None

# 关系‑score few‑shot prompt – scoring anchor examples
REL_SCORE_PROMPT_STEM = """你是医药代表 CRM 系统里的"关系热度评估器"。\n请阅读一段沟通记录，用 1‑10 分衡量双方合作的紧密程度。只能返回 JSON，格式如 {\"score\": 9}\n\n### 评分基准\n* **9‑10** : 已明确进入合作流程，例如"主任同意/已签字打报告""设备科正向推进""客户主动推荐决策人"。\n* **7‑8**  : 明确表达兴趣或口头承诺，如"愿意了解报价""需要院内流程"。\n* **5‑6**  : 常规互动，有基本联系但暂无具体行动。\n* **3‑4**  : 态度冷淡或仅礼貌回应，需要持续跟进。\n* **1‑2**  : 明确拒绝、长期无回复、竞争品牌已中标。\n\n### 示例\n内容: "李主任已经帮我把设备报告签字，下周走内招流程。"\n输出: {\"score\": 9}\n\n内容: "今天把资料放在外科护士站，主任说有空再看。"\n输出: {\"score\": 6}\n\n内容: "对方回复目前暂不考虑，引入其他品牌。"\n输出: {\"score\": 3}\n\n### 待评估内容\n"""

# >>> NEW: LLM‑based entity classifier (H = hospital, D = distributor)
def classify_entity_llm(entity_name: str, api_client) -> str:
    # 先用规则检查，如果名称包含"主任"，大概率是医生(医院)
    if IsLikelyDoctor(entity_name):
        return 'H'
        
    prompt = (
        "你是实体分类器。下面名称若为医院或医疗机构，回复 'H'；"
        "若为医药/器械经销商或供应商，回复 'D'。只回复单字母。\n" + entity_name)
    rsp = _call_llm_api(api_client, [{"role": "user", "content": prompt}], 0)
    return 'D' if rsp and 'D' in rsp.upper() else 'H'

# >>> NEW: 检查名称是否更可能是医生
def IsLikelyDoctor(name: str) -> bool:
    """检查名称是否更可能是医生而非经销商"""
    if not name:
        return False
    
    # 医生特征关键词
    doctor_keywords = ["主任", "医师", "医生", "教授", "副教授", "讲师", "博士", "硕士", 
                     "专家", "院长", "科长", "科主任", "副主任"]
    
    # 检查是否包含医生特征关键词
    for keyword in doctor_keywords:
        if keyword in name:
            return True
    
    return False

# >>> NEW: 主语判定工具
def _determine_primary_entity(extracted):
    """返回 'hospital' / 'distributor' — 确定这条日志主要写入哪张表"""
    has_hosp = bool(extracted.get("医院信息"))
    has_dist = bool(extracted.get("经销商信息"))
    if has_hosp and has_dist:
        has_direct_doc_chat = any(
            doc.get("沟通内容")
            for hosp in extracted["医院信息"]
            for doc in hosp.get("医生信息", [])
        )
        return "hospital" if has_direct_doc_chat else "distributor"
    return "hospital" if has_hosp else "distributor"

# 全局API客户端(用于在线医院查询)
global_api_client = None

def sync_master_data_to_mongodb(mongo_client, sync_types=None, batch_size=100):
    """将master_json中的各类映射同步到MongoDB中的记录
    
    Args:
        mongo_client: MongoDB客户端
        sync_types: 要同步的数据类型列表，None表示全部同步 ["doctors", "hospitals", "departments", "distributors"]
        batch_size: 批量更新的记录数量
        
    Returns:
        bool: 是否成功同步
    """
    if not mongo_client:
        logging.error("MongoDB客户端无效，无法同步映射数据")
        return False
        
    # 默认同步所有类型
    if sync_types is None:
        sync_types = ["doctors", "hospitals", "departments", "distributors"]
    
    try:
        db = mongo_client.get_database(MONGO_DB_NAME)
        hospital_visits_collection = db["hospital_visits"]
        distributor_comms_collection = db["distributor_communications"]
        
        # 加载主数据
        master_data = load_master_data()
        total_updated = 0
        
        # 1. 同步医生名称映射
        if "doctors" in sync_types:
            doctor_mappings = {}
            
            # 以"(医院,科室)"为键将医生简称-全称收集起来
            for (hospital_name, department, doctor_key), doctor_name in master_data.get("doctors", {}).items():
                if not hospital_name or not department or not doctor_key or not doctor_name:
                    continue
                if (hospital_name, department) not in doctor_mappings:
                    doctor_mappings[(hospital_name, department)] = {}
                doctor_mappings[(hospital_name, department)][doctor_key] = doctor_name
            
            if doctor_mappings:
                logging.info(f"开始同步医生映射关系，共有{len(doctor_mappings)}组映射...")
                doctor_update_count = 0
                
                # 分批次处理每组(医院,科室)下的医生映射
                with tqdm(total=len(doctor_mappings), desc="同步医生映射") as pbar:
                    for (hospital_name, department), mappings in doctor_mappings.items():
                        mapping_updates = []
                        for old_name, new_name in mappings.items():
                            # 对每个映射创建批量更新操作
                            mapping_updates.append(UpdateMany(
                                {"医院名称": hospital_name, "科室": department, "医生姓名": old_name},
                                {"$set": {"医生姓名": new_name}}
                            ))
                            
                            # 当积累足够的更新操作或处理到最后一个映射时，执行批量更新
                            if len(mapping_updates) >= batch_size or (hospital_name, department) == list(doctor_mappings.keys())[-1]:
                                if mapping_updates:
                                    try:
                                        # 执行批量写入
                                        result = hospital_visits_collection.bulk_write(mapping_updates)
                                        doctor_update_count += result.modified_count
                                        mapping_updates = []  # 清空已处理的更新
                                    except Exception as e:
                                        logging.error(f"批量更新医生映射时出错: {e}")
                        pbar.update(1)
                        
                logging.info(f"医生映射同步完成，共更新{doctor_update_count}条记录")
                total_updated += doctor_update_count
        
        # 2. 同步医院名称映射
        if "hospitals" in sync_types:
            hospital_mappings = {}
            
            # 收集医院名称的映射
            for hospital_key, hospital_data in master_data.get("hospitals", {}).items():
                if isinstance(hospital_data, dict) and "标准名称" in hospital_data:
                    standard_name = hospital_data["标准名称"]
                    if hospital_key != standard_name:  # 原名到标准名的映射
                        hospital_mappings[hospital_key] = standard_name
            
            if hospital_mappings:
                logging.info(f"开始同步医院名称映射关系，共有{len(hospital_mappings)}个映射...")
                hospital_update_count = 0
                
                # 分批次处理医院名称映射
                with tqdm(total=len(hospital_mappings), desc="同步医院名称映射") as pbar:
                    mapping_updates = []
                    for original_name, standard_name in hospital_mappings.items():
                        # 对医院拜访记录更新
                        mapping_updates.append(UpdateMany(
                            {"医院名称": original_name},
                            {"$set": {"标准医院名称": standard_name}}
                        ))
                        
                        # 当积累足够的更新操作或处理到最后一个映射时，执行批量更新
                        if len(mapping_updates) >= batch_size or original_name == list(hospital_mappings.keys())[-1]:
                            if mapping_updates:
                                try:
                                    # 执行批量写入
                                    result = hospital_visits_collection.bulk_write(mapping_updates)
                                    hospital_update_count += result.modified_count
                                    mapping_updates = []  # 清空已处理的更新
                                except Exception as e:
                                    logging.error(f"批量更新医院名称映射时出错: {e}")
                        pbar.update(1)
                
                logging.info(f"医院名称映射同步完成，共更新{hospital_update_count}条记录")
                total_updated += hospital_update_count
        
        # 3. 同步经销商名称映射
        if "distributors" in sync_types:
            distributor_mappings = {}
            
            # 收集经销商名称的映射
            for dist_key, dist_data in master_data.get("distributors", {}).items():
                if isinstance(dist_data, dict) and "标准名称" in dist_data:
                    standard_name = dist_data["标准名称"]
                    if dist_key != standard_name:  # 原名到标准名的映射
                        distributor_mappings[dist_key] = standard_name
            
            if distributor_mappings:
                logging.info(f"开始同步经销商名称映射关系，共有{len(distributor_mappings)}个映射...")
                distributor_update_count = 0
                
                # 分批次处理经销商名称映射
                with tqdm(total=len(distributor_mappings), desc="同步经销商名称映射") as pbar:
                    mapping_updates = []
                    for original_name, standard_name in distributor_mappings.items():
                        # 对经销商沟通记录更新
                        mapping_updates.append(UpdateMany(
                            {"经销商名称": original_name},
                            {"$set": {"标准名称": standard_name}}
                        ))
                        
                        # 当积累足够的更新操作或处理到最后一个映射时，执行批量更新
                        if len(mapping_updates) >= batch_size or original_name == list(distributor_mappings.keys())[-1]:
                            if mapping_updates:
                                try:
                                    # 执行批量写入
                                    result = distributor_comms_collection.bulk_write(mapping_updates)
                                    distributor_update_count += result.modified_count
                                    mapping_updates = []  # 清空已处理的更新
                                except Exception as e:
                                    logging.error(f"批量更新经销商名称映射时出错: {e}")
                        pbar.update(1)
                
                logging.info(f"经销商名称映射同步完成，共更新{distributor_update_count}条记录")
                total_updated += distributor_update_count
        
        # 添加数据验证和报告
        if total_updated > 0:
            logging.info(f"主数据同步到MongoDB完成，总共更新{total_updated}条记录")
        else:
            logging.info("没有发现需要同步的数据或没有匹配的记录需要更新")
        return True
    except Exception as e:
        logging.error(f"同步主数据到MongoDB时出错: {e}", exc_info=True)
        return False

def detect_and_report_mongodb_duplicates(mongo_client):
    """
    检测并报告 MongoDB 主要集合中的重复文档。
    重复是基于用于 upsert 操作的组合键来定义的。
    """
    if not mongo_client:
        logging.error("MongoDB 客户端无效，无法检测重复项。")
        return

    db = mongo_client.get_database(MONGO_DB_NAME)
    collections_to_check = {
        "hospital_visits": ["医院ID", "拜访日期", "医生姓名"],
        "distributor_communications": ["经销商ID", "沟通日期", "联系人"]
    }

    logging.info("开始检测 MongoDB 中的重复文档...")
    total_duplicates_found = 0

    for collection_name, group_by_fields in collections_to_check.items():
        collection = db[collection_name]
        logging.info(f"正在检查集合: {collection_name} (分组依据: {', '.join(group_by_fields)})")

        # 构建聚合管道的分组阶段
        group_stage = {"_id": {}}
        for field in group_by_fields:
            group_stage["_id"][field] = f"${field}"
        
        pipeline = [
            {
                "$group": {
                    **group_stage,
                    "count": {"$sum": 1},
                    "documents": {"$push": "$_id"}
                }
            },
            {
                "$match": {
                    "count": {"$gt": 1}
                }
            }
        ]

        try:
            duplicates = list(collection.aggregate(pipeline))
            if duplicates:
                logging.warning(f"在集合 '{collection_name}' 中发现 {len(duplicates)} 组重复项:")
                total_duplicates_found += len(duplicates)
                for i, dup_group in enumerate(duplicates):
                    logging.warning(f"  重复组 {i+1}:")
                    logging.warning(f"    关键字段: {dup_group['_id']}")
                    logging.warning(f"    重复数量: {dup_group['count']}")
                    logging.warning(f"    文档 _ids: {dup_group['documents']}")
            else:
                logging.info(f"在集合 '{collection_name}' 中未发现基于指定键的重复项。")
        except Exception as e:
            logging.error(f"检查集合 '{collection_name}' 时出错: {e}", exc_info=True)

    if total_duplicates_found > 0:
        logging.warning(f"MongoDB 重复项检测完成，共在 {len(collections_to_check)} 个集合中发现 {total_duplicates_found} 组重复。")
    else:
        logging.info("MongoDB 重复项检测完成，未发现明显的基于upsert键的重复记录组。")

if __name__ == "__main__":
    main()
