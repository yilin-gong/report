# -*- coding: utf-8 -*-
import streamlit as st
import json
import os
import datetime
import time
import threading
import queue
import concurrent.futures
import shutil
import uuid
from typing import Dict, List, Any, Optional
from openai import OpenAI
from dotenv import load_dotenv
import traceback # 用于详细错误日志
from pymongo import MongoClient
import urllib.parse
import ast
import pandas as pd
# 导入MongoDB连接模块
import mongodb_connector
# 导入新模块
import new_report

st.set_page_config(layout="wide", page_title="医疗销售报告分析器") # 翻译 # MOVED HERE

# --- 配置与初始化 ---

# 确保数据目录存在
os.makedirs("data", exist_ok=True)
CONFIG_FILE = os.path.join("data", "config.json")
DEFAULT_HOSPITAL_DATA = os.path.join("data", "hospital_data.json")
DEFAULT_DISTRIBUTOR_DATA = os.path.join("data", "distributor_data.json")

# 从mongodb_connector导入连接器
from mongodb_connector import mongodb_connector, MONGO_DB_NAME, MONGO_HOSPITALS_COLLECTION, MONGO_DISTRIBUTORS_COLLECTION

# 加载 .env 文件中的环境变量 (用于 API 密钥)
load_dotenv()

# 初始化 OpenAI 客户端
# 优先级：1.用户配置的API密钥 2.Streamlit secrets 3.环境变量
# openai_api_key = None # 旧的初始化方式
# openai_base_url = "https://ark.cn-beijing.volces.com/api/v3"

# # 加载配置文件中的非敏感配置，但不用于API密钥或会话数据库密码的初始化
# # API密钥和会话数据库密码由secrets, env, 或用户会话输入处理
# # if os.path.exists(CONFIG_FILE):
# #     try:
# #         with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
# #             config = json.load(f)
# #             # user_api_key_from_config = config.get("user_api_key", "") # Removed this line
# #             # if user_api_key_from_config and user_api_key_from_config.strip(): # Removed this block
# #             #     openai_api_key = user_api_key_from_config.strip() # Removed this line
# #     except Exception as e:
# #         print(f"加载配置文件时出错 (仅用于读取非敏感数据，API密钥不从此加载): {e}") # Modified log

# # 如果配置文件中没有API密钥 (现在总是如此，因为我们不再从config加载API密钥到openai_api_key变量),
# # 尝试从Streamlit secrets获取
# # if not openai_api_key: # This condition effectively becomes true unless secrets/env below set it
# try:
#     # 尝试访问 Streamlit secrets (用于部署)
#     openai_api_key = st.secrets["OPENAI_API_KEY"]
#     openai_base_url = st.secrets.get("OPENAI_BASE_URL", openai_base_url)
# except (FileNotFoundError, KeyError):
#     # 本地开发回退到环境变量
#     openai_api_key = os.getenv('OPENAI_API_KEY', None)
#     if os.getenv('OPENAI_BASE_URL'):
#         openai_base_url = os.getenv('OPENAI_BASE_URL')

# client = None
# # 翻译：API Key 未配置警告
# api_key_warning_message_cn = "OpenAI API 密钥未配置。分析功能将无法工作。请在\"配置\"页面输入您的API密钥，或者通过 .env 文件或 Streamlit secrets 配置。"

# if openai_api_key:
#     try:
#         client = OpenAI(api_key=openai_api_key, base_url=openai_base_url)
#     except Exception as e:
#         # 翻译：初始化失败错误
#         st.error(f"初始化 OpenAI 客户端失败: {e}")
# else:
#     # 只在客户端未能成功初始化时显示警告
#     if not client:
#       st.warning(api_key_warning_message_cn, icon="⚠️")

# --- 新的 OpenAI 客户端初始化逻辑 ---
# 确定 openai_base_url (默认 -> 环境变量 -> Secrets)
openai_base_url = "https://ark.cn-beijing.volces.com/api/v3" # 默认值
if os.getenv('OPENAI_BASE_URL'):
    openai_base_url = os.getenv('OPENAI_BASE_URL')
try:
    # Streamlit secrets (st.secrets) 只能在 Streamlit Cloud 部署环境中使用，
    # 或者在本地通过 secrets.toml 文件配置。
    # 在本地开发且没有 secrets.toml 时，st.secrets 会引发错误或返回空。
    # 我们需要确保 st.secrets 的访问是安全的。
    if hasattr(st, 'secrets'):
      openai_base_url = st.secrets.get("OPENAI_BASE_URL", openai_base_url)
except (AttributeError, FileNotFoundError, KeyError): # FileNotFoundError for secrets.toml, KeyError for specific secret
    pass # 保持之前确定的 openai_base_url

# 确定 openai_api_key (会话状态 -> Secrets -> 环境变量)
openai_api_key = None
if 'user_api_key' in st.session_state and st.session_state.user_api_key and st.session_state.user_api_key.strip():
    openai_api_key = st.session_state.user_api_key.strip()
else:
    try:
        if hasattr(st, 'secrets'):
          openai_api_key = st.secrets.get("OPENAI_API_KEY") # 使用 .get() 避免 KeyError
    except (AttributeError, FileNotFoundError): # AttributeError if st.secrets doesn't exist
        pass # 继续尝试环境变量
    
    if not openai_api_key: # 如果 secrets 中没有或无法访问
        openai_api_key = os.getenv('OPENAI_API_KEY', None)

client = None
api_key_warning_message_cn = "OpenAI API 密钥未配置。分析功能将无法工作。请在\"配置\"页面输入您的API密钥，或者通过 .env 文件或 Streamlit secrets 配置。"

if openai_api_key:
    try:
        client = OpenAI(api_key=openai_api_key, base_url=openai_base_url)
    except Exception as e:
        st.error(f"初始化 OpenAI 客户端失败: {e}")
else:
    # 只有在 openai_api_key 为空 (即 client 未能初始化) 时显示警告
    st.warning(api_key_warning_message_cn, icon="⚠️")
# --- 结束新的 OpenAI 客户端初始化逻辑 ---

# --- 后端函数 (大部分不变, 适配 Streamlit 状态/日志) ---

# 初始化 Session State (对 Streamlit 很重要)
if 'process_logs' not in st.session_state:
    st.session_state.process_logs = []
if 'max_workers' not in st.session_state:
    st.session_state.max_workers = 5
if 'use_multithreading_default' not in st.session_state:
    st.session_state.use_multithreading_default = True
if 'hospital_data_path' not in st.session_state:
    st.session_state.hospital_data_path = DEFAULT_HOSPITAL_DATA
if 'distributor_data_path' not in st.session_state:
    st.session_state.distributor_data_path = DEFAULT_DISTRIBUTOR_DATA
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None # 存储字典 {text: ..., html: ..., message: ...}
if 'search_results_html' not in st.session_state:
    st.session_state.search_results_html = None
if 'user_api_key' not in st.session_state:
    st.session_state.user_api_key = "" # 用户输入的API密钥
if 'db_password' not in st.session_state:  # Add this back
    st.session_state.db_password = "" # 数据库密码
if 'use_mongodb' not in st.session_state:
    st.session_state.use_mongodb = True # 默认使用MongoDB

def init_client_with_key(api_key):
    """使用给定的API密钥初始化OpenAI客户端"""
    global client
    try:
        client = OpenAI(api_key=api_key, base_url=openai_base_url)
        return True, ""
    except Exception as e:
        return False, str(e)

def load_config():
    """从文件加载配置到 session state。"""
    try:
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                config = json.load(f)
                st.session_state.max_workers = config.get("max_workers", 5)
                st.session_state.use_multithreading_default = config.get("use_multithreading", True)
                # st.session_state.user_api_key = config.get("user_api_key", "") # Removed
                st.session_state.use_mongodb = config.get("use_mongodb", True)
    except Exception as e:
        # 翻译：加载配置警告
        st.warning(f"无法加载配置文件 ({CONFIG_FILE}): {e}")

def save_config():
    """从 session state 保存配置到文件。"""
    config = {
        "max_workers": st.session_state.max_workers,
        "use_multithreading": st.session_state.use_multithreading_default,
        # "user_api_key": st.session_state.user_api_key, # Removed
        "use_mongodb": st.session_state.use_mongodb
    }
    try:
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        # 翻译：保存成功消息
        st.success(f"配置已保存: 工作线程数={st.session_state.max_workers}, 默认多线程={st.session_state.use_multithreading_default}, 数据库={('MongoDB' if st.session_state.use_mongodb else '本地文件')}")
    except Exception as e:
        # 翻译：保存失败错误
        st.error(f"保存配置时出错: {e}")

# 启动时加载配置
load_config()

# 如果默认数据文件不存在则创建空文件
for file_path in [DEFAULT_HOSPITAL_DATA, DEFAULT_DISTRIBUTOR_DATA]:
    if not os.path.exists(file_path):
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                if 'hospital' in file_path:
                    json.dump({"hospitals": []}, f, ensure_ascii=False, indent=2)
                else:
                    json.dump({"distributors": []}, f, ensure_ascii=False, indent=2)
        except Exception as e:
            # 翻译：创建文件错误
            st.error(f"无法创建默认数据文件 {file_path}: {e}")


def add_process_log(message: str) -> None:
    """向 session state 日志列表添加条目。[内部消息保持英文或按需翻译]"""
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    # 保持核心日志信息为英文可能有助于调试，但可以在format时翻译
    log_entry = f"[{timestamp}] {message}"
    
    # 检查是否在主线程中运行（有Streamlit上下文）
    try:
        if 'process_logs' not in st.session_state:
            st.session_state.process_logs = []
        # 避免重复添加完全相同的日志条目（例如在快速重新渲染中）
        if not st.session_state.process_logs or st.session_state.process_logs[-1] != log_entry:
            st.session_state.process_logs.append(log_entry)
    except:
        # 在子线程中，只打印日志但不尝试访问st.session_state
        pass
    
    # 控制台日志总是打印，无论是否在主线程中
    print(f"Process log: {log_entry}")

def get_process_logs_str() -> str:
    """从 session state 获取所有日志字符串。"""
    if 'process_logs' not in st.session_state:
         return ""
    return "\n".join(st.session_state.process_logs)

def clear_process_logs() -> None:
    """清空 session state 中的日志。"""
    st.session_state.process_logs = []

def format_process_details_html() -> str:
    """将 session state 中的日志格式化为 HTML。[翻译标题和无记录提示]"""
    html_output = ["<div style='padding: 10px; background-color: #f8f9fa; border-radius: 5px; border: 1px solid #dee2e6;'>"]
    html_output.append("<h6>处理步骤 (Analysis Steps)</h6>") # 翻译
    html_output.append("<ul style='font-size: 0.9em; margin-left: -15px;'>")

    logs = st.session_state.get('process_logs', [])
    if not logs:
        html_output.append("<li>尚无处理步骤记录。</li>") # 翻译

    for log in logs:
        parts = log.split("] ", 1)
        if len(parts) == 2:
            timestamp = parts[0] + "]"
            message = parts[1] # 日志消息本身可以保持英文或选择性翻译
            html_output.append(f"<li style='margin-bottom: 3px;'><strong style='color: #007bff;'>{timestamp}</strong> {message}</li>") # Display original log message
        else:
            html_output.append(f"<li style='margin-bottom: 3px;'>{log}</li>")

    html_output.append("</ul>")
    html_output.append("</div>")
    return "".join(html_output)

# --- LLM 和数据处理函数 (确保错误消息和部分日志是中文) ---

def extract_entities_with_llm(daily_report_text: str) -> Dict[str, Any]:
    """ 使用 LLM 提取实体。[翻译错误消息] """
    if not client:
        # 这个函数可能在线程中被调用，所以不能直接使用st.error
        print("OpenAI 客户端未初始化。无法提取实体。")
        return {"hospitals": [], "doctors": [], "departments": [], "products": [], "distributors": []}
    try:
        # 使用print而不是add_process_log，避免在线程中访问st.session_state
        print("Starting entity extraction from daily report.")
        # System prompt 已在之前修改为要求处理中文内容
        prompt_system = """你是一位医疗销售分析专家。请分析以下**中文日报内容**，提取所有提到的：
1.  医院名称
2.  医生/主任/联系人姓名及其角色/职位和所属医院(如报告中提到)
3.  科室名称
4.  医疗产品名称
5.  经销商公司名称及联系人姓名

请严格按照以下 JSON 格式返回，确保列表中的值为报告中出现的原始文本（通常是中文）:
{
    "hospitals": ["医院名称1", "医院名称2", ...],
    "doctors": [{"name": "医生姓名1", "role": "角色1", "hospital": "所属医院名称1"}, {"name": "姓名2", "role": "角色2", "hospital": "所属医院名称2"}, ...],
    "departments": ["科室名称1", "科室名称2", ...],
    "products": ["产品名称1", "产品名称2", ...],
    "distributors": [{"name": "联系人姓名1", "company": "公司名称1"}, {"name": "姓名2", "company": "公司2"}, ...]
}
如果某项信息在报告中未提及，请返回空字符串 ""。例如，如果报告中只提到医生姓名，但未提到所属医院，则 "hospital" 字段应为 ""。如果某类别信息在报告中未提及，请返回空列表 `[]`。"""

        response = client.chat.completions.create(
            model="doubao-1-5-pro-32k-250115",
            messages=[
                {"role": "system", "content": prompt_system},
                {"role": "user", "content": daily_report_text}
            ],
            response_format={"type": "json_object"}
        )
        try:
             result = json.loads(response.choices[0].message.content)
        except json.JSONDecodeError:
             print(f"Entity extraction JSON parsing failed. Raw: {response.choices[0].message.content[:100]}...")
             result = {"hospitals": [], "doctors": [], "departments": [], "products": [], "distributors": []}

        # 内部日志可以保持英文或添加中文
        log_h = len(result.get('hospitals', []))
        log_d = len(result.get('doctors', []))
        log_di = len(result.get('distributors', []))
        print(f"Entity extraction successful: Found {log_h} hospitals, {log_d} doctors/contacts, {log_di} distributors.")
        return result
    except Exception as e:
        error_msg = f"实体提取过程中出错: {str(e)}"
        print(f"Error during entity extraction: {str(e)}")
        print(traceback.format_exc())
        return {"hospitals": [], "doctors": [], "departments": [], "products": [], "distributors": []}


def evaluate_with_llm(text: str, context: Dict[str, str]) -> Dict[str, str]:
    """ 使用 LLM 评估关系。[Prompt 已修改为中文, 翻译错误消息] """
    if not client:
        # 在线程中运行时避免使用st.error
        print("OpenAI 客户端未初始化。无法评估关系。")
        return {"status": "错误", "analysis": "OpenAI 客户端不可用。"}

    entity_name = context.get('name', '未知联系人')
    entity_role = context.get('role', '未知角色')

    if not text:
        log_msg = f"未找到与 {entity_name} 的历史沟通记录，无法评估关系。"
        print(f"No historical communication record found for {entity_name}. Cannot evaluate relationship.")
        return {"status": "未知", "analysis": log_msg}

    try:
        print(f"Starting relationship evaluation for {entity_name} ({entity_role}).")
        prompt = f'''
        作为一名医疗销售领域的关系分析专家，请基于以下的沟通记录，评估与联系人 **{entity_name} ({entity_role})** 的当前关系状态。

        沟通记录: "{text}"

        请从以下几个方面进行**中文分析**:
        1. 对方对产品的接受度和满意度
        2. 沟通中反映的合作意愿和积极性
        3. 存在的问题或障碍
        4. 关系发展的潜力
        5. 需要关注的风险点

        请给出**中文的关系状态评估** (从以下选项中选择一个最合适的: **良好、偏向积极、中性、偏向消极、需改善、问题严重、新接触、未知**)
        以及**详细的中文分析**。请将分析内容限制在150字以内。

        **请严格按照以下 JSON 格式返回，确保 status 和 analysis 的值是中文**:
        {{"status": "关系状态中文值", "analysis": "详细中文分析"}}
        '''
        response = client.chat.completions.create(
            model="doubao-1-5-pro-32k-250115", # 确认模型支持流畅的中文输出
            messages=[
                {"role": "system", "content": "你是一位医疗销售关系分析专家，擅长从沟通记录中评估业务关系并使用中文进行分析。"}, # System prompt 也提示中文
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )
        try:
            result = json.loads(response.choices[0].message.content)
            if not isinstance(result.get("status"), str): result["status"] = "解析状态失败"
            if not isinstance(result.get("analysis"), str): result["analysis"] = "解析分析失败"
        except json.JSONDecodeError:
             print(f"Relationship evaluation JSON parsing failed for {entity_name}. Raw: {response.choices[0].message.content[:100]}...")
             result = {"status": "格式错误", "analysis": "LLM 返回的 JSON 格式无效。"}

        print(f"Relationship evaluation complete for {entity_name}. Status: {result.get('status', '未知')}")
        return result
    except Exception as e:
        error_msg = f"评估与 {entity_name} 的关系时出错: {str(e)}"
        print(f"Error during relationship evaluation for {entity_name}: {str(e)}")
        print(traceback.format_exc())
        return {
            "status": "评估出错",
            "analysis": f"评估与 {entity_name} 的关系时系统出错，请检查日志。"
        }

def generate_suggested_actions(evaluation_result: Dict[str, str], entity_info: Dict[str, str] = None) -> List[str]:
    """ 生成建议行动。[Prompt 已修改为中文, 翻译错误消息和回退建议] """
    if not client:
        # 在线程中运行时避免使用st.error
        print("OpenAI 客户端未初始化。无法生成建议行动。")
        return ["错误：OpenAI 客户端不可用。"]

    if entity_info is None: entity_info = {}
    entity_name = entity_info.get('name', '未知联系人')
    entity_role = entity_info.get('role', '未知角色')

    try:
        print(f"Starting action suggestion generation for {entity_name} ({entity_role}).")
        prompt = f'''
        基于以下关系评估结果，请为医疗销售代表提供3-5条具体、可行的**中文后续行动建议**:

        人员信息:
        - 姓名: {entity_name}
        - 角色/公司: {entity_role}

        关系评估:
        - 状态: {evaluation_result.get('status', '未知')}
        - 分析: {evaluation_result.get('analysis', '无分析内容')}

        请提供**具体、可执行的中文建议**，避免泛泛而谈。每一条建议应清晰说明要做什么，为什么做（目的），以及预期的效果。
        请确保每条建议单独成行，易于阅读。
        '''
        response = client.chat.completions.create(
            model="doubao-1-5-pro-32k-250115", # 确认模型支持流畅的中文输出
            messages=[
                {"role": "system", "content": "你是一位医疗销售策略专家，擅长根据关系评估提供具体、可行的中文行动建议。"}, # System prompt 也提示中文
                {"role": "user", "content": prompt}
            ]
        )
        suggestions = [line.strip() for line in response.choices[0].message.content.split("\n") if line.strip() and len(line.strip()) > 5]
        print(f"Generated {len(suggestions[:5])} action suggestions for {entity_name}.")
        return suggestions[:5]  # 限制返回5条建议
    except Exception as e:
        error_msg = f"为 {entity_name} 生成建议时出错: {str(e)}"
        print(f"Error generating action suggestions for {entity_name}: {str(e)}")
        print(traceback.format_exc())
        return [f"建议生成出错: {str(e)}", "请检查系统日志获取更多信息。"]


def load_json_data(file_path: str) -> Dict[str, Any]:
    """安全地加载数据（从本地文件或MongoDB）。[翻译警告和错误]"""
    # 如果使用MongoDB，从数据库加载数据
    if st.session_state.use_mongodb:
        # 检查MongoDB连接
        try:
            # 检查mongodb_connector是否有db属性
            if not hasattr(mongodb_connector, 'db') or mongodb_connector.db is None:
                if st.session_state.db_password:
                    success, error = mongodb_connector.connect(st.session_state.db_password)
                    if not success:
                        st.error(f"连接MongoDB失败: {error}")
                        # 回退到本地文件
                        st.warning("将使用本地文件作为回退选项")
                        return load_local_json_data(file_path)
                else:
                    st.error("MongoDB密码未配置，无法连接数据库")
                    # 回退到本地文件
                    st.warning("将使用本地文件作为回退选项")
                    return load_local_json_data(file_path)
            
            # 从MongoDB加载数据
            if "hospital" in file_path:
                return mongodb_connector.get_hospital_data()
            else:
                return mongodb_connector.get_distributor_data()
        except Exception as e:
            st.error(f"访问MongoDB时出错: {str(e)}")
            st.warning("将使用本地文件作为回退选项")
            return load_local_json_data(file_path)
    else:
        # 使用本地文件
        return load_local_json_data(file_path)

def load_local_json_data(file_path: str) -> Dict[str, Any]:
    """安全地加载本地JSON数据文件。[翻译警告和错误]"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        st.warning(f"数据文件未找到: {file_path}。将返回空数据。") # 翻译
        if 'hospital' in file_path: return {"hospitals": []}
        else: return {"distributors": []}
    except json.JSONDecodeError:
        st.error(f"文件 JSON 格式无效: {file_path}。将返回空数据。") # 翻译
        if 'hospital' in file_path: return {"hospitals": []}
        else: return {"distributors": []}
    except Exception as e:
        st.error(f"加载数据文件 {file_path} 时出错: {e}") # 翻译
        print(traceback.format_exc())
        if 'hospital' in file_path: return {"hospitals": []}
        else: return {"distributors": []}

# --- 搜索和格式化函数 (搜索结果表格标题翻译) ---

def search_doctors_in_hospital_data(hospital_data: Dict[str, Any], doctor_names: List[str]) -> Dict[str, List[Dict[str, str]]]:
    """在医院数据中查找指定医生的最新记录。"""
    results = {}
    
    # 如果使用MongoDB并且连接可用
    try:
        mongodb_db_exists = hasattr(mongodb_connector, 'db') and mongodb_connector.db is not None
        if st.session_state.use_mongodb and mongodb_db_exists:
            for doctor in doctor_names:
                # 支持直接传递字符串名称或包含name和hospital的字典
                doctor_name = doctor["name"] if isinstance(doctor, dict) else doctor
                doctor_hospital = doctor.get("hospital", "") if isinstance(doctor, dict) else ""
                
                # 构建MongoDB查询条件 - 使用正则表达式实现模糊匹配
                query = {"医生姓名": {"$regex": doctor_name, "$options": "i"}}
                
                # 如果指定了医院，增加医院名称条件
                if doctor_hospital:
                    # 改进：同时匹配"医院名称"和"标准医院名称"
                    hospital_pattern = {"$regex": doctor_hospital, "$options": "i"}
                    query["$or"] = [
                        {"医院名称": hospital_pattern},
                        {"标准医院名称": hospital_pattern}
                    ]
                
                # 查询医生记录
                doctor_records = list(mongodb_connector.db[MONGO_HOSPITALS_COLLECTION].find(
                    query, 
                    {"_id": 0} # 确保返回所有需要的字段
                ).sort("拜访日期", -1).limit(1))
                
                # 调试日志
                add_process_log(f"MongoDB查询医生: {doctor_name}, 找到记录数: {len(doctor_records)}")
                
                # 格式化结果
                if doctor_records:
                    record = doctor_records[0]
                    results[doctor_name] = [{
                        "医院": record.get("医院名称", ""),
                        "科室": record.get("科室", ""),
                        "拜访日期": record.get("拜访日期", ""),
                        "拜访员工": record.get("拜访员工", ""),
                        "沟通内容": record.get("沟通内容", ""),
                        "后续行动": record.get("后续行动", "")
                    }]
        else:
            # 如果不使用MongoDB或连接不可用，使用原来的方法
            for doctor in doctor_names:
                # 支持直接传递字符串名称或包含name和hospital的字典
                doctor_name = doctor["name"] if isinstance(doctor, dict) else doctor
                doctor_hospital = doctor.get("hospital", "") if isinstance(doctor, dict) else ""
                
                all_records = []
                for hospital in hospital_data.get("hospitals", []):
                    hospital_name = hospital.get("医院名称", "")
                    
                    # 如果医生有指定医院且当前医院不匹配，则跳过
                    if doctor_hospital and hospital_name and not (
                        doctor_hospital.lower() in hospital_name.lower() or 
                        hospital_name.lower() in doctor_hospital.lower()
                    ):
                        continue
                        
                    for record in hospital.get("历史记录", []):
                        record_doctor_name = record.get("医生姓名", "")
                        # 允许部分匹配以提高灵活性
                        if record_doctor_name and (doctor_name.lower() in record_doctor_name.lower() or record_doctor_name.lower() in doctor_name.lower()):
                            all_records.append({
                                "医院": hospital.get("医院名称", ""),
                                "科室": record.get("科室", ""),
                                "拜访日期": record.get("拜访日期", ""),
                                "拜访员工": record.get("拜访员工", ""),
                                "沟通内容": record.get("沟通内容", ""),
                                "后续行动": record.get("后续行动", "")
                            })
                if all_records:
                    all_records.sort(key=lambda x: x.get("拜访日期", ""), reverse=True)
                    results[doctor_name] = [all_records[0]] # 只保留最新的记录
    except Exception as e:
        # 错误处理：记录错误并使用原始方法处理
        print(f"MongoDB查询医生记录失败: {str(e)}")
        # 回退到使用本地数据
        for doctor in doctor_names:
            doctor_name = doctor["name"] if isinstance(doctor, dict) else doctor
            doctor_hospital = doctor.get("hospital", "") if isinstance(doctor, dict) else ""
            
            all_records = []
            for hospital in hospital_data.get("hospitals", []):
                hospital_name = hospital.get("医院名称", "")
                
                # 如果医生有指定医院且当前医院不匹配，则跳过
                if doctor_hospital and hospital_name and not (
                    doctor_hospital.lower() in hospital_name.lower() or 
                    hospital_name.lower() in doctor_hospital.lower()
                ):
                    continue
                    
                for record in hospital.get("历史记录", []):
                    record_doctor_name = record.get("医生姓名", "")
                    # 允许部分匹配以提高灵活性
                    if record_doctor_name and (doctor_name.lower() in record_doctor_name.lower() or record_doctor_name.lower() in doctor_name.lower()):
                        all_records.append({
                            "医院": hospital.get("医院名称", ""),
                            "科室": record.get("科室", ""),
                            "拜访日期": record.get("拜访日期", ""),
                            "拜访员工": record.get("拜访员工", ""),
                            "沟通内容": record.get("沟通内容", ""),
                            "后续行动": record.get("后续行动", "")
                        })
            if all_records:
                all_records.sort(key=lambda x: x.get("拜访日期", ""), reverse=True)
                results[doctor_name] = [all_records[0]] # 只保留最新的记录
                
    return {k: v for k, v in results.items() if v}


# 修改函数签名以接受字典列表
def search_distributors_in_data(distributor_data: Dict[str, Any], distributor_info_list: List[Dict[str, str]]) -> Dict[str, List[Dict[str, str]]]:
    """在经销商数据中查找指定联系人的最新记录。"""
    results = {}
    
    # 如果使用MongoDB并且连接可用
    try:
        mongodb_db_exists = hasattr(mongodb_connector, 'db') and mongodb_connector.db is not None
        if st.session_state.use_mongodb and mongodb_db_exists:
            # 迭代包含姓名和公司信息的字典列表
            for dist_info in distributor_info_list:
                contact_name = dist_info.get("name")
                company_name = dist_info.get("company")
                if not contact_name: continue # 跳过没有联系人姓名的条目
                
                # 直接查询经销商联系人记录 - 使用正则表达式实现模糊匹配
                # 更新查询逻辑以使用联系人姓名和公司名称
                query = {
                    "联系人姓名": {"$regex": contact_name, "$options": "i"}
                }
                # 如果提供了公司名称，则将其添加到查询条件中
                if company_name:
                    query["经销商名称"] = {"$regex": company_name, "$options": "i"}
                
                # 查询匹配的经销商记录
                dist_records = list(mongodb_connector.db[MONGO_DISTRIBUTORS_COLLECTION].find(
                    query, 
                    {"_id": 0}
                ).sort("沟通日期", -1).limit(1))
                
                # 调试日志
                add_process_log(f"MongoDB查询经销商: 联系人={contact_name}, 公司={company_name or '任意'}, 找到记录数: {len(dist_records)}")
                
                # 格式化结果
                if dist_records:
                    record = dist_records[0]
                    # 使用联系人姓名作为key
                    results[contact_name] = [{
                        "经销商名称": record.get("经销商名称", ""),
                        "联系人姓名": record.get("联系人姓名", contact_name), # 确保返回记录中的姓名
                        "沟通日期": record.get("沟通日期", ""),
                        "沟通员工": record.get("沟通员工", ""),
                        "沟通内容": record.get("沟通内容", ""),
                        "后续计划": record.get("后续计划", "")
                    }]
        else:
            # 使用原来的方法处理本地JSON数据
            for dist_info in distributor_info_list:
                dist_contact_name = dist_info.get("name")
                if not dist_contact_name: continue
                
                all_contact_records = []
                for distributor in distributor_data.get("distributors", []):
                    dist_company_name = distributor.get("经销商名称", "")
                    # 检查联系人列表
                    for contact in distributor.get("联系人", []):
                         contact_name = contact.get("姓名", "")
                         if contact_name and (dist_contact_name.lower() in contact_name.lower() or contact_name.lower() in dist_contact_name.lower()):
                            # 查找与此联系人相关的沟通记录
                            records_for_this_contact = []
                            for record in distributor.get("沟通记录", []):
                                record_contact = record.get("联系人", "")
                                if record_contact and (contact_name.lower() in record_contact.lower() or record_contact.lower() in contact_name.lower()):
                                     records_for_this_contact.append({
                                        "经销商名称": dist_company_name,
                                        "联系人姓名": contact_name, # 添加联系人姓名以明确
                                        "沟通日期": record.get("沟通日期", ""),
                                        "沟通员工": record.get("沟通员工", ""),
                                        "沟通内容": record.get("沟通内容", ""),
                                        "后续计划": record.get("后续计划", "")
                                    })
                            if records_for_this_contact:
                                records_for_this_contact.sort(key=lambda x: x.get("沟通日期", ""), reverse=True)
                                # 为此联系人添加最新的记录
                                all_contact_records.append(records_for_this_contact[0])
        
                # 如果为输入的名字找到了多个联系人的记录（例如，同名），只保留绝对最新的那条
                if all_contact_records:
                    all_contact_records.sort(key=lambda x: x.get("沟通日期", ""), reverse=True)
                    results[dist_contact_name] = [all_contact_records[0]]
    except Exception as e:
        # 错误处理：记录错误并使用原始方法处理
        print(f"MongoDB查询经销商记录失败: {str(e)}")
        # 回退到使用本地数据
        for dist_info in distributor_info_list:
            dist_contact_name = dist_info.get("name")
            if not dist_contact_name: continue
            
            all_contact_records = []
            for distributor in distributor_data.get("distributors", []):
                dist_company_name = distributor.get("经销商名称", "")
                # 检查联系人列表
                for contact in distributor.get("联系人", []):
                     contact_name = contact.get("姓名", "")
                     if contact_name and (dist_contact_name.lower() in contact_name.lower() or contact_name.lower() in dist_contact_name.lower()):
                        # 查找与此联系人相关的沟通记录
                        records_for_this_contact = []
                        for record in distributor.get("沟通记录", []):
                            record_contact = record.get("联系人", "")
                            if record_contact and (contact_name.lower() in record_contact.lower() or record_contact.lower() in contact_name.lower()):
                                 records_for_this_contact.append({
                                    "经销商名称": dist_company_name,
                                    "联系人姓名": contact_name, # 添加联系人姓名以明确
                                    "沟通日期": record.get("沟通日期", ""),
                                    "沟通员工": record.get("沟通员工", ""),
                                    "沟通内容": record.get("沟通内容", ""),
                                    "后续计划": record.get("后续计划", "")
                                })
                        if records_for_this_contact:
                            records_for_this_contact.sort(key=lambda x: x.get("沟通日期", ""), reverse=True)
                            # 为此联系人添加最新的记录
                            all_contact_records.append(records_for_this_contact[0])
    
            # 如果为输入的名字找到了多个联系人的记录（例如，同名），只保留绝对最新的那条
            if all_contact_records:
                all_contact_records.sort(key=lambda x: x.get("沟通日期", ""), reverse=True)
                results[dist_contact_name] = [all_contact_records[0]]
    
    return {k: v for k, v in results.items() if v}

def format_date(date_str: str) -> str:
    """格式化日期字符串。"""
    if not date_str or date_str == "未知日期":
        return date_str
    
    # 如果已经是日期对象，直接格式化
    if isinstance(date_str, datetime.datetime):
        return date_str.strftime("%Y年%m月%d日")
        
    try:
        # 尝试解析不同格式的日期
        for fmt in ["%Y-%m-%d", "%Y/%m/%d", "%Y.%m.%d", "%Y年%m月%d日"]:
            try:
                date_obj = datetime.datetime.strptime(date_str, fmt)
                return date_obj.strftime("%Y年%m月%d日")
            except ValueError:
                continue
        
        # 如果是ISO格式带时间的
        try:
            date_obj = datetime.datetime.fromisoformat(date_str.replace('Z', '+00:00'))
            return date_obj.strftime("%Y年%m月%d日")
        except (ValueError, AttributeError):
            pass
            
        # 尝试使用dateutil解析
        try:
            from dateutil import parser
            date_obj = parser.parse(date_str)
            return date_obj.strftime("%Y年%m月%d日")
        except (ImportError, ValueError):
            pass
    except Exception as e:
        add_process_log(f"格式化日期出错: {date_str}, 错误: {e}")
        
    # 如果所有尝试都失败，则返回原始字符串
    return date_str

def process_doctor_name(doctor_name):
    """从可能的元组格式中提取医生姓名"""
    original_name = doctor_name
    
    if isinstance(doctor_name, str) and doctor_name.startswith("(") and doctor_name.endswith(")"):
        try:
            name_tuple = ast.literal_eval(doctor_name)
            if isinstance(name_tuple, tuple) and len(name_tuple) > 0:
                # 提取元组中的最后一个元素作为医生姓名
                doctor_name = str(name_tuple[-1]).strip()
                add_process_log(f"处理医生姓名: '{original_name}' -> '{doctor_name}'")
        except Exception as e:
            add_process_log(f"解析医生姓名时出错 '{doctor_name}': {e}")
    
    return doctor_name

# --- 分析核心逻辑 ---
# APIThreadPool 类保持不变
class APIThreadPool:
    """ 管理线程池以并发执行 API 调用。 """
    def __init__(self, max_workers=None):
        # 在初始化时从主线程获取配置值，避免在线程中访问st.session_state
        try:
            resolved_workers = max_workers
            if resolved_workers is None:
                resolved_workers = st.session_state.get('max_workers', 5)
            resolved_workers = int(resolved_workers)
        except (ValueError, TypeError):
            resolved_workers = 5  # 回退到默认值
        
        self.max_workers = max(1, resolved_workers)  # 确保至少有 1 个 worker
        self.thread_safe_logs = []  # 线程安全的日志存储
        
        # 打印日志但不使用add_process_log，避免在线程中访问st.session_state
        print(f"API Thread Pool initialized with {self.max_workers} workers.")
        
        # 为线程池设置线程名称前缀，帮助调试
        self.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_workers,
            thread_name_prefix="APIWorker"
        )
        self.futures = []

    def submit_task(self, func, *args, **kwargs):
        # 创建一个包装函数，该函数不会尝试访问st.session_state
        def thread_safe_wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # 捕获并记录错误，但不使用st.session_state
                print(f"Error in thread: {str(e)}")
                import traceback
                print(traceback.format_exc())
                raise e
                
        future = self.executor.submit(thread_safe_wrapper, *args, **kwargs)
        self.futures.append(future)
        return future

    def shutdown(self):
        print("Shutting down API Thread Pool.")
        self.executor.shutdown(wait=True)


def analyze_daily_report_core(daily_report_text: str, hospital_data_path: str, distributor_data_path: str,
                            use_multithreading: bool, progress_callback=None):
    """ 分析核心逻辑。[翻译日志和错误] """
    if not client:
         st.error("OpenAI 客户端不可用。分析中止。") # 翻译
         return None

    add_process_log(f"Starting analysis. Multithreading: {use_multithreading}") # 内部日志
    if progress_callback: progress_callback(0.05, "初始化分析...") # 翻译

    results_dict = {"entities": {}, "doctor_evaluations": {}, "distributor_evaluations": {}, "error": None}
    thread_pool = None
    
    # 收集主线程中的错误和日志，避免子线程直接访问Streamlit
    errors = []
    
    if use_multithreading:
        # 安全获取max_workers值，防止在线程中访问session_state
        max_workers = None
        try:
            max_workers = st.session_state.get('max_workers', 5)
        except Exception as e:
            # 如果无法访问session_state，使用默认值
            max_workers = 5
            print(f"Could not access session_state for max_workers: {e}")
        
        thread_pool = APIThreadPool(max_workers=max_workers)

    try:
        # 1. Extract Entities
        if progress_callback: progress_callback(0.1, "提取实体信息...") # 翻译
        if use_multithreading and thread_pool:
            entity_future = thread_pool.submit_task(extract_entities_with_llm, daily_report_text)
        else:
            entities = extract_entities_with_llm(daily_report_text)
            results_dict["entities"] = entities

        # 2. Load Data - 现在优先使用MongoDB查询
        add_process_log("Loading historical data...") # 内部日志
        
        # 获取MongoDB配置，在提交线程任务前
        use_mongodb = False
        db_password = None
        try:
            # 添加安全检查，确保mongodb_connector有db属性
            mongodb_db_exists = hasattr(mongodb_connector, 'db') and mongodb_connector.db is not None
            use_mongodb = st.session_state.use_mongodb and mongodb_db_exists
            
            # 如果需要MongoDB但连接不可用，获取密码以尝试连接
            if st.session_state.use_mongodb and not mongodb_db_exists:
                db_password = st.session_state.db_password
        except Exception as e:
            print(f"Could not access session_state for MongoDB config: {e}")
        
        if not use_mongodb:
            # 如果不使用MongoDB，则使用本地文件
            hospital_data = load_json_data(hospital_data_path)
            distributor_data = load_json_data(distributor_data_path)
        else:
            # 如果使用MongoDB，创建适应接口的数据结构
            add_process_log("Using MongoDB for data queries") # 内部日志
            hospital_data = {"hospitals": []}  # 创建兼容格式的数据结构
            distributor_data = {"distributors": []}  # 创建兼容格式的数据结构
            
            # 连接检查
            if not hasattr(mongodb_connector, 'db') or mongodb_connector.db is None:
                try:
                    if db_password:
                        success, error = mongodb_connector.connect(db_password)
                        if not success:
                            add_process_log(f"MongoDB connection failed: {error}") # 内部日志
                            st.error(f"MongoDB连接失败: {error}")
                            # 回退到本地文件
                            hospital_data = load_json_data(hospital_data_path)
                            distributor_data = load_json_data(distributor_data_path)
                            use_mongodb = False
                except Exception as e:
                    add_process_log(f"MongoDB connection error: {e}") # 内部日志
                    # 回退到本地文件
                    hospital_data = load_json_data(hospital_data_path)
                    distributor_data = load_json_data(distributor_data_path)
                    use_mongodb = False
        
        if progress_callback: progress_callback(0.2, "历史数据已加载。") # 翻译

        # 3. Get Entity Extraction Results (if threaded)
        if use_multithreading and 'entity_future' in locals():
            try:
                entities = entity_future.result()
                results_dict["entities"] = entities
                if progress_callback: progress_callback(0.25, "实体提取完成。") # 翻译
            except Exception as e:
                msg = f"获取实体提取结果失败: {e}" # 翻译
                add_process_log(f"Failed to get entity extraction results: {e}") # 英文日志
                errors.append(msg)
                results_dict["error"] = msg
                if thread_pool: thread_pool.shutdown()
                return results_dict

        # 检查实体是否实际提取成功
        if not results_dict.get("entities"):
            add_process_log("Entity extraction failed or returned empty.") # 英文日志
            add_process_log("实体提取失败或返回空结果。") # 中文日志
            results_dict["error"] = "实体提取失败。" # 翻译
            if thread_pool: thread_pool.shutdown()
            return results_dict # 如果实体提取失败则停止

        # 检查是否找到需要评估的目标
        doctors_to_process = results_dict["entities"].get("doctors", [])
        distributors_to_process = results_dict["entities"].get("distributors", [])
        if not doctors_to_process and not distributors_to_process:
             add_process_log("No doctors or distributors found in the report to evaluate.") # 英文日志
             add_process_log("报告中未找到需要评估的医生或经销商。") # 中文日志
             # 即使没有评估目标，仍然返回已提取的实体（如医院、产品）
             if thread_pool: thread_pool.shutdown()
             return results_dict

        # 4. Search Historical Records
        add_process_log("Searching historical records...") # 英文日志
        add_process_log(f"Found {len(doctors_to_process)} doctors and {len(distributors_to_process)} distributors to search") # 添加更多日志
        
        # 创建医生搜索列表，包含姓名和医院信息
        doctors_to_search = []
        for doc in doctors_to_process:
            doctors_to_search.append({
                "name": doc["name"],
                "hospital": doc.get("hospital", "")
            })
            add_process_log(f"Will search for doctor: {doc['name']} at hospital: {doc.get('hospital', 'any')}") # 添加更多日志
        
        # 使用修改后的函数搜索历史记录
        doctor_records = search_doctors_in_hospital_data(hospital_data, doctors_to_search)
        add_process_log(f"Found records for {len(doctor_records)} doctors") # 添加更多日志
        
        # 修改这里：传递 distributors_to_process 列表
        distributor_records = search_distributors_in_data(distributor_data, distributors_to_process)
        add_process_log(f"Found records for {len(distributor_records)} distributors") # 添加更多日志
        
        if progress_callback: progress_callback(0.3, "历史记录搜索完毕。") # 翻译

        # --- Evaluation & Suggestion ---
        evaluation_futures = {}
        suggestion_futures = {} # 存储建议任务的 future

        # 5. Submit Evaluation Tasks (Doctors)
        add_process_log(f"Submitting evaluation tasks for {len(doctors_to_process)} doctors.") # 英文日志
        for doctor in doctors_to_process:
            doc_name = doctor.get("name")
            if not doc_name: continue # 跳过没有名字的医生条目
            doc_role = doctor.get("role", "医生") # 默认角色
            records = doctor_records.get(doc_name, [])
            latest_record_content = records[0].get("沟通内容", "") if records else ""
            context = {"name": doc_name, "role": doc_role}

            if use_multithreading and thread_pool:
                future = thread_pool.submit_task(evaluate_with_llm, latest_record_content, context)
                evaluation_futures[f"doctor_{doc_name}"] = {"future": future, "records": records, "role": doc_role, "context": context}
            else:
                evaluation = evaluate_with_llm(latest_record_content, context)
                results_dict["doctor_evaluations"][doc_name] = {
                    "records": records, "evaluation": evaluation, "role": doc_role
                }

        # 收集错误并显示在主线程中
        if errors:
            for error in errors:
                st.error(error)

        # 6. Submit Evaluation Tasks (Distributors)
        add_process_log(f"Submitting evaluation tasks for {len(distributors_to_process)} distributors.") # 英文日志
        for distributor in distributors_to_process:
            dist_name = distributor.get("name")
            if not dist_name: continue
            dist_company = distributor.get("company", "未知公司") # 翻译
            records = distributor_records.get(dist_name, [])
            latest_record_content = records[0].get("沟通内容", "") if records else ""
            contact_name_in_record = records[0].get("联系人姓名", dist_name) if records else dist_name
            context = {"name": contact_name_in_record, "role": "经销商联系人", "company": dist_company} # 翻译角色

            if use_multithreading and thread_pool:
                future = thread_pool.submit_task(evaluate_with_llm, latest_record_content, context)
                evaluation_futures[f"distributor_{dist_name}"] = {"future": future, "records": records, "company": dist_company, "context": context}
            else:
                evaluation = evaluate_with_llm(latest_record_content, context)
                results_dict["distributor_evaluations"][dist_name] = {
                    "records": records, "evaluation": evaluation, "company": dist_company
                }

        if progress_callback: progress_callback(0.5, "评估任务已提交。") # 翻译

        # 7. Collect Evaluation Results & Submit Suggestion Tasks (if threaded)
        if use_multithreading and thread_pool:
            add_process_log("Collecting evaluation results and submitting suggestion tasks...") # 英文日志
            completed_evaluations = 0
            total_evaluations = len(evaluation_futures)
            if total_evaluations == 0:
                 if progress_callback: progress_callback(0.8, "无评估任务需处理。") # 翻译
            else:
                futures_to_process = [f_data["future"] for f_data in evaluation_futures.values() if f_data.get("future")]
                for future in concurrent.futures.as_completed(futures_to_process):
                     completed_evaluations += 1
                     # 安全地计算进度，避免 total_evaluations 为 0
                     progress = 0.5 + (0.3 * (completed_evaluations / total_evaluations)) if total_evaluations > 0 else 0.8
                     if progress_callback: progress_callback(progress, f"处理评估结果 ({completed_evaluations}/{total_evaluations})...") # 翻译

                     entity_key = None
                     future_data = None
                     for k, fd in evaluation_futures.items():
                         if fd.get("future") == future:
                             entity_key = k
                             future_data = fd
                             break

                     if entity_key and future_data:
                        try:
                            evaluation = future.result()
                            if evaluation.get("status") == "评估出错": # 检查内部错误
                                msg = f"LLM evaluation failed for {entity_key}: {evaluation.get('analysis')}"
                                add_process_log(msg)
                                errors.append(msg)
                                continue

                            entity_context = future_data["context"] # 使用提交任务时的 context

                            if entity_key.startswith("doctor_"):
                                doc_name = entity_key.split("doctor_", 1)[1]
                                results_dict["doctor_evaluations"][doc_name] = {
                                    "records": future_data["records"], "evaluation": evaluation, "role": future_data["role"]
                                }
                                sug_future = thread_pool.submit_task(generate_suggested_actions, evaluation, entity_context)
                                suggestion_futures[entity_key] = sug_future
                            elif entity_key.startswith("distributor_"):
                                dist_name = entity_key.split("distributor_", 1)[1]
                                results_dict["distributor_evaluations"][dist_name] = {
                                    "records": future_data["records"], "evaluation": evaluation, "company": future_data["company"]
                                }
                                sug_future = thread_pool.submit_task(generate_suggested_actions, evaluation, entity_context)
                                suggestion_futures[entity_key] = sug_future
                        except Exception as e:
                            msg = f"处理评估结果时出错 ({entity_key}): {e}" # 翻译
                            add_process_log(f"Error processing evaluation result for {entity_key}: {e}") # 英文日志
                            errors.append(msg)
                            # 存储错误状态
                            if entity_key.startswith("doctor_"):
                                doc_name = entity_key.split("doctor_", 1)[1]
                                results_dict["doctor_evaluations"][doc_name] = {"error": msg, "records": future_data.get("records",[]), "role": future_data.get("role","医生")}
                            elif entity_key.startswith("distributor_"):
                                dist_name = entity_key.split("distributor_", 1)[1]
                                results_dict["distributor_evaluations"][dist_name] = {"error": msg, "records": future_data.get("records",[]), "company": future_data.get("company","未知公司")}

                # 在主线程中显示收集到的错误
                if errors:
                    for error in errors:
                        st.error(error)

        # 8. Generate Suggestions (single-threaded) or Collect Suggestion Results (threaded)
        if not use_multithreading:
            add_process_log("Generating action suggestions (single-threaded)...") # 英文日志
            if progress_callback: progress_callback(0.8, "生成行动建议...") # 翻译
            # Doctors
            for doc_name, data in results_dict.get("doctor_evaluations", {}).items():
                if "error" not in data and "evaluation" in data:
                    context = {"name": doc_name, "role": data.get("role", "医生")}
                    suggestions = generate_suggested_actions(data["evaluation"], context)
                    results_dict["doctor_evaluations"][doc_name]["suggestions"] = suggestions
            # Distributors
            for dist_name, data in results_dict.get("distributor_evaluations", {}).items():
                 if "error" not in data and "evaluation" in data:
                    context = {"name": dist_name, "role": "经销商联系人", "company": data.get("company", "未知公司")}
                    suggestions = generate_suggested_actions(data["evaluation"], context)
                    results_dict["distributor_evaluations"][dist_name]["suggestions"] = suggestions
            if progress_callback: progress_callback(0.9, "建议已生成。") # 翻译
        else:
            # Collect suggestion results from futures
            add_process_log("Collecting action suggestion results...") # 英文日志
            completed_suggestions = 0
            total_suggestions = len(suggestion_futures)
            if total_suggestions == 0:
                if progress_callback: progress_callback(0.95, "无建议任务需处理。") # 翻译
            else:
                futures_to_process_sug = [f for f in suggestion_futures.values() if f]
                for future in concurrent.futures.as_completed(futures_to_process_sug):
                    completed_suggestions += 1
                    progress = 0.8 + (0.15 * (completed_suggestions / total_suggestions)) if total_suggestions > 0 else 0.95
                    if progress_callback: progress_callback(progress, f"处理建议结果 ({completed_suggestions}/{total_suggestions})...") # 翻译

                    entity_key = None
                    for k, f in suggestion_futures.items():
                        if f == future:
                            entity_key = k
                            break

                    if entity_key:
                        try:
                            suggestions = future.result()
                            # 检查内部错误
                            if suggestions and isinstance(suggestions, list) and suggestions[0].startswith("错误："):
                                msg = f"LLM suggestion generation failed for {entity_key}"
                                add_process_log(msg)
                                errors.append(msg)
                                # 使用备用建议
                                suggestions = ["审核评估结果", "制定后续计划", "保持沟通"]

                            if entity_key.startswith("doctor_"):
                                doc_name = entity_key.split("doctor_", 1)[1]
                                if doc_name in results_dict["doctor_evaluations"] and "error" not in results_dict["doctor_evaluations"][doc_name]:
                                    results_dict["doctor_evaluations"][doc_name]["suggestions"] = suggestions
                            elif entity_key.startswith("distributor_"):
                                dist_name = entity_key.split("distributor_", 1)[1]
                                if dist_name in results_dict["distributor_evaluations"] and "error" not in results_dict["distributor_evaluations"][dist_name]:
                                    results_dict["distributor_evaluations"][dist_name]["suggestions"] = suggestions
                        except Exception as e:
                            msg = f"处理建议结果时出错 ({entity_key}): {e}" # 翻译
                            add_process_log(f"Error processing suggestion result for {entity_key}: {e}") # 英文日志
                            errors.append(msg)
                            # 添加中文回退建议
                            fallback_suggestions = ["手动审阅评估结果。", "根据具体情况规划后续步骤。"]
                            if entity_key.startswith("doctor_"):
                                doc_name = entity_key.split("doctor_", 1)[1]
                                if doc_name in results_dict["doctor_evaluations"]: results_dict["doctor_evaluations"][doc_name]["suggestions"] = fallback_suggestions
                            elif entity_key.startswith("distributor_"):
                                dist_name = entity_key.split("distributor_", 1)[1]
                                if dist_name in results_dict["distributor_evaluations"]: results_dict["distributor_evaluations"][dist_name]["suggestions"] = fallback_suggestions

                # 显示最终收集的错误
                if errors:
                    for error in errors:
                        st.error(error)

        if progress_callback: progress_callback(0.98, "整理最终结果...") # 翻译
        add_process_log("Analysis core processing finished.") # 英文日志

    except Exception as e:
        error_msg = f"分析过程中发生意外错误: {str(e)}" # 翻译
        add_process_log(f"An unexpected error occurred during analysis: {str(e)}") # 英文日志
        st.error(error_msg)
        print(traceback.format_exc())
        results_dict["error"] = error_msg
    finally:
        if thread_pool:
            thread_pool.shutdown()

    return results_dict


def format_results_html(results_data: Dict[str, Any]) -> str:
    """ 将分析结果字典格式化为 HTML。[翻译标题和标签] """
    if not results_data or results_data.get("error"):
        # 翻译错误提示
        return f"<div style='color: red;'>分析失败或未产生结果。请检查日志。错误: {results_data.get('error', '未知错误')}</div>"

    html_output = []
    report_date = datetime.datetime.now().strftime("%Y年%m月%d日")
    html_output.append(f"<h4>分析结果 ({report_date})</h4>") # 标题

    # 状态颜色映射 (保持不变)
    status_colors = {
        "良好": "#28a745", "偏向积极": "#5cb85c", "中性": "#17a2b8",
        "偏向消极": "#ffc107", "需改善": "#fd7e14", "问题严重": "#dc3545",
        "新接触": "#6c757d", "未知": "#6c757d", "Error": "#dc3545", "评估失败": "#dc3545",
        "评估出错": "#dc3545", "格式错误": "#ffc107", "错误": "#dc3545", "解析状态失败": "#ffc107"
    }
    default_color = "#6c757d"

    # --- 医生 ---
    html_output.append("<h5>医生 / 主任</h5>") # 翻译
    doctor_evals = results_data.get("doctor_evaluations", {})
    if not doctor_evals:
        html_output.append("<p><em>未找到或处理医生信息。</em></p>") # 翻译
    else:
        for name, data in doctor_evals.items():
            html_output.append(f"<div style='margin-bottom: 15px; padding: 12px; border-left: 5px solid #007bff; background-color: #f8f9fa; border-radius: 4px; border: 1px solid #dee2e6;'>")
            role = data.get('role', '未知角色')
            html_output.append(f"<h6>{name} <span style='font-size: 0.9em; color: #6c757d;'>({role})</span></h6>") # 标签

            if "error" in data:
                 html_output.append(f"<p style='color: red;'><strong>处理时出错:</strong> {data['error']}</p>") # 翻译
            else:
                status = data.get('evaluation', {}).get('status', '未知')
                status_color = status_colors.get(status, default_color)
                html_output.append(f"<div style='margin-bottom: 10px;'><span style='display: inline-block; padding: 3px 10px; border-radius: 12px; font-size: 0.85em; font-weight: bold; color: white; background-color: {status_color};'>{status}</span></div>") # 状态值本身是中文

                if data.get("records"):
                    record = data["records"][0]
                    hospital_name = record.get('医院', '未知医院') # 翻译
                    dept_name = record.get('科室', '未知科室') # 翻译
                    visit_date = format_date(record.get('拜访日期', ''))
                    visit_staff = record.get('拜访员工', '未知人员') # 翻译
                    comm_content = record.get('沟通内容', '无') # 翻译
                    html_output.append(f"<p style='margin-bottom: 5px;'><strong>所属医院:</strong> {hospital_name}</p>") # 翻译
                    html_output.append(f"<p style='margin-bottom: 5px;'><strong>所属科室:</strong> {dept_name}</p>") # 翻译
                    html_output.append(f"<p style='margin-bottom: 5px;'><strong>最近拜访:</strong> {visit_date} (由 {visit_staff})</p>") # 翻译
                    # 使用 st.expander 可能更好，但保持 HTML 格式
                    html_output.append(f"<details><summary style='font-size: 0.9em; cursor: pointer;'>沟通纪要</summary><p style='font-size: 0.9em; margin-top: 5px; padding: 8px; background-color: #e9ecef; border-radius: 3px;'>{comm_content}</p></details>") # 翻译
                else:
                    html_output.append("<p><em>未找到历史沟通记录。</em></p>") # 翻译

                analysis = data.get('evaluation', {}).get('analysis', '无') # 翻译 (内容本身是中文)
                if analysis == "解析分析失败": analysis = "关系分析解析失败" # 更友好的提示
                html_output.append(f"<p style='margin-bottom: 5px; margin-top: 10px;'><strong>关系分析:</strong> {analysis}</p>") # 翻译

                suggestions = data.get("suggestions", ["未生成建议。"]) # 翻译
                html_output.append("<p style='margin-bottom: 5px;'><strong>建议行动:</strong></p>") # 翻译
                html_output.append("<ul style='font-size: 0.9em; margin-left: -15px;'>")
                for suggestion in suggestions:
                    html_output.append(f"<li style='margin-bottom: 3px;'>{suggestion}</li>") # 建议本身是中文
                html_output.append("</ul>")
            html_output.append("</div>")

    # --- 经销商 ---
    html_output.append("<h5>经销商 / 联系人</h5>") # 翻译
    distributor_evals = results_data.get("distributor_evaluations", {})
    if not distributor_evals:
         html_output.append("<p><em>未找到或处理经销商信息。</em></p>") # 翻译
    else:
        for name, data in distributor_evals.items():
             html_output.append(f"<div style='margin-bottom: 15px; padding: 12px; border-left: 5px solid #28a745; background-color: #f8f9fa; border-radius: 4px; border: 1px solid #dee2e6;'>")
             company = data.get('company', '未知公司') # 翻译
             contact_name = name
             if data.get("records"): contact_name = data["records"][0].get("联系人姓名", name)
             html_output.append(f"<h6>{contact_name} <span style='font-size: 0.9em; color: #6c757d;'>({company})</span></h6>") # 标签

             if "error" in data:
                  html_output.append(f"<p style='color: red;'><strong>处理时出错:</strong> {data['error']}</p>") # 翻译
             else:
                status = data.get('evaluation', {}).get('status', '未知')
                status_color = status_colors.get(status, default_color)
                html_output.append(f"<div style='margin-bottom: 10px;'><span style='display: inline-block; padding: 3px 10px; border-radius: 12px; font-size: 0.85em; font-weight: bold; color: white; background-color: {status_color};'>{status}</span></div>") # 状态值本身是中文

                if data.get("records"):
                    record = data["records"][0]
                    comm_date = format_date(record.get('沟通日期', ''))
                    comm_staff = record.get('沟通员工', '未知人员') # 翻译
                    comm_content = record.get('沟通内容', '无') # 翻译
                    html_output.append(f"<p style='margin-bottom: 5px;'><strong>最近沟通:</strong> {comm_date} (由 {comm_staff})</p>") # 翻译
                    html_output.append(f"<details><summary style='font-size: 0.9em; cursor: pointer;'>沟通纪要</summary><p style='font-size: 0.9em; margin-top: 5px; padding: 8px; background-color: #e9ecef; border-radius: 3px;'>{comm_content}</p></details>") # 翻译
                else:
                    html_output.append("<p><em>未找到历史沟通记录。</em></p>") # 翻译

                analysis = data.get('evaluation', {}).get('analysis', '无') # 翻译 (内容本身是中文)
                if analysis == "解析分析失败": analysis = "关系分析解析失败" # 更友好的提示
                html_output.append(f"<p style='margin-bottom: 5px; margin-top: 10px;'><strong>关系分析:</strong> {analysis}</p>") # 翻译

                suggestions = data.get("suggestions", ["未生成建议。"]) # 翻译
                html_output.append("<p style='margin-bottom: 5px;'><strong>建议行动:</strong></p>") # 翻译
                html_output.append("<ul style='font-size: 0.9em; margin-left: -15px;'>")
                for suggestion in suggestions:
                    html_output.append(f"<li style='margin-bottom: 3px;'>{suggestion}</li>") # 建议本身是中文
                html_output.append("</ul>")
             html_output.append("</div>")

    # 返回 HTML 字符串 (摘要信息由 Streamlit 列处理)
    final_html = "".join(html_output)
    return final_html


def format_results_text(results_data: Dict[str, Any]) -> str:
    """ 将分析结果字典格式化为纯文本。[翻译标签和提示] """
    if not results_data or results_data.get("error"):
        return f"分析失败或未产生结果。请检查日志。\n错误: {results_data.get('error', '未知错误')}" # 翻译

    output = []
    report_date = datetime.datetime.now().strftime("%Y-%m-%d")
    output.append(f"=== 日报分析结果 ({report_date}) ===") # 翻译

    # Doctors
    output.append("\n--- 医生 / 主任 ---") # 翻译
    doctor_evals = results_data.get("doctor_evaluations", {})
    if not doctor_evals: output.append("未找到医生信息。") # 翻译
    else:
        for name, data in doctor_evals.items():
            role = data.get('role', '未知角色') # 翻译
            output.append(f"\n姓名: {name} ({role})") # 翻译
            if "error" in data:
                output.append(f"  错误: {data['error']}") # 翻译
                continue
            status = data.get('evaluation', {}).get('status', '未知')
            output.append(f"  状态: {status}") # 翻译 (状态值本身是中文)
            if data.get("records"):
                record = data["records"][0]
                output.append(f"  医院: {record.get('医院', '无')}") # 翻译
                output.append(f"  科室: {record.get('科室', '无')}") # 翻译
                output.append(f"  最近拜访: {format_date(record.get('拜访日期', ''))} (由 {record.get('拜访员工', '未知')})") # 翻译
                output.append(f"  沟通纪要: {record.get('沟通内容', '无')}") # 翻译
            else:
                output.append("  历史记录: 未找到历史沟通记录。") # 翻译
            analysis = data.get('evaluation', {}).get('analysis', '无')
            if analysis == "解析分析失败": analysis = "关系分析解析失败"
            output.append(f"  关系分析: {analysis}") # 翻译
            output.append("  建议行动:") # 翻译
            suggestions = data.get("suggestions", ["无"]) # 翻译
            for i, sug in enumerate(suggestions, 1):
                output.append(f"    {i}. {sug}") # 建议本身是中文

    # Distributors
    output.append("\n--- 经销商 / 联系人 ---") # 翻译
    distributor_evals = results_data.get("distributor_evaluations", {})
    if not distributor_evals: output.append("未找到经销商信息。") # 翻译
    else:
        for name, data in distributor_evals.items():
            company = data.get('company', '未知公司') # 翻译
            contact_name = name
            if data.get("records"): contact_name = data["records"][0].get("联系人姓名", name)
            output.append(f"\n姓名: {contact_name} ({company})") # 翻译
            if "error" in data:
                output.append(f"  错误: {data['error']}") # 翻译
                continue
            status = data.get('evaluation', {}).get('status', '未知')
            output.append(f"  状态: {status}") # 翻译
            if data.get("records"):
                record = data["records"][0]
                output.append(f"  最近沟通: {format_date(record.get('沟通日期', ''))} (由 {record.get('沟通员工', '未知')})") # 翻译
                output.append(f"  沟通纪要: {record.get('沟通内容', '无')}") # 翻译
            else:
                 output.append("  历史记录: 未找到历史沟通记录。") # 翻译
            analysis = data.get('evaluation', {}).get('analysis', '无')
            if analysis == "解析分析失败": analysis = "关系分析解析失败"
            output.append(f"  关系分析: {analysis}") # 翻译
            output.append("  建议行动:") # 翻译
            suggestions = data.get("suggestions", ["无"]) # 翻译
            for i, sug in enumerate(suggestions, 1):
                output.append(f"    {i}. {sug}") # 建议本身是中文

    # Summary
    output.append("\n--- 概要信息 ---") # 翻译
    entities = results_data.get("entities", {})
    output.append("医院: " + ", ".join(entities.get("hospitals", ["无"]))) # 翻译
    output.append("科室: " + ", ".join(entities.get("departments", ["无"]))) # 翻译
    output.append("产品: " + ", ".join(entities.get("products", ["无"]))) # 翻译

    return "\n".join(output)

# --- 搜索函数 ---
def search_all_entities(search_term: str) -> Dict[str, Any]:
    """在医院和经销商数据中搜索关键词。"""
    results = {"doctors": [], "hospitals": [], "distributors": []}
    if not search_term: return results
    term_lower = search_term.lower()
    
    try:
        # 检查是否使用MongoDB以及连接是否有效
        mongodb_db_exists = hasattr(mongodb_connector, 'db') and mongodb_connector.db is not None
        if st.session_state.use_mongodb and mongodb_db_exists:
            # 构建搜索模式
            search_pattern = {"$regex": term_lower, "$options": "i"}
            
            # 搜索医院
            hospital_query = {"$or": [
                {"医院名称": search_pattern},
                {"标准医院名称": search_pattern}
            ]}
            
            # 1. 先获取匹配的医院基本信息
            hospitals_data = mongodb_connector.get_hospital_data()
            matching_hospitals = []
            
            for hospital in hospitals_data.get("hospitals", []):
                h_name = hospital.get("医院名称", "")
                h_std_name = hospital.get("标准医院名称", "")
                h_id = hospital.get("hospitalId", "")
                
                if (term_lower in h_name.lower() or 
                    (h_std_name and term_lower in h_std_name.lower())):
                    # 2. 对每个匹配的医院，获取其拜访记录
                    if h_id:
                        visits = mongodb_connector.get_visits_for_hospital(h_id)
                        # 提取此医院的医生信息
                        doctors_info = []
                        for record in visits[:10]:  # 限制为前10个记录
                            doc_name = record.get("医生姓名", "")
                            visit_date = record.get("拜访日期", "未知日期")
                            employee = record.get("拜访员工", "未知人员")
                            doctors_info.append({
                                "姓名": doc_name,
                                "拜访日期": visit_date,
                                "拜访员工": employee,
                                "沟通内容": record.get("沟通内容", ""),
                                "后续行动": record.get("后续行动", "")
                            })
                        
                        # 将此医院添加到结果中
                        matching_hospitals.append({
                            "名称": h_name,
                            "科室数量": 0,  # 在新结构中无法轻松获取科室数
                            "历史记录数量": len(visits),
                            "医生信息": doctors_info
                        })
            
            # 将匹配的医院添加到结果中
            results["hospitals"].extend(matching_hospitals[:10])
            
            # 搜索医生
            doctor_query = {
                "$or": [
                    # 直接匹配医生姓名字段
                    {"医生姓名": search_pattern},
                    # 匹配元组格式中的最后一个元素（实际医生姓名）
                    {"医生姓名": {"$regex": f".*'{term_lower}'.*", "$options": "i"}}
                ]
            }
            doctor_records = list(mongodb_connector.db[MONGO_HOSPITALS_COLLECTION].find(
                doctor_query, 
                {
                    "_id": 0, 
                    "医院名称": 1, 
                    "标准医院名称": 1,
                    "医生姓名": 1, 
                    "科室": 1, 
                    "拜访日期": 1, 
                    "拜访员工": 1,
                    "沟通内容": 1,
                    "后续行动": 1
                }
            ).sort("拜访日期", -1).limit(10))
            
            for doc in doctor_records:
                hospital_name = doc.get("标准医院名称") or doc.get("医院名称", "未知医院")
                # 使用process_doctor_name处理医生姓名
                processed_name = process_doctor_name(doc.get("医生姓名", ""))
                results["doctors"].append({
                    "姓名": processed_name,
                    "医院": hospital_name,
                    "科室": doc.get("科室", "未知科室"),
                    "最近拜访": doc.get("拜访日期", "未知日期"),
                    "拜访员工": doc.get("拜访员工", "未知人员"),
                    "沟通内容": doc.get("沟通内容", ""),
                    "后续行动": doc.get("后续行动", "")
                })
            
            # 搜索经销商公司名
            distributor_query = {"$or": [
                {"经销商名称": search_pattern},
                {"distributorId": search_pattern},
                {"name": search_pattern}
            ]}
            
            for distributor in mongodb_connector.db[MONGO_DISTRIBUTORS_COLLECTION].find(distributor_query, {"_id": 0}):
                comp_name = distributor.get("经销商名称") or distributor.get("name", "")
                contact_count = len(distributor.get("联系人", []))
                comm_records = distributor.get("沟通记录", [])
                
                results["distributors"].append({
                    "公司/联系人": comp_name,
                    "类型": "公司",
                    "详细信息": f"联系人数: {contact_count}, 沟通记录数: {len(comm_records)}",
                    "沟通记录": comm_records
                })
            
            # 搜索经销商联系人
            contact_query = {"联系人.姓名": search_pattern}
            for dist in mongodb_connector.db[MONGO_DISTRIBUTORS_COLLECTION].find(contact_query, {"_id": 0}):
                comp_name = dist.get("经销商名称") or dist.get("name", "")
                
                # 找出匹配的联系人
                for contact in dist.get("联系人", []):
                    if isinstance(contact, dict):
                        cont_name = contact.get("姓名", "")
                        if term_lower in cont_name.lower():
                            results["distributors"].append({
                                "公司/联系人": cont_name,
                                "类型": "联系人",
                                "详细信息": f"所属公司: {comp_name}, 职位: {contact.get('职位', '未知')}, 电话: {contact.get('电话', '未知')}",
                                "沟通记录": dist.get("沟通记录", [])
                            })
                    elif isinstance(contact, str):
                        if term_lower in contact.lower():
                            results["distributors"].append({
                                "公司/联系人": contact,
                                "类型": "联系人",
                                "详细信息": f"所属公司: {comp_name}, 职位: 未知, 电话: 未知",
                                "沟通记录": dist.get("沟通记录", [])
                            })
        else:
            # 使用本地文件
            hospital_data = load_json_data(st.session_state.hospital_data_path)
            distributor_data = load_json_data(st.session_state.distributor_data_path)

            # 搜索医院
            for hospital in hospital_data.get("hospitals", []):
                name = hospital.get("医院名称", "")
                if term_lower in name.lower():
                    # 获取该医院的医生信息
                    doctors_info = []
                    for record in hospital.get("历史记录", []):
                        doc_name = record.get("医生姓名", "")
                        visit_date = record.get("拜访日期", "未知日期")
                        employee = record.get("拜访员工", "未知人员")
                        doctors_info.append({
                            "姓名": doc_name,
                            "拜访日期": visit_date,
                            "拜访员工": employee
                        })
                    
                    results["hospitals"].append({
                        "名称": name,
                        "科室数量": len(hospital.get("科室", [])),
                        "历史记录数量": len(hospital.get("历史记录", [])),
                        "医生信息": doctors_info
                    })

            # 搜索医生 (在医院历史记录中)
            for hospital in hospital_data.get("hospitals", []):
                h_name = hospital.get("医院名称", "未知医院")
                for record in hospital.get("历史记录", []):
                    doc_name = record.get("医生姓名", "")
                    if term_lower in doc_name.lower():
                        results["doctors"].append({
                            "姓名": doc_name,
                            "医院": h_name,
                            "科室": record.get("科室", "未知科室"),
                            "最近拜访": record.get("拜访日期", "未知日期"),
                            "拜访员工": record.get("拜访员工", "未知人员")
                        })

            # 搜索经销商 (公司名和联系人名)
            for distributor in distributor_data.get("distributors", []):
                comp_name = distributor.get("经销商名称", "")
                # 搜索公司名
                if term_lower in comp_name.lower():
                    results["distributors"].append({
                        "公司/联系人": comp_name,
                        "类型": "公司",
                        "详细信息": f"联系人数: {len(distributor.get('联系人', []))}, 沟通记录数: {len(distributor.get('沟通记录', []))}"
                    })

                # 搜索联系人名
                for contact in distributor.get("联系人", []):
                    # 检查联系人是否为字符串类型
                    if isinstance(contact, str):
                        # 如果是字符串，直接用字符串本身作为联系人姓名
                        cont_name = contact
                        if term_lower in cont_name.lower():
                            results["distributors"].append({
                                "公司/联系人": cont_name,
                                "类型": "联系人",
                                "详细信息": f"所属公司: {comp_name}, 职位: 未知, 电话: 未知"
                            })
                    else:
                        # 如果是字典，按原方式处理
                        cont_name = contact.get("姓名", "")
                        if term_lower in cont_name.lower():
                            results["distributors"].append({
                                "公司/联系人": cont_name,
                                "类型": "联系人",
                                "详细信息": f"所属公司: {comp_name}, 职位: {contact.get('职位', '未知')}, 电话: {contact.get('电话', '未知')}"
                            })
    except Exception as e:
        st.error(f"搜索过程中出错: {e}")
        print(traceback.format_exc())
        
        # 发生错误时尝试使用本地文件作为回退
        try:
            hospital_data = load_local_json_data(st.session_state.hospital_data_path)
            distributor_data = load_local_json_data(st.session_state.distributor_data_path)
            
            # 搜索医院 (使用简化的搜索逻辑减少出错可能)
            for hospital in hospital_data.get("hospitals", []):
                name = hospital.get("医院名称", "")
                if term_lower in name.lower():
                    results["hospitals"].append({
                        "名称": name,
                        "科室数量": len(hospital.get("科室", [])),
                        "历史记录数量": len(hospital.get("历史记录", [])),
                        "医生信息": []
                    })
                    
            # 简化搜索医生
            for hospital in hospital_data.get("hospitals", []):
                h_name = hospital.get("医院名称", "")
                for record in hospital.get("历史记录", []):
                    doc_name = record.get("医生姓名", "")
                    if doc_name and term_lower in doc_name.lower():
                        results["doctors"].append({
                            "姓名": doc_name,
                            "医院": h_name,
                            "科室": record.get("科室", ""),
                            "最近拜访": record.get("拜访日期", ""),
                            "拜访员工": record.get("拜访员工", "")
                        })
        except Exception as e2:
            st.error(f"回退到本地文件搜索时也出错: {e2}")

    return results


def format_search_results_html(results: Dict[str, Any]) -> str:
    """ 将搜索结果格式化为 HTML。"""
    if not results or (not results["doctors"] and not results["hospitals"] and not results["distributors"]):
        return "<p>未找到匹配的记录。</p>"

    # 生成一个随机ID作为HTML元素的前缀，避免同页面多次搜索的ID冲突
    uid = f"search_{uuid.uuid4().hex[:8]}"
    
    # 添加CSS样式
    html_output = [f"""
    <style>
    .clickable-row {{
        cursor: pointer;
        transition: background-color 0.2s;
    }}
    .clickable-row:hover {{
        background-color: #f5f5f5;
    }}
    .communication-content {{
        display: none;
        padding: 10px;
        margin-top: 2px;
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 4px;
    }}
    </style>
    <script>
    function toggleContent(id) {{
        var content = document.getElementById(id);
        if (content.style.display === 'block') {{
            content.style.display = 'none';
        }} else {{
            content.style.display = 'block';
        }}
    }}
    </script>
    <h4>搜索结果</h4>
    """]

    # Doctors
    if results["doctors"]:
        html_output.append("<h5>医生</h5>")
        html_output.append("<table style='width:100%; border-collapse: collapse; font-size: 0.9em;'><thead><tr style='background-color: #e9ecef;'>")
        headers = ["姓名", "医院", "科室", "最近拜访", "拜访员工"]
        for h in headers: html_output.append(f"<th style='border: 1px solid #dee2e6; padding: 6px; text-align: left;'>{h}</th>")
        html_output.append("</tr></thead><tbody>")
        
        for i, doc in enumerate(results["doctors"]):
            row_id = f"{uid}_doc_{i}"
            content_id = f"{uid}_doc_content_{i}"
            
            html_output.append(f"<tr class='clickable-row' onclick=\"toggleContent('{content_id}')\">")
            html_output.append(f"<td style='border: 1px solid #dee2e6; padding: 6px;'>{doc.get('姓名','')}</td>")
            html_output.append(f"<td style='border: 1px solid #dee2e6; padding: 6px;'>{doc.get('医院','')}</td>")
            html_output.append(f"<td style='border: 1px solid #dee2e6; padding: 6px;'>{doc.get('科室','')}</td>")
            html_output.append(f"<td style='border: 1px solid #dee2e6; padding: 6px;'>{format_date(doc.get('最近拜访',''))}</td>")
            html_output.append(f"<td style='border: 1px solid #dee2e6; padding: 6px;'>{doc.get('拜访员工','')}</td>")
            html_output.append("</tr>")
            
            # 添加折叠内容行
            comm_content = doc.get('沟通内容', '无记录').replace('\n', '<br>')
            action_content = doc.get('后续行动', '无记录').replace('\n', '<br>')
            html_output.append(f"<tr><td colspan='5' style='padding: 0;'><div id='{content_id}' class='communication-content'>")
            html_output.append(f"<p><strong>沟通内容：</strong><br>{comm_content}</p>")
            html_output.append(f"<p><strong>后续行动：</strong><br>{action_content}</p>")
            html_output.append("</div></td></tr>")
            
        html_output.append("</tbody></table>")

     # Hospitals
    if results["hospitals"]:
        html_output.append("<h5 style='margin-top: 15px;'>医院</h5>")
        html_output.append("<table style='width:100%; border-collapse: collapse; font-size: 0.9em;'><thead><tr style='background-color: #e9ecef;'>")
        headers = ["名称", "科室数量", "历史记录数", "医生信息"]
        for h in headers: html_output.append(f"<th style='border: 1px solid #dee2e6; padding: 6px; text-align: left;'>{h}</th>")
        html_output.append("</tr></thead><tbody>")
        
        for h_idx, h in enumerate(results["hospitals"]):
            html_output.append("<tr>")
            html_output.append(f"<td style='border: 1px solid #dee2e6; padding: 6px;'>{h.get('名称','')}</td>")
            html_output.append(f"<td style='border: 1px solid #dee2e6; padding: 6px;'>{h.get('科室数量','')}</td>")
            html_output.append(f"<td style='border: 1px solid #dee2e6; padding: 6px;'>{h.get('历史记录数量','')}</td>")
            
            # 添加医生信息列，使每个医生记录可点击
            doctors_info = h.get('医生信息', [])
            if doctors_info:
                doctor_html = "<ul style='margin: 0; padding-left: 15px;'>"
                for d_idx, doc in enumerate(doctors_info):
                    doc_content_id = f"{uid}_hosp_{h_idx}_doc_{d_idx}"
                    doctor_html += f"<li class='clickable-row' onclick=\"toggleContent('{doc_content_id}')\">"
                    doctor_html += f"{doc.get('姓名', '')}: {doc.get('拜访日期', '')} 由 {doc.get('拜访员工', '')} 拜访"
                    doctor_html += "</li>"
                    
                    # 添加折叠内容
                    comm_content = doc.get('沟通内容', '无记录').replace('\n', '<br>')
                    action_content = doc.get('后续行动', '无记录').replace('\n', '<br>')
                    doctor_html += f"<div id='{doc_content_id}' class='communication-content'>"
                    doctor_html += f"<p><strong>沟通内容：</strong><br>{comm_content}</p>"
                    doctor_html += f"<p><strong>后续行动：</strong><br>{action_content}</p>"
                    doctor_html += "</div>"
                doctor_html += "</ul>"
                html_output.append(f"<td style='border: 1px solid #dee2e6; padding: 6px;'>{doctor_html}</td>")
            else:
                html_output.append("<td style='border: 1px solid #dee2e6; padding: 6px;'>无医生记录</td>")
            
            html_output.append("</tr>")
        html_output.append("</tbody></table>")

    # Distributors
    if results["distributors"]:
        html_output.append("<h5 style='margin-top: 15px;'>经销商 / 联系人</h5>")
        html_output.append("<table style='width:100%; border-collapse: collapse; font-size: 0.9em;'><thead><tr style='background-color: #e9ecef;'>")
        headers = ["公司 / 联系人", "类型", "详细信息"]
        for h in headers: html_output.append(f"<th style='border: 1px solid #dee2e6; padding: 6px; text-align: left;'>{h}</th>")
        html_output.append("</tr></thead><tbody>")
        
        for d_idx, d in enumerate(results["distributors"]):
            row_id = f"{uid}_dist_{d_idx}"
            content_id = f"{uid}_dist_content_{d_idx}"
            
            html_output.append(f"<tr class='clickable-row' onclick=\"toggleContent('{content_id}')\">")
            html_output.append(f"<td style='border: 1px solid #dee2e6; padding: 6px;'>{d.get('公司/联系人','')}</td>")
            html_output.append(f"<td style='border: 1px solid #dee2e6; padding: 6px;'>{d.get('类型','')}</td>")
            # 详细信息中的标签翻译
            details_text = d.get('详细信息','')
            details_text = details_text.replace("联系人数:", "联系人数:")
            details_text = details_text.replace("沟通记录数:", "沟通记录数:")
            details_text = details_text.replace("所属公司:", "所属公司:")
            details_text = details_text.replace("职位:", "职位:")
            details_text = details_text.replace("电话:", "电话:")
            html_output.append(f"<td style='border: 1px solid #dee2e6; padding: 6px;'>{details_text}</td>")
            html_output.append("</tr>")
            
            # 添加折叠内容行
            comm_content = d.get('沟通内容', '无记录').replace('\n', '<br>')
            html_output.append(f"<tr><td colspan='3' style='padding: 0;'><div id='{content_id}' class='communication-content'>")
            html_output.append(f"<p><strong>沟通内容：</strong><br>{comm_content}</p>")
            if d.get('沟通记录'):
                html_output.append("<p><strong>历史沟通记录：</strong></p><ul>")
                for record in d.get('沟通记录', []):
                    record_date = record.get('沟通日期', '未知日期')
                    record_employee = record.get('沟通员工', '未知人员')
                    record_content = record.get('沟通内容', '无记录').replace('\n', '<br>')
                    html_output.append(f"<li><strong>{record_date} - {record_employee}：</strong><br>{record_content}</li>")
                html_output.append("</ul>")
            html_output.append("</div></td></tr>")
            
        html_output.append("</tbody></table>")

    # 添加提示信息
    if results["doctors"] or results["hospitals"] or results["distributors"]:
        html_output.append("<p style='margin-top: 10px; font-style: italic; color: #6c757d;'>点击行可查看详细沟通内容</p>")
        
        if len(results["doctors"]) == 10 or any(len(h.get('医生信息', [])) == 10 for h in results["hospitals"]):
            html_output.append("<p style='font-style: italic; color: #6c757d;'>（仅显示最新的10条记录）</p>")

    return "".join(html_output)


# --- 示例数据 (保持不变) ---
sample_report = '''曹硕 2024/04/02 19:30:03企业微信用户
1、上午前往七院，通过陆总推送经销商老板邱春，了解到七院代表是王彦，添加联系方式后进行沟通，了解到院内月腔镜手术量在7～8台左右，我们的产品占能占6台。之前有跟进过胸外科主任叶亮和肝胆外科主任杨珏，叶亮手术量太少，三个月才一台腔镜手术，肝胆外科主任杨珏的手术也少，所以之前没有跟进。回复还是建议都做一下，拓宽客户面，本来七院的体量就不大，能多一份是一份。
2、拜访一病区医生办公室，偶遇血管外科张遂亮老师，表示厂家身份，老师回复是血管外科的，用不上，随即询问得知一病区包括了甲乳外科和血管外科，当时办公室里其他的老师都是规培生，得知普外科大主任赵滨的办公室在最东边，感谢后离开。
...
1、调研仁济医院人事、竞品、科室情况。
2、参加公司培训活动。'''
sample_hospital_data = { # 内容不变
    "hospitals": [
        {
            "医院名称": "七院", "科室": ["普外科", "血管外科", "胸外科", "肝胆外科", "甲乳外科"],
            "历史记录": [
                {"医生姓名": "叶亮", "科室": "胸外科", "拜访日期": "2024-02-15", "拜访员工": "李明", "沟通内容": "向叶主任介绍了我们最新的腔镜设备，他表示对价格比较敏感，手术量太少，三个月才一台腔镜手术，可能不太会考虑采购。", "后续行动": "定期回访，关注手术量变化"},
                {"医生姓名": "杨珏", "科室": "肝胆外科", "拜访日期": "2024-03-10", "拜访员工": "张华", "沟通内容": "杨主任提出现有腔镜设备使用中存在的一些问题，希望了解我们产品是否能解决这些问题。但由于手术量较少，暂时没有更换计划。", "后续行动": "提供技术方案，持续跟进"}
            ]
        },
        {
            "医院名称": "仁济医院", "科室": ["普外科", "消化内科", "骨科", "妇产科"],
            "历史记录": [
                {"医生姓名": "赵滨", "科室": "普外科", "拜访日期": "2024-03-25", "拜访员工": "王刚", "沟通内容": "与赵主任深入交流了我们最新一代腔镜设备的优势，他表示有兴趣了解更多技术细节和临床案例。医院计划在下半年更新普外科的腔镜设备。", "后续行动": "安排产品演示，提供临床案例资料"}
            ]
        }
    ]
}
sample_distributor_data = { # 内容不变
    "distributors": [
        {
            "经销商名称": "健康医疗器械有限公司",
            "联系人": [{"姓名": "陆总", "职位": "销售总监"}, {"姓名": "王彦", "职位": "区域经理"}],
            "沟通记录": [
                {"联系人": "陆总", "沟通日期": "2024-03-15", "沟通员工": "张华", "沟通内容": "与陆总讨论了七院的市场情况，他介绍说七院的代表是王彦，腔镜手术量不高但稳定，建议我们与王彦直接联系。陆总表示愿意牵线搭桥，帮助推进产品在七院的销售。", "后续计划": "联系王彦，安排拜访七院"},
                {"联系人": "王彦", "沟通日期": "2024-03-20", "沟通员工": "曹硕", "沟通内容": "通过电话与王彦初步沟通，了解到七院月腔镜手术量在7-8台左右，我们的产品占6台。王彦对我们产品的使用反馈相对积极，但也提到医院采购预算有限。", "后续计划": "安排面对面会议，讨论合作细节"}
            ]
        },
        {"经销商名称": "邱氏医疗设备公司", "联系人": [{"姓名": "邱春", "职位": "公司老板"}], "沟通记录": []}
    ]
}

def load_sample_data_action():
    """加载示例数据"""
    try:
        sample_data_path = os.path.join("sample_data")
        hospital_sample = os.path.join(sample_data_path, "hospital_data.json")
        distributor_sample = os.path.join(sample_data_path, "distributor_data.json")
        
        if os.path.exists(hospital_sample) and os.path.exists(distributor_sample):
            # 复制到本地文件（作为备份）
            shutil.copy(hospital_sample, st.session_state.hospital_data_path)
            shutil.copy(distributor_sample, st.session_state.distributor_data_path)
            
            # 如果使用MongoDB，也保存到MongoDB
            if st.session_state.use_mongodb:
                if not hasattr(mongodb_connector, 'db') or mongodb_connector.db is None:
                    if st.session_state.db_password:
                        success, error = mongodb_connector.connect(st.session_state.db_password)
                        if not success:
                            return f"连接MongoDB失败，示例数据仅保存在本地文件: {error}"
                    else:
                        return "MongoDB密码未配置，示例数据仅保存在本地文件"
                
                # 加载并保存示例数据到MongoDB
                with open(hospital_sample, 'r', encoding='utf-8') as f:
                    hospital_data = json.load(f)
                with open(distributor_sample, 'r', encoding='utf-8') as f:
                    distributor_data = json.load(f)
                
                mongodb_success = True
                if not mongodb_connector.save_hospital_data(hospital_data):
                    mongodb_success = False
                if not mongodb_connector.save_distributor_data(distributor_data):
                    mongodb_success = False
                
                if mongodb_success:
                    return "示例数据加载成功！（已保存到本地文件和MongoDB）"
                else:
                    return "示例数据加载到本地文件成功，但保存到MongoDB失败"
            
            return "示例数据加载成功！"
        else:
            return "示例数据文件不存在！"
    except Exception as e:
        print(traceback.format_exc())
        return f"加载示例数据时出错: {str(e)}"

def view_hospital_data():
    try:
        data = load_json_data(st.session_state.hospital_data_path)
        
        # 处理MongoDB数据
        if st.session_state.use_mongodb and "hospitals" in data:
            # 移除_id字段并转换日期时间对象
            for hospital in data["hospitals"]:
                if "_id" in hospital:
                    del hospital["_id"]
                # 处理历史记录中的日期
                if "历史记录" in hospital:
                    for record in hospital["历史记录"]:
                        # 转换datetime对象为字符串
                        if "拜访日期" in record and isinstance(record["拜访日期"], (datetime.datetime, str)):
                            if isinstance(record["拜访日期"], datetime.datetime):
                                record["拜访日期"] = record["拜访日期"].strftime("%Y-%m-%d")
        
        # 限制只显示前10家医院
        if "hospitals" in data and len(data["hospitals"]) > 10:
            limited_data = {"hospitals": data["hospitals"][:10]}
            pretty_json = json.dumps(limited_data, ensure_ascii=False, indent=2, default=str)
            return pretty_json + f"\n\n... (仅显示前10条医院记录，共{len(data['hospitals'])}条)"
        else:
            pretty_json = json.dumps(data, ensure_ascii=False, indent=2, default=str)
            return pretty_json
    except Exception as e:
        print(traceback.format_exc())
        return f"查看医院数据时出错: {str(e)}"

def view_distributor_data():
    """查看经销商数据"""
    try:
        data = load_json_data(st.session_state.distributor_data_path)
        
        # 处理MongoDB数据
        if st.session_state.use_mongodb and "distributors" in data:
            # 移除_id字段并转换日期时间对象
            for distributor in data["distributors"]:
                if "_id" in distributor:
                    del distributor["_id"]
                # 处理沟通记录中的日期
                if "沟通记录" in distributor:
                    for record in distributor["沟通记录"]:
                        # 转换datetime对象为字符串
                        if "沟通日期" in record and isinstance(record["沟通日期"], (datetime.datetime, str)):
                            if isinstance(record["沟通日期"], datetime.datetime):
                                record["沟通日期"] = record["沟通日期"].strftime("%Y-%m-%d")
        
        # 限制只显示前10个经销商
        if "distributors" in data and len(data["distributors"]) > 10:
            limited_data = {"distributors": data["distributors"][:10]}
            pretty_json = json.dumps(limited_data, ensure_ascii=False, indent=2, default=str)
            return pretty_json + "\n\n... (仅显示前10条记录，共" + str(len(data["distributors"])) + "条)"
        else:
            pretty_json = json.dumps(data, ensure_ascii=False, indent=2, default=str)
            return pretty_json
    except Exception as e:
        print(traceback.format_exc())
        return f"查看经销商数据时出错: {str(e)}"

def upload_hospital_data(file):
    """上传医院数据文件"""
    try:
        if file is None:
            return "请选择要上传的文件"
        
        # 保存文件到本地（即使使用MongoDB也先保存本地备份）
        save_path = os.path.join("data", file.name)
        with open(save_path, "wb") as f:
            f.write(file.getbuffer())
        
        # 验证JSON格式
        data = load_local_json_data(save_path)
        if "hospitals" not in data:
            return "文件格式无效：上传的医院数据文件缺少 'hospitals' 键"
        
        # 设置为当前文件路径
        st.session_state.hospital_data_path = save_path
        
        # 如果使用MongoDB，则保存到MongoDB
        if st.session_state.use_mongodb:
            if not hasattr(mongodb_connector, 'db') or mongodb_connector.db is None:
                if st.session_state.db_password:
                    success, error = mongodb_connector.connect(st.session_state.db_password)
                    if not success:
                        return f"连接MongoDB失败，数据仅保存在本地文件: {error}"
                else:
                    return "MongoDB密码未配置，数据仅保存在本地文件"
            
            # 保存到MongoDB
            if mongodb_connector.save_hospital_data(data):
                return f"医院数据文件 {file.name} 上传成功并保存到MongoDB"
            else:
                return f"保存到MongoDB失败，数据仅保存在本地文件"
        else:
            return f"医院数据文件 {file.name} 上传成功并设为当前数据源"
    except Exception as e:
        print(traceback.format_exc())
        return f"处理上传的医院文件时出错: {str(e)}"

def upload_distributor_data(file):
    """上传经销商数据文件"""
    try:
        if file is None:
            return "请选择要上传的文件"
        
        # 保存文件到本地（即使使用MongoDB也先保存本地备份）
        save_path = os.path.join("data", file.name)
        with open(save_path, "wb") as f:
            f.write(file.getbuffer())
        
        # 验证JSON格式
        data = load_local_json_data(save_path)
        if "distributors" not in data:
            return "文件格式无效：上传的经销商数据文件缺少 'distributors' 键"
        
        # 设置为当前文件路径
        st.session_state.distributor_data_path = save_path
        
        # 如果使用MongoDB，则保存到MongoDB
        if st.session_state.use_mongodb:
            if not hasattr(mongodb_connector, 'db') or mongodb_connector.db is None:
                if st.session_state.db_password:
                    success, error = mongodb_connector.connect(st.session_state.db_password)
                    if not success:
                        return f"连接MongoDB失败，数据仅保存在本地文件: {error}"
                else:
                    return "MongoDB密码未配置，数据仅保存在本地文件"
            
            # 保存到MongoDB
            if mongodb_connector.save_distributor_data(data):
                return f"经销商数据文件 {file.name} 上传成功并保存到MongoDB"
            else:
                return f"保存到MongoDB失败，数据仅保存在本地文件"
        else:
            return f"经销商数据文件 {file.name} 上传成功并设为当前数据源"
    except Exception as e:
        print(traceback.format_exc())
        return f"处理上传的经销商文件时出错: {str(e)}"

# --- Streamlit UI ---

# 页面配置
# st.set_page_config(layout="wide", page_title="医疗销售报告分析器") # 翻译 # MOVED FROM HERE

# 侧边栏
st.sidebar.title("导航") # 翻译
# 从 session_state 读取或设置默认值，确保在重跑时保持选择
if 'app_mode' not in st.session_state:
    st.session_state.app_mode = "分析报告" # 默认选项

app_mode = st.sidebar.radio("选择功能区:", # 翻译
                            ("分析报告", "搜索实体", "数据管理", "数据同步", "配置", "使用指南"), # 添加"数据同步"选项
                            index=["分析报告", "搜索实体", "数据管理", "数据同步", "配置", "使用指南"].index(st.session_state.app_mode), # 更新索引列表
                            key="app_mode_radio",
                            on_change=lambda: setattr(st.session_state, 'app_mode', st.session_state.app_mode_radio)) # 更新状态

st.sidebar.markdown("---")

# 显示当前数据源信息
st.sidebar.markdown("### 当前数据源")
if st.session_state.use_mongodb and hasattr(mongodb_connector, 'db') and mongodb_connector.db is not None:
    # 当使用MongoDB时
    st.sidebar.success("**数据源:** MongoDB数据库")
    st.sidebar.info(f"**数据库:** {MONGO_DB_NAME}")
    st.sidebar.info(f"**集合:** {MONGO_HOSPITALS_COLLECTION}, {MONGO_DISTRIBUTORS_COLLECTION}")
    
    # 尝试获取记录数量
    try:
        hospital_count = mongodb_connector.db[MONGO_HOSPITALS_COLLECTION].count_documents({})
        distributor_count = mongodb_connector.db[MONGO_DISTRIBUTORS_COLLECTION].count_documents({})
        st.sidebar.info(f"**医院记录:** {hospital_count} 条")
        st.sidebar.info(f"**经销商记录:** {distributor_count} 条")
    except Exception as e:
        st.sidebar.warning(f"无法获取记录数量: {e}")
else:
    # 当使用本地文件时
    st.sidebar.info("**数据源:** 本地JSON文件")
    st.sidebar.info(f"**医院数据文件:** {os.path.basename(st.session_state.hospital_data_path)}")
    st.sidebar.info(f"**经销商数据文件:** {os.path.basename(st.session_state.distributor_data_path)}")
    
    # 尝试获取记录数量
    try:
        hospital_data = load_local_json_data(st.session_state.hospital_data_path)
        distributor_data = load_local_json_data(st.session_state.distributor_data_path)
        hospital_count = len(hospital_data.get("hospitals", []))
        distributor_count = len(distributor_data.get("distributors", []))
        st.sidebar.info(f"**医院记录:** {hospital_count} 条")
        st.sidebar.info(f"**经销商记录:** {distributor_count} 条")
    except Exception as e:
        pass

# --- 主页面内容 ---

if st.session_state.app_mode == "分析报告":
    st.title("🩺 医疗销售日报分析器") # 翻译
    # 翻译描述
    st.markdown("在此粘贴日报内容，系统将自动分析实体信息、评估关系并提供行动建议。")

    col1, col2 = st.columns([2, 3]) # 调整比例

    with col1:
        # 翻译标签
        report_text = st.text_area("日报内容", height=350, value=sample_report, key="report_input_area")
        # 翻译标签和帮助文本
        use_mt = st.checkbox("使用多线程处理 (推荐)", value=st.session_state.use_multithreading_default, key="use_mt_checkbox", help="对于包含多个医生/经销商的报告，开启此项可以显著加快分析速度。")

        # 翻译按钮文本, 增加 disabled 状态下的提示
        analyze_button_disabled = (not client or not report_text.strip())
        analyze_button_tooltip = "请先配置API密钥并输入报告内容" if analyze_button_disabled else "开始分析报告" # 翻译
        analyze_button = st.button("分析报告", type="primary", disabled=analyze_button_disabled, help=analyze_button_tooltip, key="analyze_button") # 翻译

        # 进度条和状态文本占位符
        status_placeholder = st.empty()
        progress_placeholder = st.empty()

    if analyze_button:
        if not report_text.strip():
            st.warning("请先粘贴日报内容再进行分析。") # 翻译
        elif not client:
             st.error("无法进行分析：OpenAI 客户端未初始化。") # 翻译
        else:
            clear_process_logs()
            st.session_state.analysis_results = None

            status_placeholder.info("⏳ 开始分析，请稍候...") # 翻译 + 图标
            progress_bar = progress_placeholder.progress(0, text="初始化分析...") # 翻译
            def update_progress(percentage, message):
                percentage = max(0.0, min(1.0, percentage))
                try:
                    progress_bar.progress(percentage, text=str(message)) # 确保消息是字符串
                except Exception as e:
                    print(f"更新进度条时出错: {e}") # 英文日志

            try:
                analysis_output = analyze_daily_report_core(
                    report_text,
                    st.session_state.hospital_data_path,
                    st.session_state.distributor_data_path,
                    use_mt,
                    progress_callback=update_progress
                )

                if analysis_output and not analysis_output.get("error"):
                     formatted_html = format_results_html(analysis_output)
                     formatted_text = format_results_text(analysis_output)
                     num_doctors = len(analysis_output.get("doctor_evaluations", {}))
                     num_distributors = len(analysis_output.get("distributor_evaluations", {}))
                     # 翻译完成消息
                     final_message = f"✅ 分析完成！已处理 {num_doctors} 位医生/联系人和 {num_distributors} 个经销商的信息。"

                     st.session_state.analysis_results = {
                         "text": formatted_text,
                         "html": formatted_html,
                         "message": final_message,
                         "raw": analysis_output
                     }
                     status_placeholder.success(final_message)
                     progress_bar.progress(1.0, text="分析完成！") # 翻译

                else:
                    error_msg = analysis_output.get('error', '分析过程中发生未知错误。') if analysis_output else '分析未返回任何结果。' # 翻译
                    st.session_state.analysis_results = {"error": error_msg}
                    status_placeholder.error(f"分析失败: {error_msg}") # 翻译
                    progress_bar.progress(1.0, text="分析失败！") # 翻译

            except Exception as e:
                 final_error = f"执行分析时发生严重错误: {e}" # 翻译
                 st.session_state.analysis_results = {"error": final_error}
                 status_placeholder.error(final_error)
                 progress_bar.progress(1.0, text="严重错误！") # 翻译
                 print(traceback.format_exc())


    # 显示结果 (在按钮点击后的重新渲染时，或如果结果已存在于 session state)
    if st.session_state.analysis_results:
        results = st.session_state.analysis_results
        with col2:
            if "error" in results:
                st.error(f"分析出错: {results['error']}") # 翻译
            elif results.get("html"):
                 st.markdown("### 分析输出") # 翻译
                 st.markdown(results["html"], unsafe_allow_html=True) # HTML 内容本身已包含中文

                 # 摘要信息 (使用修正后的 for 循环)
                 st.markdown("<hr>", unsafe_allow_html=True)
                 st.markdown("<h5>概要信息</h5>", unsafe_allow_html=True) # 翻译
                 entities = results.get("raw", {}).get("entities", {})
                 scol1, scol2, scol3 = st.columns(3)
                 with scol1:
                     st.markdown("<h6>医院</h6>", unsafe_allow_html=True) # 翻译
                     hospitals = entities.get("hospitals", [])
                     if hospitals:
                         for h in hospitals: # 使用 for 循环
                             st.markdown(f"- {h}")
                     else:
                         st.caption("未找到") # 翻译
                 with scol2:
                     st.markdown("<h6>科室</h6>", unsafe_allow_html=True) # 翻译
                     departments = entities.get("departments", [])
                     if departments:
                         for d in departments: # 使用 for 循环
                             st.markdown(f"- {d}")
                     else:
                         st.caption("未找到") # 翻译
                 with scol3:
                     st.markdown("<h6>产品</h6>", unsafe_allow_html=True) # 翻译
                     products = entities.get("products", [])
                     if products:
                         for p in products: # 使用 for 循环
                             st.markdown(f"- {p}")
                     else:
                         st.caption("未找到") # 翻译

                 # 折叠项
                 with st.expander("查看纯文本结果"): # 翻译
                     # 翻译标签和默认值
                     st.text_area("文本输出", value=results.get("text", "无纯文本结果。"), height=400, disabled=True, key="text_output_area")
                 with st.expander("查看处理步骤"): # 翻译
                      st.markdown(format_process_details_html(), unsafe_allow_html=True) # HTML 内容已包含中文标题
            else:
                 st.info("分析未生成可显示的结果。") # 翻译


elif st.session_state.app_mode == "搜索实体":
    st.title("🔍 搜索实体") # 翻译
    st.markdown("在已加载的历史数据中搜索医生、医院或经销商信息。") # 翻译

    # 显示MongoDB连接状态
    mongodb_db_exists = hasattr(mongodb_connector, 'db') and mongodb_connector.db is not None
    if st.session_state.use_mongodb:
        if mongodb_db_exists:
            st.success("MongoDB 连接状态: 已连接")
        else:
            st.error("MongoDB 连接状态: 未连接，请在设置中配置数据库")
    else:
        st.info("当前使用本地文件模式，未使用MongoDB")

    # 翻译标签
    search_term = st.text_input("输入搜索关键词 (例如：医生姓名、医院名称、经销商名称):", key="search_input")
    
    # 替换原有单一搜索按钮，改为两种不同的搜索按钮
    col1, col2 = st.columns(2)
    with col1:
        search_doctor_button = st.button("按医生搜索", type="primary", key="search_doctor_button")
    with col2:
        search_hospital_button = st.button("按医院搜索", type="primary", key="search_hospital_button")

    # 按医生搜索
    if search_doctor_button and search_term:
        # 添加处理日志清理
        clear_process_logs()
        # 翻译 Spinner 文本
        with st.spinner(f"正在搜索医生 \"{search_term}\"..."):
            search_results = {"doctors": [], "hospitals": [], "distributors": []}
            try:
                # 检查MongoDB连接是否有效
                mongodb_db_exists = hasattr(mongodb_connector, 'db') and mongodb_connector.db is not None
                add_process_log(f"MongoDB连接状态: {'有效' if mongodb_db_exists and st.session_state.use_mongodb else '未使用或无效'}")
                
                if st.session_state.use_mongodb and mongodb_db_exists:
                    # 使用医生姓名模糊搜索
                    search_pattern = {"$regex": search_term, "$options": "i"}
                    
                    # 修改查询逻辑，添加更多匹配模式来处理元组格式的医生姓名
                    doctor_query = {
                        "$or": [
                            # 直接匹配医生姓名字段
                            {"医生姓名": search_pattern},
                            # 匹配元组格式中的最后一个元素（实际医生姓名）
                            {"医生姓名": {"$regex": f".*'{search_term}'.*", "$options": "i"}}
                        ]
                    }
                    add_process_log(f"搜索医生: 使用MongoDB查询 = {doctor_query}")
                    
                    all_docs = list(mongodb_connector.db[MONGO_HOSPITALS_COLLECTION].find(doctor_query, {"_id": 0}))
                    add_process_log(f"搜索医生: 找到 {len(all_docs)} 条记录")
                    
                    # 将沟通内容添加到结果中
                    for doc in all_docs:
                        # 处理医生姓名，从元组格式中提取实际姓名
                        doctor_name = doc.get("医生姓名", "")
                        original_doctor_name = doctor_name  # 保存原始名称以便在调试时显示
                        
                        # 使用process_doctor_name函数处理
                        processed_name = process_doctor_name(doctor_name)
                        
                        add_process_log(f"处理医生记录: 原始姓名='{original_doctor_name}', 处理后='{processed_name}'")
                        
                        search_results["doctors"].append({
                            "姓名": processed_name,
                            "医院": doc.get("医院名称", "未知医院"),
                            "科室": doc.get("科室", "未知科室"),
                            "最近拜访": format_date(str(doc.get("拜访日期", "未知日期"))),
                            "拜访员工": doc.get("拜访员工", "未知人员"),
                            "沟通内容": doc.get("沟通内容", "无记录"),
                            "后续行动": doc.get("后续行动", "无记录"),
                            "拜访日期原始值": doc.get("拜访日期", "")  # 用于排序
                        })
                else:
                    # 使用本地文件
                    add_process_log(f"搜索医生: 使用本地文件 {st.session_state.hospital_data_path}")
                    hospital_data = load_json_data(st.session_state.hospital_data_path)
                    
                    # 记录本地数据统计
                    total_hospitals = len(hospital_data.get("hospitals", []))
                    total_doctors = sum(len(h.get("历史记录", [])) for h in hospital_data.get("hospitals", []))
                    add_process_log(f"搜索医生: 本地数据包含 {total_hospitals} 家医院, {total_doctors} 条医生记录")
                    
                    for hospital in hospital_data.get("hospitals", []):
                        h_name = hospital.get("医院名称", "未知医院")
                        for record in hospital.get("历史记录", []):
                            doc_name = record.get("医生姓名", "")
                            # 修改为模糊匹配
                            if search_term.lower() in doc_name.lower():
                                # 处理医生姓名
                                processed_name = process_doctor_name(doc_name)
                                search_results["doctors"].append({
                                    "姓名": processed_name,
                                    "医院": h_name,
                                    "科室": record.get("科室", "未知科室"),
                                    "最近拜访": record.get("拜访日期", "未知日期"),
                                    "拜访员工": record.get("拜访员工", "未知人员"),
                                    "沟通内容": record.get("沟通内容", "无记录"),
                                    "后续行动": record.get("后续行动", "无记录"),
                                    "拜访日期原始值": record.get("拜访日期", "")  # 用于排序
                                })
                
                # 按照拜访日期排序（时间从新到旧）并只保留前10条
                search_results["doctors"] = sorted(
                    search_results["doctors"], 
                    key=lambda x: str(x.get("拜访日期原始值", "")), 
                    reverse=True
                )[:10]
                
                add_process_log(f"搜索医生: 找到并排序后的结果数量 = {len(search_results['doctors'])}")
                # 记录搜索结果中的医生姓名
                for i, doc in enumerate(search_results["doctors"]):
                    add_process_log(f"结果 #{i+1}: 医生姓名 = '{doc.get('姓名', '')}', 医院 = '{doc.get('医院', '')}'")
                
                # 使用Streamlit原生组件显示结果
                st.session_state.search_results = search_results
                st.session_state.search_type = "doctor"
            except Exception as e:
                error_msg = f"搜索医生过程中出错: {e}"
                add_process_log(error_msg)
                add_process_log(traceback.format_exc())
                st.error(error_msg)
                print(traceback.format_exc())
    
    # 按医院搜索
    elif search_hospital_button and search_term:
        clear_process_logs()
        add_process_log(f"开始按医院搜索，关键词: '{search_term}'")
        with st.spinner(f"正在搜索医院 \"{search_term}\"..."):
            search_results_data = {"doctors": [], "hospitals": [], "distributors": []}
            try:
                mongodb_db_exists = hasattr(mongodb_connector, 'db') and mongodb_connector.db is not None
                add_process_log(f"MongoDB连接状态: {'有效' if mongodb_db_exists and st.session_state.use_mongodb else '未使用或无效'}")

                if st.session_state.use_mongodb and mongodb_db_exists:
                    search_pattern = {"$regex": search_term, "$options": "i"}
                    # Query for hospitals matching the search term in either name field
                    hospital_query = {"$or": [
                        {"医院名称": search_pattern},
                        {"标准医院名称": search_pattern}
                    ]}
                    add_process_log(f"搜索医院: 使用MongoDB查询 = {hospital_query}")

                    # Fetch all matching documents
                    all_matching_hospital_documents = list(mongodb_connector.db[MONGO_HOSPITALS_COLLECTION].find(hospital_query, {"_id": 0}))
                    add_process_log(f"搜索医院: MongoDB找到 {len(all_matching_hospital_documents)} 个原始匹配文档。")

                    if not all_matching_hospital_documents:
                        add_process_log("MongoDB查询未返回任何与医院相关的文档。")
                    
                    # --- Grouping and Aggregation Logic ---
                    grouped_hospitals = {} # Key: hospital_id (or standard_name), Value: aggregated data

                    for hospital_doc in all_matching_hospital_documents:
                        # Determine the grouping key: Prefer 医院ID, fallback to 标准医院名称
                        grouping_key = hospital_doc.get("医院ID")
                        if not grouping_key: # Fallback if 医院ID is missing or empty
                            grouping_key = hospital_doc.get("标准医院名称")
                        
                        if not grouping_key: # If both are missing, use 医院名称 as a last resort or skip
                            grouping_key = hospital_doc.get("医院名称", f"unknown_hospital_{uuid.uuid4().hex[:6]}")
                            add_process_log(f"警告: 医院文档缺少 医院ID 和 标准医院名称。使用临时键: {grouping_key}. Doc: {hospital_doc.get('医院名称')}")

                        if grouping_key not in grouped_hospitals:
                            grouped_hospitals[grouping_key] = {
                                "display_name": hospital_doc.get("标准医院名称") or hospital_doc.get("医院名称"), # Prefer standard name for display
                                "all_names_found": {hospital_doc.get("医院名称"), hospital_doc.get("标准医院名称")}, # Set to store unique names
                                "all_departments": set(), # To store unique department names
                                "doctor_visit_records": [],
                                "original_doc_count": 0 # How many raw DB docs contributed to this group
                            }
                        
                        # Aggregate data
                        grouped_hospitals[grouping_key]["original_doc_count"] += 1
                        grouped_hospitals[grouping_key]["all_names_found"].add(hospital_doc.get("医院名称"))
                        grouped_hospitals[grouping_key]["all_names_found"].add(hospital_doc.get("标准医院名称"))
                        
                        # Assuming your documents store科室 at the top level or within visit records
                        if hospital_doc.get("科室"):
                             grouped_hospitals[grouping_key]["all_departments"].add(hospital_doc.get("科室"))

                        # Process and add doctor visit records from this document
                        # The example doc shows visit details at the top level, not in a "历史记录" array.
                        # Let's assume one document = one visit for this structure.
                        # If a single hospital document *can* have multiple "历史记录", this needs adjustment.
                        
                        # For the provided schema, one document IS one visit record.
                        doctor_name_raw = hospital_doc.get("医生姓名")
                        processed_doc_name = doctor_name_raw # Default
                        if isinstance(doctor_name_raw, str) and doctor_name_raw.startswith("(") and doctor_name_raw.endswith(")"):
                            try:
                                name_tuple = ast.literal_eval(doctor_name_raw)
                                if isinstance(name_tuple, tuple) and len(name_tuple) > 0:
                                    processed_doc_name = str(name_tuple[-1]).strip()
                            except: pass # Keep raw if parsing fails
                        
                        # Add this "visit" to the grouped hospital's records
                        grouped_hospitals[grouping_key]["doctor_visit_records"].append({
                            "姓名": processed_doc_name,
                            "拜访日期": format_date(str(hospital_doc.get("拜访日期", ""))), # ensure string for format_date
                            "拜访员工": hospital_doc.get("拜访员工", "未知人员"),
                            "沟通内容": hospital_doc.get("沟通内容", "无记录"),
                            "后续行动": hospital_doc.get("后续行动", "无记录"),
                            "拜访日期原始值": hospital_doc.get("拜访日期", ""), # For sorting
                            "科室": hospital_doc.get("科室", "未知科室") # If visit-specific department
                        })
                        # If科室 is part of the visit, add it to the hospital's set
                        if hospital_doc.get("科室"):
                            grouped_hospitals[grouping_key]["all_departments"].add(hospital_doc.get("科室"))


                    add_process_log(f"搜索医院: {len(all_matching_hospital_documents)} 原始文档被分组为 {len(grouped_hospitals)} 个独立医院。")

                    # Now, format the grouped_hospitals for display
                    final_hospital_list_for_results = []
                    for key, agg_data in grouped_hospitals.items():
                        # Sort doctor visits within this hospital by date
                        sorted_doctor_visits = sorted(
                            agg_data["doctor_visit_records"],
                            key=lambda x: str(x.get("拜访日期原始值", "")), # Convert to string for robust sort
                            reverse=True
                        )[:10] # Limit to latest 10 visits per hospital

                        final_hospital_list_for_results.append({
                            "名称": agg_data["display_name"], # The "best" name for display
                            "科室数量": len(agg_data["all_departments"]),
                            "历史记录数量": len(agg_data["doctor_visit_records"]), # Total visits for this hospital
                            "医生信息": sorted_doctor_visits, # Top 10 sorted visits for display
                            "debug_grouped_from_names": list(filter(None, agg_data["all_names_found"])),
                            "debug_original_doc_count": agg_data["original_doc_count"]
                        })
                        add_process_log(f"处理合并后医院 '{agg_data['display_name']}': 总拜访记录 {len(agg_data['doctor_visit_records'])}, 显示前 {len(sorted_doctor_visits)}。原始名称变体: {list(filter(None, agg_data['all_names_found']))}")

                    search_results_data["hospitals"] = final_hospital_list_for_results
                
                else: # Local file search
                    add_process_log(f"搜索医院: 使用本地文件 {st.session_state.hospital_data_path}")
                    hospital_data_local = load_json_data(st.session_state.hospital_data_path)
                    # IMPORTANT: The local file structure is different ("hospitals" list, each with "历史记录" array)
                    # The grouping logic for local files needs to be adapted if you want similar consolidation.
                    # For now, I'll keep the existing local file logic which doesn't do this deep grouping.
                    # If you need it, it would involve iterating hospitals, then their history, and grouping by hospital name.

                    temp_local_hospitals = []
                    for hospital_entry in hospital_data_local.get("hospitals", []): # This is a list of hospital objects
                        hospital_name_local = hospital_entry.get("医院名称", "")
                        if search_term.lower() in hospital_name_local.lower():
                            doctors_info = []
                            for record in hospital_entry.get("历史记录", []): # This is the list of visits
                                doc_name_raw = record.get("医生姓名", "")
                                processed_doc_name = doc_name_raw
                                if isinstance(doc_name_raw, str) and doc_name_raw.startswith("(") and doc_name_raw.endswith(")"):
                                    try:
                                        name_tuple = ast.literal_eval(doc_name_raw)
                                        if isinstance(name_tuple, tuple) and len(name_tuple) > 0:
                                            processed_doc_name = str(name_tuple[-1]).strip()
                                    except: pass
                                
                                doctors_info.append({
                                    "姓名": processed_doc_name,
                                    "拜访日期": format_date(record.get("拜访日期", "未知日期")),
                                    "拜访员工": record.get("拜访员工", "未知人员"),
                                    "沟通内容": record.get("沟通内容", "无记录"),
                                    "后续行动": record.get("后续行动", "无记录"),
                                    "拜访日期原始值": record.get("拜访日期", "")
                                })
                            
                            sorted_doctors = sorted(
                                doctors_info, 
                                key=lambda x: str(x.get("拜访日期原始值", "")),
                                reverse=True
                            )[:10]
                            
                            temp_local_hospitals.append({
                                "名称": hospital_name_local,
                                "科室数量": len(hospital_entry.get("科室", [])), # Assumes "科室" is a list at hospital_entry level
                                "历史记录数量": len(hospital_entry.get("历史记录", [])),
                                "医生信息": sorted_doctors
                            })
                    search_results_data["hospitals"] = temp_local_hospitals
                    add_process_log(f"本地文件搜索医院: 找到 {len(search_results_data['hospitals'])} 家医院。")

                add_process_log(f"搜索医院: 最终结果中的合并后医院数量 = {len(search_results_data['hospitals'])}")
                st.session_state.search_results = search_results_data
                st.session_state.search_type = "hospital"

            except Exception as e:
                error_msg = f"搜索医院过程中出错: {e}"
                add_process_log(error_msg)
                add_process_log(traceback.format_exc())
                st.error(error_msg)
                print(traceback.format_exc())
                st.session_state.search_results = {"doctors": [], "hospitals": [], "distributors": []}
                st.session_state.search_type = "hospital"
    
    # 搜索提示
    elif (search_doctor_button or search_hospital_button) and not search_term:
        st.warning("请输入搜索关键词。") # 翻译

    # 使用Streamlit原生组件显示结果
    if 'search_results' in st.session_state and 'search_type' in st.session_state:
        search_results = st.session_state.search_results
        search_type = st.session_state.search_type
        
        # 添加处理日志展示区
        with st.expander("📋 显示处理日志", expanded=False):
            st.code(get_process_logs_str(), language="text")
            
        # 显示医生搜索结果
        if search_type == "doctor" and search_results["doctors"]:
            st.subheader("医生搜索结果")
            st.caption("（最多显示最新的10条记录）")
            
            for i, doc in enumerate(search_results["doctors"]):
                with st.container():
                    cols = st.columns([3, 2, 2, 2, 1])
                    with cols[0]:
                        st.markdown(f"**{doc.get('姓名', '')}**")
                    with cols[1]:
                        st.markdown(f"医院: {doc.get('医院', '')}")
                    with cols[2]:
                        st.markdown(f"科室: {doc.get('科室', '')}")
                    with cols[3]:
                        st.markdown(f"拜访: {doc.get('最近拜访', '')}")
                    with cols[4]:
                        st.markdown(f"拜访人: {doc.get('拜访员工', '')}")
                
                # 使用expander显示详细信息
                with st.expander("查看沟通内容"):
                    st.markdown("**沟通内容:**")
                    st.markdown(doc.get('沟通内容', '无记录'))
                    st.markdown("**后续行动:**")
                    st.markdown(doc.get('后续行动', '无记录'))
                
                # 添加分隔线
                if i < len(search_results["doctors"]) - 1:
                    st.markdown("---")
        
        # 显示医院搜索结果
        elif search_type == "hospital" and search_results["hospitals"]:
            st.subheader("医院搜索结果")
            st.caption("（最多显示10个医院）")
            
            for hospital in search_results["hospitals"]:
                st.markdown(f"## {hospital.get('名称', '')}")
                st.markdown(f"科室数量: {hospital.get('科室数量', 0)}, 历史记录数量: {hospital.get('历史记录数量', 0)}")
                
                # 显示医生信息
                doctors_info = hospital.get('医生信息', [])
                if doctors_info:
                    st.markdown("### 医生记录 (最新10条)")
                    
                    for i, doc in enumerate(doctors_info):
                        with st.container():
                            cols = st.columns([2, 2, 2])
                            with cols[0]:
                                st.markdown(f"**{doc.get('姓名', '')}**")
                            with cols[1]:
                                st.markdown(f"拜访: {doc.get('拜访日期', '')}")
                            with cols[2]:
                                st.markdown(f"拜访人: {doc.get('拜访员工', '')}")
                        
                        # 使用expander显示详细信息
                        with st.expander("查看沟通内容"):
                            st.markdown("**沟通内容:**")
                            st.markdown(doc.get('沟通内容', '无记录'))
                            st.markdown("**后续行动:**")
                            st.markdown(doc.get('后续行动', '无记录'))
                        
                        # 添加分隔线
                        if i < len(doctors_info) - 1:
                            st.markdown("---")
                else:
                    st.markdown("该医院暂无医生记录")
                
                # 医院之间添加更明显的分隔
                st.markdown("<hr style='height:3px;border:none;background-color:#f0f2f6;'>", unsafe_allow_html=True)


elif st.session_state.app_mode == "数据管理":
    st.title("📊 数据管理") # 翻译
    
    # 修复MongoDB数据结构按钮
    if st.session_state.use_mongodb:
        st.subheader("修复MongoDB数据结构")
        st.markdown("如果您的MongoDB中数据显示不正确（例如看到`_id`字段或日期时间对象），请使用此功能修复数据结构。")
        
        fix_col1, fix_col2 = st.columns(2)
        with fix_col1:
            if st.button("修复医院数据结构", help="将扁平化的医院记录转换为正确的嵌套结构"):
                if not hasattr(mongodb_connector, 'db') or mongodb_connector.db is None:
                    if st.session_state.db_password:
                        success, error = mongodb_connector.connect(st.session_state.db_password)
                        if not success:
                            st.error(f"连接MongoDB失败: {error}")
                    else:
                        st.error("请先在配置页面设置MongoDB密码")
                
                if hasattr(mongodb_connector, 'db') and mongodb_connector.db is not None:
                    if hasattr(mongodb_connector, 'fix_hospital_data') and callable(mongodb_connector.fix_hospital_data):
                        if mongodb_connector.fix_hospital_data():
                            st.success("医院数据结构修复成功！")
                        else:
                            st.error("医院数据结构修复失败，请查看控制台日志。")
                    else:
                        st.error("mongodb_connector没有fix_hospital_data方法，请检查MongoDB连接器代码。")
        
        with fix_col2:
            if st.button("修复经销商数据结构", help="将扁平化的经销商记录转换为正确的嵌套结构"):
                if not hasattr(mongodb_connector, 'db') or mongodb_connector.db is None:
                    if st.session_state.db_password:
                        success, error = mongodb_connector.connect(st.session_state.db_password)
                        if not success:
                            st.error(f"连接MongoDB失败: {error}")
                    else:
                        st.error("请先在配置页面设置MongoDB密码")
                
                if hasattr(mongodb_connector, 'db') and mongodb_connector.db is not None:
                    if hasattr(mongodb_connector, 'fix_distributor_data') and callable(mongodb_connector.fix_distributor_data):
                        if mongodb_connector.fix_distributor_data():
                            st.success("经销商数据结构修复成功！")
                        else:
                            st.error("经销商数据结构修复失败，请查看控制台日志。")
                    else:
                        st.error("mongodb_connector没有fix_distributor_data方法，请检查MongoDB连接器代码。")
        
        st.markdown("---")
    
    # 使用tabs代替radio按钮
    tab1, tab2, tab3 = st.tabs(["加载示例数据", "查看数据", "上传数据"])
    
    with tab1:
        st.subheader("加载示例数据")
        st.markdown("**警告:** 加载示例数据将覆盖当前数据。如果您使用MongoDB，数据也会保存到数据库中。")
        
        # 新增：加载示例数据前的确认
        confirm_load_sample = st.checkbox("我确认要覆盖当前数据并加载示例数据。", key="confirm_load_sample_checkbox")
        
        if st.button("加载示例数据", key="load_sample", disabled=(not confirm_load_sample)):
            if confirm_load_sample:
                result = load_sample_data_action()
                st.success(result) # 使用成功样式
            else:
                st.warning("请先勾选确认框，再加载示例数据。")
    
    with tab2:
        st.subheader("查看当前数据")
        view_tab1, view_tab2 = st.tabs(["医院数据", "经销商数据"])
        
        with view_tab1:
            if st.button("刷新医院数据", key="view_hospital"):
                with st.spinner("加载医院数据中..."):
                    try:
                        data_json = view_hospital_data()
                        st.code(data_json)
                    except Exception as e:
                        st.error(f"加载医院数据时出错: {str(e)}")
        
        with view_tab2:
            if st.button("刷新经销商数据", key="view_distributor"):
                with st.spinner("加载经销商数据中..."):
                    try:
                        data_json = view_distributor_data()
                        st.code(data_json)
                    except Exception as e:
                        st.error(f"加载经销商数据时出错: {str(e)}")
    
    with tab3:
        st.subheader("上传数据")
        
        # 显示当前数据源信息
        if st.session_state.use_mongodb:
            st.info("数据将上传到MongoDB数据库和本地文件")
        else:
            st.info("数据将仅上传到本地文件")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("医院数据")
            hospital_file_uploader = st.file_uploader("上传医院数据文件 (JSON)", type=["json"], key="hospital_file_uploader")

            if hospital_file_uploader is not None:
                try:
                    # 尝试读取并预览
                    st.session_state.uploaded_hospital_file_content = json.loads(hospital_file_uploader.getvalue().decode("utf-8"))
                    st.write("文件预览 (前5条医院记录):")
                    preview_data = st.session_state.uploaded_hospital_file_content.get("hospitals", [])[:5]
                    if preview_data:
                        st.json(preview_data, expanded=False)
                    else:
                        st.warning("文件中未找到 'hospitals' 列表或列表为空。")
                    
                    if "hospitals" not in st.session_state.uploaded_hospital_file_content:
                         st.error("文件格式无效：上传的医院数据文件缺少 'hospitals' 键。无法进行上传。")
                    elif st.button("确认上传医院数据", key="confirm_upload_hospital"):
                        # 使用暂存的内容进行上传
                        result = upload_hospital_data(hospital_file_uploader, st.session_state.uploaded_hospital_file_content)
                        st.success(result)
                        st.session_state.uploaded_hospital_file_content = None # 清理
                        st.experimental_rerun() # 重新运行以更新uploader状态

                except json.JSONDecodeError:
                    st.error("无法解析JSON文件。请确保文件格式正确。")
                    st.session_state.uploaded_hospital_file_content = None
                except Exception as e:
                    st.error(f"预览文件时出错: {e}")
                    st.session_state.uploaded_hospital_file_content = None
            
        
        with col2:
            st.subheader("经销商数据")
            distributor_file_uploader = st.file_uploader("上传经销商数据文件 (JSON)", type=["json"], key="distributor_file_uploader")

            if distributor_file_uploader is not None:
                try:
                    st.session_state.uploaded_distributor_file_content = json.loads(distributor_file_uploader.getvalue().decode("utf-8"))
                    st.write("文件预览 (前5条经销商记录):")
                    preview_data = st.session_state.uploaded_distributor_file_content.get("distributors", [])[:5]
                    if preview_data:
                        st.json(preview_data, expanded=False)
                    else:
                        st.warning("文件中未找到 'distributors' 列表或列表为空。")

                    if "distributors" not in st.session_state.uploaded_distributor_file_content:
                        st.error("文件格式无效：上传的经销商数据文件缺少 'distributors' 键。无法进行上传。")
                    elif st.button("确认上传经销商数据", key="confirm_upload_distributor"):
                        result = upload_distributor_data(distributor_file_uploader, st.session_state.uploaded_distributor_file_content)
                        st.success(result)
                        st.session_state.uploaded_distributor_file_content = None # 清理
                        st.experimental_rerun() # 重新运行以更新uploader状态
                
                except json.JSONDecodeError:
                    st.error("无法解析JSON文件。请确保文件格式正确。")
                    st.session_state.uploaded_distributor_file_content = None
                except Exception as e:
                    st.error(f"预览文件时出错: {e}")
                    st.session_state.uploaded_distributor_file_content = None


elif st.session_state.app_mode == "数据同步":
    st.title("📊 数据同步") # 翻译
    st.markdown("在这里，您可以执行数据同步、主数据管理和实体标准化等操作。")
    
    # 检查MongoDB连接状态
    mongodb_db_exists = hasattr(mongodb_connector, 'db') and mongodb_connector.db is not None
    if st.session_state.use_mongodb:
        if not mongodb_db_exists:
            st.error("MongoDB 未连接，请在配置页面设置数据库连接。")
        else:
            st.success("MongoDB 已连接，可以执行数据同步操作。")
    else:
        st.warning("当前使用本地文件模式，未使用MongoDB。要进行高级数据同步功能，建议启用MongoDB。")
    
    tab1, tab2, tab3 = st.tabs(["主数据同步", "名称标准化", "数据优化"])
    
    with tab1:
        st.subheader("主数据同步")
        st.markdown("将主数据（医院、医生、科室、经销商等）同步到MongoDB，用于数据标准化。")
        
        sync_types = []
        # 创建多选框
        col1, col2 = st.columns(2)
        with col1:
            if st.checkbox("同步医院数据", value=True):
                sync_types.append("hospitals")
            if st.checkbox("同步医生数据", value=True):
                sync_types.append("doctors")
        with col2:
            if st.checkbox("同步科室数据", value=True):
                sync_types.append("departments")
            if st.checkbox("同步经销商数据", value=True):
                sync_types.append("distributors")
        
        # 批次大小选择
        batch_size = st.slider("批量处理大小", min_value=10, max_value=500, value=100, step=10,
                            help="每批次处理的记录数量，较小的值更安全但速度较慢")
        
        # 同步按钮
        if st.button("开始同步主数据", type="primary"):
            if not mongodb_db_exists:
                st.error("MongoDB未连接，无法执行同步操作。")
            elif not sync_types:
                st.warning("请至少选择一种数据类型进行同步。")
            else:
                # 清空处理日志
                clear_process_logs()
                with st.spinner(f"正在同步主数据 ({', '.join(sync_types)})..."):
                    try:
                        # 获取MongoDB客户端
                        mongo_client = mongodb_connector.client
                        add_process_log(f"开始同步主数据，选择的类型: {sync_types}")
                        
                        # 调用new_report中的同步函数
                        result = new_report.sync_master_data_to_mongodb(
                            mongo_client, 
                            sync_types=sync_types, 
                            batch_size=batch_size
                        )
                        
                        if result:
                            st.success("主数据同步成功！")
                        else:
                            st.error("主数据同步失败，请查看日志。")
                    except Exception as e:
                        error_msg = f"同步主数据时出错: {e}"
                        add_process_log(error_msg)
                        add_process_log(traceback.format_exc())
                        st.error(error_msg)
        
        # 显示处理日志
        with st.expander("同步处理日志", expanded=False):
            st.code(get_process_logs_str(), language="text")
    
    with tab2:
        st.subheader("名称标准化")
        st.markdown("对医院名称和医生姓名进行标准化处理，确保数据一致性。")
        
        with st.expander("步骤 1: 上传标准医院参考数据", expanded=True):
            st.markdown("上传包含医院标准名称和位置信息的Excel文件，用于后续的名称匹配和标准化。")
            hospital_reference_file = st.file_uploader("上传医院参考数据 (Excel 文件，需含 'name' 和 'location' 列)", type=["xlsx"], key="hospital_ref_uploader_sync")
            
            if hospital_reference_file:
                try:
                    # 保存上传的文件
                    save_path = os.path.join("data", "hospital_reference_data.xlsx") # 使用固定文件名或基于会话的唯一文件名
                    with open(save_path, "wb") as f:
                        f.write(hospital_reference_file.getbuffer())
                    st.session_state.hospital_reference_path = save_path
                    st.success(f"已上传参考文件: {hospital_reference_file.name} 并保存为 {save_path}")
                    
                    # 加载并预览数据
                    df = pd.read_excel(save_path)
                    if 'name' not in df.columns or 'location' not in df.columns:
                        st.warning("文件必须包含 'name' 和 'location' 列。请检查文件内容。")
                        st.session_state.hospital_reference_df = None
                    else:
                        st.session_state.hospital_reference_df = df
                        st.write(f"已加载 {len(df)} 条医院参考数据。")
                        if st.checkbox("显示参考数据预览 (前10条)"):
                            st.dataframe(df.head(10))
                except Exception as e:
                    st.error(f"读取或保存Excel文件时出错: {e}")
                    st.session_state.hospital_reference_path = None
                    st.session_state.hospital_reference_df = None
            else:
                 # 清理状态，如果用户移除了文件
                 st.session_state.hospital_reference_path = None
                 st.session_state.hospital_reference_df = None

        with st.expander("步骤 2: 执行医院名称标准化", expanded=False):
            st.markdown("使用上传的参考数据，对MongoDB中的医院名称进行标准化处理。")
            if not st.session_state.get("hospital_reference_df", None) is not None:
                st.warning("请先在步骤 1 上传有效的医院参考数据文件。")
            else:
                if st.button("开始医院名称标准化", type="primary"):
                    if not mongodb_db_exists:
                        st.error("MongoDB未连接，无法执行标准化操作。")
                    else:
                        clear_process_logs()
                        with st.spinner("正在标准化医院名称..."):
                            try:
                                mongo_client = mongodb_connector.client
                                df_reference = st.session_state.hospital_reference_df
                                add_process_log("开始医院名称标准化处理...")
                                
                                # 确保 new_report 函数可用
                                if not hasattr(new_report, 'build_hospital_vector_index') or \
                                   not hasattr(new_report, 'preprocess_standardize_hospital_names'):
                                    st.error("new_report.py 模块缺少必要的函数 (build_hospital_vector_index 或 preprocess_standardize_hospital_names)")
                                    raise ImportError("new_report.py 缺少函数")

                                add_process_log("从MongoDB加载当前医院数据...")
                                hospital_data_from_db = {"hospitals": list(mongo_client.get_database(MONGO_DB_NAME)[MONGO_HOSPITALS_COLLECTION].find({}, {"_id": 0}))}
                                add_process_log(f"已从MongoDB加载 {len(hospital_data_from_db['hospitals'])} 条医院记录")
                                
                                add_process_log("构建医院向量索引...")
                                vec_model, vec_index, vec_names, vec_locs = new_report.build_hospital_vector_index(df_reference)
                                add_process_log("医院向量索引构建完成")
                                
                                employee_regions_map = {} # 实际应用中可从其他地方加载
                                
                                add_process_log("执行医院名称标准化...")
                                # 注意：preprocess_standardize_hospital_names 应该返回清晰的更新指令或直接更新
                                # 为了安全，这里我们假设它返回了更新后的完整数据
                                updated_hospital_collection = new_report.preprocess_standardize_hospital_names(
                                    hospital_data_from_db, 
                                    df_reference, 
                                    employee_regions_map,
                                    vec_model=vec_model, 
                                    vec_index=vec_index, 
                                    vec_names=vec_names, 
                                    vec_locs=vec_locs
                                )
                                
                                add_process_log("标准化完成，准备更新MongoDB...")
                                if updated_hospital_collection and "hospitals" in updated_hospital_collection:
                                    target_collection = mongo_client.get_database(MONGO_DB_NAME)[MONGO_HOSPITALS_COLLECTION]
                                    
                                    # 更安全的更新方式：逐条更新或批量更新，而不是完全替换
                                    # 这里简化为先删除后插入，实际项目中应采用更细致的更新策略
                                    with st.spinner("正在更新MongoDB中的医院数据，这可能需要一些时间..."):
                                        target_collection.delete_many({}) # 清空旧数据
                                        if updated_hospital_collection["hospitals"]: # 确保列表不为空
                                            target_collection.insert_many(updated_hospital_collection["hospitals"]) # 插入新数据
                                        st.success(f"医院名称标准化完成！成功更新/插入 {len(updated_hospital_collection['hospitals'])} 条医院记录到MongoDB。")
                                        add_process_log(f"成功更新 {len(updated_hospital_collection['hospitals'])} 条医院记录")
                                else:
                                    st.error("医院名称标准化失败，未返回有效数据结构。")
                                    add_process_log("标准化未返回有效的医院数据 ('hospitals' 键缺失或数据为空)")
                                    
                            except ImportError as ie:
                                st.error(f"导入错误: {ie}")
                                add_process_log(f"导入错误: {ie}")
                            except Exception as e:
                                error_msg = f"医院名称标准化过程中出错: {e}"
                                add_process_log(error_msg)
                                add_process_log(traceback.format_exc())
                                st.error(error_msg)
        
        # 显示处理日志
        with st.expander("标准化处理日志", expanded=False):
            st.code(get_process_logs_str(), language="text")
    
    with tab3:
        st.subheader("数据优化")
        st.markdown("对现有数据进行优化处理，提取关键信息，改进数据质量。")
        
        # 清理之前的处理结果状态 (确保在每次进入此tab时状态干净，或根据需要保留)
        # if 'optimized_records' not in st.session_state: st.session_state.optimized_records = None
        # if 'processed_hospital_data' not in st.session_state: st.session_state.processed_hospital_data = None
        # if 'processed_distributor_data' not in st.session_state: st.session_state.processed_distributor_data = None
        # if 'db_update_status' not in st.session_state: st.session_state.db_update_status = None
            
        with st.expander("步骤 1: 上传员工工作日志", expanded=True):
            st.markdown("上传包含员工日常工作记录的Excel文件。系统将尝试处理和优化这些日志内容。")
            work_log_file = st.file_uploader("上传员工工作日志Excel文件", type=["xlsx"], key="work_log_uploader_sync")
            
            if work_log_file:
                try:
                    save_path = os.path.join("data", "uploaded_work_log.xlsx") # 使用固定或唯一文件名
                    with open(save_path, "wb") as f:
                        f.write(work_log_file.getbuffer())
                    st.session_state.work_log_path = save_path
                    st.success(f"已上传工作日志: {work_log_file.name} 并保存为 {save_path}")
                except Exception as e:
                    st.error(f"保存上传的工作日志文件时出错: {e}")
                    st.session_state.work_log_path = None
            else:
                st.session_state.work_log_path = None

        with st.expander("步骤 2: 处理并优化员工日志", expanded=False):
            st.markdown("系统将加载上传的日志，进行内容分析和优化。此过程可能需要一些时间，并可能调用LLM服务。")
            if not st.session_state.get("work_log_path"):
                st.warning("请先在步骤 1 上传员工工作日志文件。")
            else:
                if st.button("开始处理并优化员工日志", type="primary", key="start_optimization_sync"):
                    # 重置相关状态
                    st.session_state.optimized_records = None
                    st.session_state.processed_hospital_data = None # 假设优化过程也可能影响这些
                    st.session_state.processed_distributor_data = None
                    st.session_state.db_update_status = None
                    
                    if not client: # 检查OpenAI客户端
                        st.error("OpenAI API 未初始化。请在配置页面设置API密钥。")
                    else:
                        clear_process_logs()
                        with st.spinner("正在处理员工日志，这可能需要几分钟时间..."):
                            try:
                                log_file_path = st.session_state.work_log_path
                                add_process_log(f"开始处理员工日志: {log_file_path}")
                                
                                # 初始化 new_report 中的 OpenAI 客户端 (如果它有自己的初始化)
                                api_client_for_new_report = new_report.initialize_openai_client() if hasattr(new_report, 'initialize_openai_client') else client
                                if not api_client_for_new_report:
                                     add_process_log("警告: new_report.py 未能初始化其OpenAI客户端，将尝试使用app.py的客户端。")
                                     api_client_for_new_report = client # Fallback
                                
                                add_process_log("加载Excel数据...")
                                df = new_report.load_data_from_excel(log_file_path)
                                if df is None or df.empty:
                                    raise ValueError("从Excel加载数据失败或数据为空。")
                                add_process_log(f"成功从Excel加载 {len(df)} 条记录。")
                                
                                add_process_log("将DataFrame转换为记录格式...")
                                initial_records = new_report.convert_df_to_records(df)
                                if not initial_records:
                                     raise ValueError("转换DataFrame到记录失败或记录为空。")
                                add_process_log(f"成功转换 {len(initial_records)} 条记录。")
                                
                                add_process_log("开始评论优化和处理...")
                                # 假设 process_comments_and_optimize 是主要的优化函数
                                optimized_result_container = new_report.process_comments_and_optimize(
                                    {"records": initial_records}, 
                                    api_client_for_new_report
                                )
                                
                                if not (optimized_result_container and "records" in optimized_result_container and optimized_result_container["records"]):
                                    st.warning("评论优化过程未返回预期的记录数据，或者记录为空。请检查日志。")
                                    # 即使部分失败，也尝试保存日志
                                    st.session_state.optimized_records = [] #  设为空列表以避免后续错误
                                else:
                                    st.session_state.optimized_records = optimized_result_container["records"]
                                    st.success(f"日志处理完成！共获得 {len(st.session_state.optimized_records)} 条优化记录。")
                                add_process_log(f"评论优化处理完成，获得 {len(st.session_state.optimized_records)} 条记录。")
                                
                                # 提示：如果 new_report.py 的处理还涉及到 hospital_data 和 distributor_data 的更新，
                                # 那么应该在这里获取这些更新后的数据并存入 session_state，例如：
                                # st.session_state.processed_hospital_data = optimized_result_container.get("updated_hospitals")
                                # st.session_state.processed_distributor_data = optimized_result_container.get("updated_distributors")
                                
                            except Exception as e:
                                error_msg = f"处理员工日志时发生严重错误: {e}"
                                add_process_log(error_msg)
                                add_process_log(traceback.format_exc())
                                st.error(error_msg)
                                st.session_state.optimized_records = None # 清理以防不一致
        
        with st.expander("步骤 3: 下载优化结果与选择性上传至数据库", expanded=False):
            st.markdown("处理完成后，您可以在此下载优化后的Excel文件。如果配置了MongoDB，还可以选择将相关更新上传到数据库。")
            if st.session_state.get('optimized_records') is None:
                st.info("请先在步骤 2 中成功处理员工日志。")
            else:
                optimized_records_list = st.session_state.optimized_records
                st.markdown(f"**已处理 {len(optimized_records_list)} 条记录。**")

                if optimized_records_list: # 只有在有记录时才显示下载
                    try:
                        output_df = pd.DataFrame(optimized_records_list)
                        # 使用上传文件的原始名称（如果可用）创建输出文件名
                        base_filename = "optimized_log.xlsx"
                        if st.session_state.get("work_log_path") and work_log_file:
                            base_filename = "optimized_" + work_log_file.name
                        
                        output_excel_path = os.path.join("data", base_filename)
                        output_df.to_excel(output_excel_path, index=False)
                        
                        with open(output_excel_path, "rb") as file_bytes:
                            st.download_button(
                                label="下载优化后的Excel文件",
                                data=file_bytes,
                                file_name=base_filename,
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                key="download_optimized_log_sync"
                            )
                    except Exception as e:
                        st.error(f"创建或准备下载优化后的Excel文件时出错: {e}")
                        add_process_log(f"下载Excel准备失败: {e}")
                
                # --- 上传到 MongoDB 的确认 ---
                st.markdown("--- ")
                st.subheader("更新云数据库 (MongoDB)")
                
                if not st.session_state.use_mongodb:
                    st.warning('未启用 MongoDB。如需将处理结果（例如，更新的医院/经销商数据，如果适用）上传到数据库，请在"配置"页面启用并连接数据库。')
                else:
                    # 检查是否有待更新的数据 (这里需要更明确的逻辑，取决于 new_report.py 的输出)
                    # 假设 new_report.py 会在 optimized_result_container 中明确指出是否有DB更新
                    # processed_hospital_data = st.session_state.get('processed_hospital_data')
                    # processed_distributor_data = st.session_state.get('processed_distributor_data')
                    # has_data_to_update_db = bool(processed_hospital_data or processed_distributor_data)
                    # 简化：目前仅假设优化记录本身不直接写入特定集合，而是其分析结果可能导致其他集合的更新
                    # 这个按钮的实际作用需要根据 new_report.py 的功能来定
                    st.info("注意: 当前版本的'数据优化'主要生成可下载的Excel。如果处理流程还包括对医院/经销商等核心数据的标准化和更新，则此处的数据库更新才有意义。请确保 `new_report.py` 的处理逻辑支持这一点。")

                    mongodb_db_exists_check = hasattr(mongodb_connector, 'db') and mongodb_connector.db is not None
                    if not mongodb_db_exists_check:
                        st.error('MongoDB 未连接。请检查"配置"页面的数据库设置。')
                    else:
                        # 演示性按钮，实际功能需后端支持
                        confirm_db_update_optimization = st.checkbox("我理解此操作可能更新数据库（如果后端处理支持），并希望继续。", key="confirm_db_upload_optimization_sync")
                        
                        if st.button("执行数据库更新 (基于优化结果，如果适用)", key="execute_db_upload_optimization_sync", disabled=(not confirm_db_update_optimization)):
                            with st.spinner("正在尝试基于优化结果更新 MongoDB 数据库..."):
                                try:
                                    # mongo_client = mongodb_connector.client
                                    add_process_log("开始基于优化结果更新 MongoDB 数据库...")
                                    
                                    # TODO: 在这里实现实际的数据库更新逻辑。
                                    # 这需要 new_report.py 在处理后返回明确的数据库更新指令或数据。
                                    # 例如: updated_hospitals = st.session_state.get('processed_hospital_data')
                                    # if updated_hospitals: mongodb_connector.save_hospital_data(updated_hospitals)
                                    # ... 同样处理经销商数据 ...
                                    
                                    st.warning("数据库更新逻辑待实现。此按钮目前仅为占位符。需要 `new_report.py` 提供清晰的更新数据。")
                                    add_process_log("数据库更新（基于优化结果）的实际逻辑尚未完全实现。")
                                    # st.session_state.db_update_status = "success" # 仅作演示
                                    # st.success("数据库更新操作已发送 (实际效果取决于后端实现)。")
                                
                                except Exception as e:
                                     st.session_state.db_update_status = "error"
                                     error_msg = f"基于优化结果更新数据库时出错: {e}"
                                     add_process_log(error_msg)
                                     add_process_log(traceback.format_exc())
                                     st.error(error_msg)
                                     
            # 显示数据库更新状态 (如果实际执行了更新)
            if st.session_state.get('db_update_status') == "success":
                 st.success("数据库已成功更新 (基于之前的操作)。")
            elif st.session_state.get('db_update_status') == "error":
                 st.error("上次数据库更新尝试失败。请检查日志或重试。")

        # 显示处理日志
        st.markdown("--- ")
        with st.expander("优化处理与上传日志", expanded=False):
            st.code(get_process_logs_str(), language="text")
            
# ... (elif st.session_state.app_mode == "配置": 等后续部分保持不变)

elif st.session_state.app_mode == "配置":
    st.title("⚙️ 配置") # 翻译

    tab1, tab2 = st.tabs(["API 配置", "系统配置"])
    
    with tab1:
        st.subheader("OpenAI API 配置") # 翻译
        
        # 添加API密钥输入框
        api_key_input = st.text_input(
            "输入您的OpenAI API密钥",
            type="password",
            value=st.session_state.user_api_key,
            help="您输入的API密钥将覆盖环境变量和Secrets中的设置。留空则使用系统默认值。"
        )
        
        col1, col2 = st.columns([1, 3])
        with col1:
            if st.button("应用API密钥"):
                if api_key_input.strip():
                    # 保存用户输入的API密钥到session_state
                    st.session_state.user_api_key = api_key_input.strip()
                    # 使用新的API密钥初始化OpenAI客户端
                    success, error_msg = init_client_with_key(api_key_input.strip())
                    if success:
                        st.success("API密钥已应用并成功初始化OpenAI客户端")
                        save_config()  # 保存到配置文件
                    else:
                        st.error(f"使用新API密钥初始化客户端失败: {error_msg}")
                else:
                    st.warning("请输入API密钥")
        
        with col2:
            if st.button("测试API连接"):
                if client:
                    try:
                        # 简单测试API连接
                        response = client.chat.completions.create(
                            model="doubao-1-5-pro-32k-250115",
                            messages=[
                                {"role": "user", "content": "测试连接，请回复'连接正常'"}
                            ],
                            max_tokens=10
                        )
                        st.success(f"API连接正常: {response.choices[0].message.content}")
                    except Exception as e:
                        st.error(f"API测试失败: {e}")
                else:
                    st.error("OpenAI客户端未初始化，无法测试连接")
        
        st.markdown("---")
        
        if client:
            st.info(f"OpenAI API 客户端已初始化。API 地址: {client.base_url}") # 翻译
        else:
             # 警告信息已翻译
             st.warning(api_key_warning_message_cn, icon="⚠️")
    
    with tab2:
        st.subheader("数据库配置") # 翻译
        
        # MongoDB设置
        st.checkbox(
            "使用MongoDB数据库（不使用则使用本地文件）",
            value=st.session_state.use_mongodb,
            key="mongodb_checkbox",
            help="勾选此项将使用MongoDB数据库，否则使用本地JSON文件",
            on_change=lambda: setattr(st.session_state, 'use_mongodb', st.session_state.mongodb_checkbox)
        )
        
        if st.session_state.mongodb_checkbox:
            st.markdown("""
            **MongoDB连接信息:**
            - 连接字符串: `mongodb+srv://jhw:<db_password>@cluster0.j2eoyii.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0`
            - 数据库: `jhw`
            - 集合: `hospital` 和 `distributor`
            """)
            
            # 数据库密码输入
            db_password_input = st.text_input(
                "MongoDB密码",
                type="password",
                value=st.session_state.db_password,
                help="输入MongoDB数据库密码"
            )
            
            col1, col2, col3 = st.columns([1, 1, 2])
            with col1:
                if st.button("应用DB密码"):
                    if db_password_input.strip():
                        # 保存密码到session_state
                        st.session_state.db_password = db_password_input.strip()
                        # 尝试连接MongoDB
                        success, error = mongodb_connector.connect(db_password_input.strip())
                        if success:
                            st.success("MongoDB连接成功!")
                            save_config()  # 保存到配置文件
                        else:
                            st.error(f"MongoDB连接失败: {error}")
                    else:
                        st.warning("请输入MongoDB密码")
            
            with col2:
                if st.button("测试连接"):
                    if st.session_state.db_password.strip():
                        with st.spinner("正在测试MongoDB连接..."):
                            # 清空之前的日志
                            clear_process_logs()
                            add_process_log("开始测试MongoDB连接...")
                            
                            try:
                                # 修改后的测试逻辑：
                                # 1. 检查是否存在活动的全局连接并尝试ping
                                if hasattr(mongodb_connector, 'client') and mongodb_connector.client is not None:
                                    add_process_log("检测到活动全局连接，尝试ping...")
                                    mongodb_connector.client.admin.command('ping')
                                    add_process_log("现有连接ping测试成功!")
                                    st.success("MongoDB (现有连接) 测试成功!")
                                else:
                                    # 2. 如果没有活动全局连接，则进行一次新的本地连接测试
                                    add_process_log("无活动全局连接，尝试新的本地连接测试...")
                                    encoded_password = urllib.parse.quote_plus(st.session_state.db_password.strip())
                                    connection_string = f"mongodb+srv://jhw:{encoded_password}@cluster0.j2eoyii.mongodb.net/?retryWrites=true&w=majority"
                                    add_process_log(f"测试连接字符串: {connection_string.replace(encoded_password, '****')}")
                                    
                                    # 使用 MongoClient 进行本地测试，设置超时
                                    local_test_client = MongoClient(connection_string, serverSelectionTimeoutMS=5000)
                                    add_process_log("本地测试客户端尝试ping...")
                                    local_test_client.admin.command('ping')
                                    add_process_log("本地测试客户端ping成功!")
                                    
                                    # 可以进一步测试数据库和集合的访问
                                    local_db = local_test_client.get_database(MONGO_DB_NAME)
                                    add_process_log(f"本地测试成功获取数据库 '{MONGO_DB_NAME}'")
                                    h_count = local_db[MONGO_HOSPITALS_COLLECTION].count_documents({})
                                    d_count = local_db[MONGO_DISTRIBUTORS_COLLECTION].count_documents({})
                                    add_process_log(f"本地测试集合访问成功: Hospitals={h_count}, Distributors={d_count}")
                                    
                                    st.success("MongoDB (新测试连接) 测试成功!")
                                    local_test_client.close() # 关闭本地测试客户端
                                    add_process_log("本地测试客户端已关闭。")

                            except Exception as e:
                                error_msg = f"MongoDB连接测试失败: {e}"
                                add_process_log(error_msg)
                                add_process_log(traceback.format_exc())
                                st.error(error_msg)
                    else:
                        st.warning("请先输入MongoDB密码")
            
            # 显示处理日志
            with st.expander("连接测试日志", expanded=False):
                st.code(get_process_logs_str(), language="text")
        
        st.markdown("---")
        
        st.subheader("性能配置") # 翻译
        # 翻译标签和帮助文本
        # 使用 on_change 回调来直接更新 session_state，而不是依赖按钮
        def update_workers():
            st.session_state.max_workers = st.session_state.worker_slider_val
        def update_mt_default():
            st.session_state.use_multithreading_default = st.session_state.mt_default_check_val

        st.slider(
            "API 工作线程数 (用于多线程处理)", # 翻译
            min_value=1,
            max_value=10,
            value=st.session_state.get('max_workers', 5), # 从 state 获取当前值
            step=1,
            key="worker_slider_val", # 用不同的 key 来触发 on_change
            help="当启用多线程时，用于并行处理 LLM API 调用的线程数量。增加此值可以加快分析速度，但会消耗更多资源。", # 翻译
            on_change=update_workers
        )
        # 翻译标签和帮助文本
        st.checkbox(
            "默认启用多线程处理", # 翻译
            value=st.session_state.get('use_multithreading_default', True), # 从 state 获取当前值
            key="mt_default_check_val", # 用不同的 key
            help="勾选此项后，\"分析报告\"页面将默认启用多线程处理选项。", # 翻译
            on_change=update_mt_default
        )

        if st.button("保存配置到文件", key="save_config_button"): # 翻译
            save_config() # 内部消息已翻译


elif st.session_state.app_mode == "使用指南":
    st.title("📖 使用指南") # 翻译
    # 翻译整个 Markdown 内容
    st.markdown("""
        欢迎使用医疗销售日报分析器！

        **1. 分析报告:**
        * 前往"分析报告"功能区。
        * 将您的日报内容粘贴到文本区域。
        * (可选) 勾选"使用多线程处理"以加快分析速度(特别是当报告中提及多位医生/经销商时推荐)。
        * 点击"分析报告"按钮。
        * 分析结果将显示在右侧，包括关系状态评估和建议的后续行动。下方的折叠项中可以查看纯文本结果和详细的处理步骤。

        **2. 搜索实体:**
        * 前往"搜索实体"功能区。
        * 在搜索框中输入关键词(医生姓名、医院名称、经销商名称或公司)。
        * 点击"搜索"按钮。
        * 系统将在已加载的历史数据文件中查找并显示匹配的记录。

        **3. 数据管理:**
        * 前往"数据管理"功能区。
        * **上传文件:** 您可以上传自己的 JSON 格式历史数据文件(医院拜访记录或经销商沟通记录)。请确保文件格式符合预期(可参考示例数据)。上传新文件后，它将成为当前分析使用的数据源。
        * **查看数据:** 点击"刷新医院数据"或"刷新经销商数据"按钮可以检查当前加载的数据的内容。
        * **示例数据:** 点击"加载示例数据"按钮将加载内置的演示数据。**注意:** 这会覆盖当前数据。如果启用了MongoDB，示例数据也会被保存到数据库中。

        **4. 数据同步:**
        * 前往"数据同步"功能区。
        * **主数据同步:** 可以将主数据（医院、医生、科室、经销商等）同步到MongoDB数据库，确保数据一致性。
        * **名称标准化:** 上传包含医院标准名称和位置的Excel文件，对医院名称进行标准化处理。
        * **数据优化:** 上传员工工作日志Excel文件，系统会处理领导评论并生成优化建议，提高日报质量。
        * 注意: 部分功能需要连接MongoDB数据库和配置API密钥才能使用。

        **5. 配置:**
        * 前往"配置"功能区。
        * **API 配置:** 显示 OpenAI API 密钥的状态。您可以直接在界面中输入您的API密钥，也可以通过 `.env` 文件或 Streamlit Secrets 来配置API密钥，否则分析功能无法使用。
        * **系统配置:**
          * **数据库设置:** 您可以选择使用MongoDB数据库或本地文件存储数据。如果使用MongoDB，需要输入数据库密码并点击"应用DB密码"进行连接。
          * **性能配置:** 调整用于多线程处理的工作线程数，以及设置是否默认启用多线程。修改后会自动更新设置，点击"保存配置到文件"可将当前设置持久化保存。

        **数据格式说明:**
        * 历史数据必须是 JSON 格式。
        * 医院数据文件需要包含一个顶层键 `"hospitals"`，其值为一个包含多个医院对象的列表。每个医院对象可以包含一个 `"历史记录"` 键，其值为拜访记录列表。
        * 经销商数据文件需要包含一个顶层键 `"distributors"`，其值为一个包含多个经销商对象的列表。每个经销商对象可以包含一个 `"沟通记录"` 键，其值为沟通记录列表。
        * 请参考示例数据的结构以获取更详细的格式信息。
        
        **MongoDB数据库:**
        * 应用现在支持使用MongoDB数据库存储数据，这样可以在不同设备上访问相同的数据。
        * 要启用MongoDB：
          1. 在"配置"页面中勾选"使用MongoDB数据库"选项
          2. 输入MongoDB密码并点击"应用DB密码"
          3. 测试连接以确保设置正确
        * 当启用MongoDB后，所有数据操作（上传、查询等）将使用数据库而非本地文件。
        * 系统会自动将数据同步到MongoDB数据库，您也可以随时关闭MongoDB使用本地文件。
    """)

# --- 页脚 ---
st.markdown("---")
st.caption("医疗销售报告分析器 v1.3 (MongoDB + 数据同步版本)") # 更新版本号