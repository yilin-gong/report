# analyzer.py
"""
This module contains all the analysis functionality imported from the original script.
Just move all your functions here, keeping their original implementation.
"""

import json
import datetime
from typing import Dict, List, Any, Optional

# You'll need to initialize your LLM client here, for example:
# from openai import OpenAI
# client = OpenAI(api_key="your-api-key")
# Or use your existing client setup

# All your original functions:
def extract_entities_with_llm(daily_report_text: str) -> Dict[str, Any]:
    # Your implementation here
    # For testing without API access, we can provide a mock implementation:
    return {
        "hospitals": ["七院", "仁济医院"],
        "doctors": [
            {"name": "叶亮", "role": "胸外科主任"},
            {"name": "杨珏", "role": "肝胆外科主任"},
            {"name": "张遂亮", "role": "血管外科医生"},
            {"name": "赵滨", "role": "普外科大主任"}
        ],
        "departments": ["血管外科", "甲乳外科", "普外科"],
        "products": ["腔镜手术设备"],
        "distributors": [
            {"name": "邱春", "company": "经销商"},
            {"name": "陆总", "company": "经销商"}
        ]
    }

def evaluate_with_llm(text: str, context: Dict[str, str]) -> Dict[str, str]:
    # Your implementation here
    # Mock implementation for testing:
    if not text:
        return {
            "status": "未知", 
            "analysis": f"没有与{context.get('name', '未知')}的历史沟通记录，无法评估关系"
        }
    
    return {
        "status": "中性",
        "analysis": f"与{context.get('name', '未知')}的关系处于初期阶段，沟通记录显示对方态度中性，尚未表现出明确的合作意向，需要进一步跟进和了解需求。"
    }

def generate_suggested_actions(evaluation_result: Dict[str, str], entity_info: Dict[str, str] = None) -> List[str]:
    # Your implementation here
    # Mock implementation for testing:
    if entity_info is None:
        entity_info = {}
    
    status = evaluation_result.get('status', '未知')
    
    if status == "新接触":
        return [
            "安排正式拜访，进行自我介绍并了解基本需求",
            "准备产品资料，重点突出适合该角色需求的产品特点",
            "收集竞品使用情况，了解对方对现有产品的满意度",
            "邀请参加相关学术活动，增进关系"
        ]
    else:
        return [
            "定期电话回访，了解产品使用情况",
            "提供新产品信息更新，保持沟通热度",
            "针对之前沟通中发现的问题提供解决方案",
            "邀请参加用户体验活动，收集改进建议"
        ]

def search_doctors_in_hospital_data(hospital_data: Dict[str, Any], doctor_names: List[str]) -> Dict[str, List[Dict[str, str]]]:
    # Your original implementation
    results = {}
    
    for doctor_name in doctor_names:
        all_records = []
        
        for hospital in hospital_data.get("hospitals", []):
            for record in hospital.get("历史记录", []):
                if (doctor_name in record.get("医生姓名", "") or 
                    record.get("医生姓名", "") in doctor_name):
                    
                    all_records.append({
                        "医院": hospital.get("医院名称", ""),
                        "科室": record.get("科室", ""),
                        "拜访日期": record.get("拜访日期", ""),
                        "拜访员工": record.get("拜访员工", ""),
                        "沟通内容": record.get("沟通内容", ""),
                        "后续行动": record.get("后续行动", "")
                    })
        
        # 按拜访日期排序，最近的排在前面
        all_records.sort(key=lambda x: x.get("拜访日期", ""), reverse=True)
        
        # 只保留最近的一条记录
        if all_records:
            results[doctor_name] = [all_records[0]]
    
    # 移除没有记录的医生
    return {k: v for k, v in results.items() if v}

def search_distributors_in_data(distributor_data: Dict[str, Any], distributor_names: List[str]) -> Dict[str, List[Dict[str, str]]]:
    # Your original implementation
    results = {}
    
    for distributor_name in distributor_names:
        all_records = []
        
        for distributor in distributor_data.get("distributors", []):
            for record in distributor.get("沟通记录", []):
                if (distributor_name in record.get("联系人", "") or 
                    record.get("联系人", "") in distributor_name):
                    
                    all_records.append({
                        "经销商名称": distributor.get("经销商名称", ""),
                        "沟通日期": record.get("沟通日期", ""),
                        "沟通员工": record.get("沟通员工", ""),
                        "沟通内容": record.get("沟通内容", ""),
                        "后续计划": record.get("后续计划", "")
                    })
        
        # 按沟通日期排序，最近的排在前面
        all_records.sort(key=lambda x: x.get("沟通日期", ""), reverse=True)
        
        # 只保留最近的一条记录
        if all_records:
            results[distributor_name] = [all_records[0]]
    
    # 移除没有记录的经销商
    return {k: v for k, v in results.items() if v}

def load_json_data(file_path: str) -> Dict[str, Any]:
    # Your original implementation
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"加载文件 {file_path} 时出错: {e}")
        return {}

def format_date(date_str: str) -> str:
    """格式化日期字符串"""
    try:
        date = datetime.datetime.strptime(date_str, "%Y-%m-%d")
        return date.strftime("%Y年%m月%d日")
    except:
        return date_str

def analyze_daily_report(daily_report_text: str, hospital_data_file: str, distributor_data_file: str) -> Dict[str, Any]:
    # Your original implementation
    # 提取日报中的实体
    entities = extract_entities_with_llm(daily_report_text)
    
    # 读取数据文件
    hospital_data = load_json_data(hospital_data_file)
    distributor_data = load_json_data(distributor_data_file)
    
    if not hospital_data:
        hospital_data = {"hospitals": []}
    if not distributor_data:
        distributor_data = {"distributors": []}
    
    # 获取医生姓名列表
    doctor_names = [doctor["name"] for doctor in entities.get("doctors", [])]
    
    # 获取经销商姓名列表
    distributor_names = [dist["name"] for dist in entities.get("distributors", [])]
    
    # 在历史数据中搜索医生的最近记录
    doctor_records = search_doctors_in_hospital_data(hospital_data, doctor_names)
    
    # 在历史数据中搜索经销商的最近记录
    distributor_records = search_distributors_in_data(distributor_data, distributor_names)
    
    # 评估医生关系
    doctor_evaluations = {}
    for doctor in entities.get("doctors", []):
        doctor_name = doctor["name"]
        records = doctor_records.get(doctor_name, [])
        
        if records:
            # 只有一条最近的记录
            latest_record = records[0]
            evaluation = evaluate_with_llm(
                latest_record.get("沟通内容", ""),
                {"name": doctor_name, "role": doctor["role"]}
            )
            doctor_evaluations[doctor_name] = {
                "records": records,
                "evaluation": evaluation,
                "role": doctor["role"]
            }
        else:
            doctor_evaluations[doctor_name] = {
                "records": [],
                "evaluation": {
                    "status": "新接触",
                    "analysis": f"{doctor_name}({doctor['role']})是新接触的联系人，尚无历史沟通记录。建议制定初次接触策略，收集其基本信息、临床需求和决策风格，为建立长期合作关系奠定基础。"
                },
                "role": doctor["role"]
            }
    
    # 评估经销商关系
    distributor_evaluations = {}
    for distributor in entities.get("distributors", []):
        distributor_name = distributor["name"]
        records = distributor_records.get(distributor_name, [])
        
        if records:
            # 只有一条最近的记录
            latest_record = records[0]
            evaluation = evaluate_with_llm(
                latest_record.get("沟通内容", ""),
                {"name": distributor_name, "role": "经销商"}
            )
            distributor_evaluations[distributor_name] = {
                "records": records,
                "evaluation": evaluation,
                "company": distributor.get("company", "未知")
            }
        else:
            distributor_evaluations[distributor_name] = {
                "records": [],
                "evaluation": {
                    "status": "新接触",
                    "analysis": f"{distributor_name}是新接触的经销商，尚无历史沟通记录。建议全面了解其业务范围、渠道资源和市场影响力，评估合作潜力，设计阶段性合作目标。"
                },
                "company": distributor.get("company", "未知")
            }
    
    # 返回分析结果
    return {
        "entities": entities,
        "doctor_evaluations": doctor_evaluations,
        "distributor_evaluations": distributor_evaluations
    }

def format_output(results: Dict[str, Any]) -> str:
    """
    格式化分析结果为可读文本
    
    参数:
    results (dict): 分析结果
    
    返回:
    str: 格式化后的文本
    """
    output = []
    
    # 提取日报作者姓名
    report_date = datetime.datetime.now().strftime("%Y年%m月%d日")
    
    # 标题
    output.append(f"=== 日报分析结果 ({report_date}) ===\n")
    
    # 医生信息
    output.append("【医生/主任信息】")
    
    for name, data in results["doctor_evaluations"].items():
        output.append(f"\n医生：{name}")
        
        if data["records"]:
            record = data["records"][0]  # 取第一条记录
            output.append(f"  所属医院：{record['医院']}")
            output.append(f"  所属科室：{record['科室'] or '未知'}")
            output.append(f"  最近拜访：{format_date(record['拜访日期'])} (by {record['拜访员工']})")
            output.append(f"  沟通内容：{record['沟通内容']}")
        else:
            output.append("  角色：" + data.get("role", "未知"))
            output.append("  无历史沟通记录")
        
        output.append(f"  关系状态：{data['evaluation']['status']}")
        output.append(f"  关系分析：{data['evaluation']['analysis']}")
        
        # 生成并显示建议行动
        suggestions = generate_suggested_actions(
            data["evaluation"], 
            {"name": name, "role": data.get("role", "")}
        )
        output.append("  建议行动：")
        for i, suggestion in enumerate(suggestions, 1):
            output.append(f"    {i}. {suggestion}")
    
    # 经销商信息
    output.append("\n【经销商信息】")
    
    for name, data in results["distributor_evaluations"].items():
        output.append(f"\n经销商：{name}")
        
        if data["records"]:
            record = data["records"][0]  # 取第一条记录
            output.append(f"  所属公司：{record['经销商名称']}")
            output.append(f"  最近沟通：{format_date(record['沟通日期'])} (by {record['沟通员工']})")
            output.append(f"  沟通内容：{record['沟通内容']}")
        else:
            output.append(f"  所属公司：{data.get('company', '未知')}")
            output.append("  无历史沟通记录")
        
        output.append(f"  关系状态：{data['evaluation']['status']}")
        output.append(f"  关系分析：{data['evaluation']['analysis']}")
        
        # 生成并显示建议行动
        suggestions = generate_suggested_actions(
            data["evaluation"], 
            {"name": name, "role": "经销商"}
        )
        output.append("  建议行动：")
        for i, suggestion in enumerate(suggestions, 1):
            output.append(f"    {i}. {suggestion}")
    
    # 医院信息
    output.append("\n【医院信息】")
    for hospital in results["entities"]["hospitals"]:
        output.append(f"- {hospital}")
    
    # 科室信息
    if results["entities"]["departments"]:
        output.append("\n【科室信息】")
        for department in results["entities"]["departments"]:
            output.append(f"- {department}")
    
    return "\n".join(output)
