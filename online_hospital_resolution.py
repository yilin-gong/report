import os
import logging
from typing import List, Sequence, Optional, Tuple, Dict, Any
import re
try:
    from tavily import TavilyClient  # pip install tavily-python
except ImportError:  # 懒加载，避免离线环境报错
    TavilyClient = None  # type: ignore

# 从项目已有模块导入工具函数
from new_report import (
    clean_entity_name,
    clean_location_term,
    _call_llm_api,
    add_to_master_data,
    lookup_in_master_data,
)

TAVILY_API_KEY: Optional[str] = "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"  # 或者写到配置文件
def _build_query(raw_name: str, location_context: Sequence[str]) -> str:
    """拼接 Tavily 查询串；若简称不含医院关键词自动补"医院"。"""
    loc_part = " ".join(clean_location_term(l) for l in location_context if l).strip()
    kw_present = any(k in raw_name for k in ("医院", "卫生院", "医科", "附属", "中心"))
    q_name = raw_name if kw_present else f"{raw_name} 医院"
    return f"{loc_part} {q_name}".strip()


def _filter_titles(
    titles: List[str],
    keep_kw: Tuple[str, ...] = (
        "附属医院",
        "医院",
        "附属",
        "中心医院",
        "医学中心",
        "卫生院",
        "医科大学",
    ),
    bad_kw: Tuple[str, ...] = (
        "招标",
        "招聘",
        "器械",
        "药业",
        "有限公司",
    ),
) -> List[str]:
    """过滤标题：去掉明显无关或占位项（如 'query'）。"""
    cleaned = [t for t in titles if t and t.lower() != "query"]
    keep = [
        t for t in cleaned
        if any(k in t for k in keep_kw) and not any(b in t for b in bad_kw)
    ]
    return (keep or cleaned)[:5]


_TITLE_REGEX = re.compile(
    r"([\u4e00-\u9fa5]{2,}(?:附属医院|中心医院|中医院|医院|卫生院|医学中心))"
)


def _extract_official(titles: List[str], query: str, api_client) -> str:
    """用 LLM 从标题挑全称；LLM 失败则正则/首标题兜底。"""
    prompt = (
        "下面是互联网上搜索 " + query + " 得到的网页标题：\n" +
        "\n".join(f"{i+1}. {t}" for i, t in enumerate(titles)) +
        "\n\n请根据标题内容，判断医院名称，只输出完整医院名称，不要编号、不要额外文字。如果无法判断，请输出原名称。"
    )
    llm_raw = _call_llm_api(api_client, [{"role": "user", "content": prompt}], temperature=0.1) or ""
    llm_raw = clean_entity_name(llm_raw)
    # 去编号/前缀
    llm_raw = re.sub(r"^[0-9．.、\s]+", "", llm_raw)
    # 正则截取
    m = _TITLE_REGEX.search(llm_raw)
    if m:
        return m.group(1)
    # LLM 结果无用 → 尝试在标题里跑正则
    for t in titles:
        m = _TITLE_REGEX.search(t)
        if m:
            return m.group(1)
    # 仍失败 → 直接返回首标题清洗后正则截取或原名
    return clean_entity_name(titles[0]) if titles else ""


# ------------------------------------------------------------------
# 主函数
# ------------------------------------------------------------------

def resolve_hospital_via_web(
    *,
    raw_name: str,
    location_context: Sequence[str],
    api_client,
    top_k: int = 5,
    return_raw_results: bool = False,
) -> Tuple[str, str, float, Optional[List[Dict[str, Any]]]]:
    """使用 Tavily + LLM 把简称解析为官方全称。"""

    # 0) cache
    cached = lookup_in_master_data("hospitals", raw_name)
    if cached:
        full = cached if isinstance(cached, str) else cached.get("全称", "")
        region = cached.get("地区", "") if isinstance(cached, dict) else ""
        return full, region, 95.0, None

    # 1) Tavily availability
    if TavilyClient is None or not TAVILY_API_KEY:
        logging.error("[online_hospital] Tavily 未配置，跳过在线解析…")
        return "", "", 0.0, None

    # 2) search
    query = _build_query(raw_name, location_context)
    try:
        tav = TavilyClient(api_key=TAVILY_API_KEY)
        search_response = tav.search(query=query, max_results=top_k)
        titles: List[str] = []
        
        # 处理新的API响应格式
        if isinstance(search_response, dict) and "results" in search_response:
            # 新API格式：结果在"results"键中
            search_results = search_response["results"]
            for r in search_results:
                if isinstance(r, dict) and "title" in r:
                    t = str(r["title"]).strip()
                    if t:
                        titles.append(t)
        else:
            # 兼容旧格式（直接返回结果列表）
            for r in search_response:
                if isinstance(r, dict):
                    t = str(r.get("title", "")).strip()
                    if t:
                        titles.append(t)
                else:
                    # 避免 Tavily 返回占位字符串 "query"
                    t = str(r).strip()
                    if t and t.lower() != "query":
                        titles.append(t)
                        
        raw_results = search_response if return_raw_results else None
    except Exception as e:
        logging.warning("[online_hospital] Tavily 查询失败: %s", e)
        return "", "", 0.0, None

    titles = _filter_titles(titles)
    if not titles:
        logging.info("[online_hospital] query=%s → titles=[]", query)
        return "", "", 0.0, None

    # 3) extract official name only from titles
    full_name = _extract_official(titles, query, api_client)
    if not full_name:
        logging.info("[online_hospital] 解析失败: %s", raw_name)
        return "", "", 0.0, raw_results

    # 4) region 使用 location_context 首个
    region = clean_location_term(location_context[0]) if location_context else ""

    # 5) cache
    add_to_master_data("hospitals", raw_name, full_name, value_type="标准名称", region=region)
    logging.info("[online_hospital] %s → %s | region=%s | score=90.0", raw_name, full_name, region)
    return full_name, region, 90.0, raw_results
