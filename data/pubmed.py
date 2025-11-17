# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : pubmed.py
# Time       ：14/3/2025 9:42 am
# Author     ：Chuang Zhao
# version    ：python 
# Description： download pubmed information for each keyword
"""
import requests
import xml.etree.ElementTree as ET
import time
import json
import os
from datetime import datetime


def batch_search_and_save(query_list, api_key, email, output_dir="paper_data", max_results_per_query=10,
                          sleep_time=0.4):
    """
    批量查询NCBI PubMed论文并高效存储元数据

    参数:
    query_list (list): 查询关键词列表
    api_key (str): NCBI API密钥
    email (str): 用户邮箱地址
    output_dir (str): 输出目录
    max_results_per_query (int): 每个查询返回的最大结果数
    sleep_time (float): 请求间隔时间(秒)

    返回:
    str: 结果文件路径
    """
    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 生成输出文件名（使用时间戳避免覆盖）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"pubmed_results_{timestamp}.json")

    # 初始化结果字典
    all_results = {}

    # 统计计数器
    total_papers = 0

    # 处理每个查询
    for query_index, query in enumerate(query_list):
        print(f"处理查询 {query_index + 1}/{len(query_list)}: '{query}'")

        try:
            # 获取当前查询的论文
            papers = search_and_fetch_papers(query, api_key, email, max_results_per_query)
            total_papers += len(papers)

            # 将结果添加到字典中
            all_results[query] = papers

            # 定期保存中间结果
            if (query_index + 1) % 5 == 0 or query_index == len(query_list) - 1:
                # 存储为JSON文件
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(all_results, f, ensure_ascii=False, indent=2)
                print(f"已保存中间结果到文件: {output_file}")

            # 添加延迟以遵守NCBI的访问限制
            time.sleep(sleep_time)

        except Exception as e:
            print(f"处理查询'{query}'时出错: {str(e)}")
            # 遇到错误也保存已有结果
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(all_results, f, ensure_ascii=False, indent=2)
            print(f"已保存部分结果到文件: {output_file}")

    print(f"批量查询完成。共处理 {len(query_list)} 个查询，获取 {total_papers} 篇论文信息。")
    return output_file


def search_pubmed(query, api_key, email, retmax=10):
    """
    使用NCBI E-utilities搜索PubMed

    参数:
    query (str): 搜索关键词
    api_key (str): NCBI API密钥
    email (str): 用户邮箱地址
    retmax (int): 返回结果的最大数量

    返回:
    list: 包含论文ID的列表，以及WebEnv和QueryKey用于后续请求
    """
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"

    # 步骤1: 使用esearch获取匹配查询的论文ID
    search_url = f"{base_url}esearch.fcgi"
    search_params = {
        "db": "pubmed",
        "term": query,
        "retmax": retmax,
        "retmode": "xml",
        "usehistory": "y",
        "api_key": api_key,
        "email": email
    }

    search_response = requests.get(search_url, params=search_params)
    search_tree = ET.fromstring(search_response.content)

    # 提取论文ID
    id_list = [id_elem.text for id_elem in search_tree.findall(".//Id")]
    id_list = id_list[:retmax]  # 限制结果数量

    # 提取WebEnv和QueryKey用于后续efetch请求
    webenv = search_tree.find(".//WebEnv").text
    query_key = search_tree.find(".//QueryKey").text

    return id_list, webenv, query_key


def fetch_article_details(api_key, email, id_list=None, webenv=None, query_key=None):
    """
    使用NCBI E-utilities获取论文的详细信息

    参数:
    api_key (str): NCBI API密钥
    email (str): 用户邮箱地址
    id_list (list): 论文ID列表
    webenv (str): WebEnv值，用于批量获取
    query_key (str): QueryKey值，用于批量获取

    返回:
    list: 包含论文详细信息的字典列表
    """
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
    fetch_url = f"{base_url}efetch.fcgi"

    # 使用WebEnv和QueryKey参数
    if webenv and query_key:
        fetch_params = {
            "db": "pubmed",
            "WebEnv": webenv,
            "query_key": query_key,
            "retmode": "xml",
            "rettype": "abstract",
            "api_key": api_key,
            "email": email
        }
    # 使用ID列表
    elif id_list:
        id_string = ",".join(id_list)
        fetch_params = {
            "db": "pubmed",
            "id": id_string,
            "retmode": "xml",
            "rettype": "abstract",
            "api_key": api_key,
            "email": email
        }
    else:
        return []

    fetch_response = requests.get(fetch_url, params=fetch_params)
    fetch_tree = ET.fromstring(fetch_response.content)

    # 解析XML获取论文信息 (优化存储空间)
    articles = []

    for article_elem in fetch_tree.findall(".//PubmedArticle"):
        # 创建紧凑的文章信息字典
        article_data = {}

        # 获取PMID (必需项)
        pmid_elem = article_elem.find(".//PMID")
        if pmid_elem is not None:
            article_data["id"] = pmid_elem.text  # 使用更短的键名
        else:
            continue  # 如果没有PMID，跳过此文章

        # 获取标题 (必需项)
        title_elem = article_elem.find(".//ArticleTitle")
        if title_elem is not None and title_elem.text:
            article_data["t"] = title_elem.text  # 使用缩写键名

        # 获取摘要 (必需项)
        abstract_elems = article_elem.findall(".//AbstractText")
        if abstract_elems:
            abstract_parts = []
            labeled_abstract = {}
            has_labels = False

            for abstract_elem in abstract_elems:
                label = abstract_elem.get("Label")
                text = abstract_elem.text
                if text:
                    if label:
                        has_labels = True
                        labeled_abstract[label] = text
                    else:
                        abstract_parts.append(text)

            if has_labels:
                article_data["a"] = labeled_abstract  # 结构化摘要
            elif abstract_parts:
                article_data["a"] = " ".join(abstract_parts)  # 普通摘要

        # 获取作者 (可选)
        authors = []
        author_elems = article_elem.findall(".//Author")
        for author_elem in author_elems:
            last_name = author_elem.find(".//LastName")
            fore_name = author_elem.find(".//ForeName")
            if last_name is not None:
                last = last_name.text
                fore = fore_name.text if fore_name is not None else ""
                if fore:
                    # 只保存姓氏和名字首字母
                    fore_initials = ''.join([n[0] for n in fore.split() if n])
                    authors.append(f"{last} {fore_initials}")
                else:
                    authors.append(last)

        if authors:
            article_data["au"] = authors[:3]  # 限制作者数量，只保存前三位

        # 获取期刊信息和年份 (可选，合并为一个字段)
        journal_elem = article_elem.find(".//Journal/Title")
        year_elem = article_elem.find(".//PubDate/Year")
        if journal_elem is not None or year_elem is not None:
            journal_info = []
            if journal_elem is not None and journal_elem.text:
                # 只保留期刊名称的关键词
                journal_words = journal_elem.text.split()
                if len(journal_words) > 2:
                    shortened_name = ' '.join(word for word in journal_words if len(word) > 3)[:30]
                    journal_info.append(shortened_name)
                else:
                    journal_info.append(journal_elem.text)
            if year_elem is not None and year_elem.text:
                journal_info.append(year_elem.text)

            if journal_info:
                article_data["j"] = ', '.join(journal_info)

        articles.append(article_data)

    return articles


def search_and_fetch_papers(query, api_key, email, max_results=10):
    """
    搜索并获取论文详细信息

    参数:
    query (str): 搜索关键词
    api_key (str): NCBI API密钥
    email (str): 用户邮箱地址
    max_results (int): 返回结果的最大数量

    返回:
    list: 包含论文详细信息的字典列表
    """
    # 搜索论文
    id_list, webenv, query_key = search_pubmed(query, api_key, email, max_results)

    # 添加延迟以遵守NCBI的访问限制(不超过每秒3个请求)
    time.sleep(0.4)

    # 获取论文详细信息
    articles = fetch_article_details(api_key, email, id_list=id_list, webenv=None, query_key=query_key)

    return articles


def load_results(file_path):
    """
    加载保存的查询结果

    参数:
    file_path (str): JSON文件路径

    返回:
    dict: 加载的结果字典
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        results = json.load(f)
    return results


def get_decompressed_results(file_path, query=None):
    """
    加载保存的查询结果并将缩写字段恢复为完整名称

    参数:
    file_path (str): JSON文件路径
    query (str): 可选，指定只解压某个查询的结果

    返回:
    dict 或 list: 解压缩后的结果
    """
    field_map = {
        'id': 'pmid',
        't': 'title',
        'a': 'abstract',
        'au': 'authors',
        'j': 'journal_info'
    }

    results = load_results(file_path)

    if query:
        if query in results:
            papers = results[query]
            decompressed_papers = []
            for paper in papers:
                decompressed = {}
                for k, v in paper.items():
                    full_key = field_map.get(k, k)
                    decompressed[full_key] = v
                decompressed_papers.append(decompressed)
            return decompressed_papers
        else:
            return []
    else:
        decompressed_results = {}
        for q, papers in results.items():
            decompressed_papers = []
            for paper in papers:
                decompressed = {}
                for k, v in paper.items():
                    full_key = field_map.get(k, k)
                    decompressed[full_key] = v
                decompressed_papers.append(decompressed)
            decompressed_results[q] = decompressed_papers
        return decompressed_results


# 示例使用
if __name__ == "__main__":
    # 查询列表示例
    queries = [
        "cancer immunotherapy",
        "CRISPR gene editing",
        "machine learning medicine"
    ]

    email = "czhaobo@connect.ust.hk"  # 替换为您的搜索关键词
    api_key = "9c942b736907d1132c1f589f99bcc30efb08"  # 替换为您的API密钥

    # 执行批量查询并保存
    result_file = batch_search_and_save(queries, api_key, email, max_results_per_query=5)

    # 加载结果 (全部或特定查询)
    all_results = load_results(result_file)
    specific_results = get_decompressed_results(result_file, "CRISPR gene editing")

    print(f"共获取 {sum(len(papers) for papers in all_results.values())} 篇论文信息")

    # 显示特定查询的结果
    if specific_results:
        print("\n特定查询结果示例:")
        for i, paper in enumerate(specific_results[:2], 1):
            print(f"\n--- 论文 {i} ---")
            for k, v in paper.items():
                if k == "authors":
                    print(f"{k}: {', '.join(v)}")
                else:
                    print(f"{k}: {v}")