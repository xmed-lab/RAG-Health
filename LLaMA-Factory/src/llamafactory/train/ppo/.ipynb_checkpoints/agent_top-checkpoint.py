# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : agent_top.py
# Time       ：20/3/2025 9:17 am
# Author     ：Chuang Zhao
# version    ：python 
# Description：# 需要返回summary, negative, meta_path, chosen entity
"""
import re
# import random
from typing import List, Dict, Any, Tuple
# from langchain import LLMChain, PromptTemplate
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import BaseLLM
from collections import deque
from .agent_low import AgentLow
from .instructions_template import prompt_templates


# AgentTop类， 直接执行
class AgentTop:
    def __init__(self, metapaths, agent_low: AgentLow, ratio: float = 0.3, max_iterations: int = 3, confidence: float = 0.7):
        self.metapaths = metapaths  # metapaths是一个字典列表，包含name和description
        self.agent_low = agent_low
        self.ratio = ratio
        self.max_iterations = max_iterations
        self.confidence_score = confidence

    def clean_subquery(self, subqueries):
        # 清洗奇怪符号和无效字符
        cleaned_subqueries = []
        for sq in subqueries:
            sq = re.sub(r'[^\x00-\x7F]+', ' ', sq)  # 保留ASCII范围内字符
            sq = re.sub(r'\s+', ' ', sq).strip()

            # 保留字母、数字、空格和常见标点（根据需要调整）
            # 正则表达式只保留 a-zA-Z0-9 空格和基本标点
            cleaned = re.sub(r'[^\w\s\?\.!,-]', '', sq)
            # 去除连续空格和标点
            cleaned = re.sub(r'\s+', ' ', cleaned).strip()
            cleaned = re.sub(r'[^\w\s]$', '', cleaned)  # 去除结尾的特殊标点
            if cleaned and len(cleaned)>5:  # 只保留非空结果
                if '?' not in cleaned:
                    cleaned += '?'  # 补充问号
                cleaned_subqueries.append(cleaned)


        subqueries = list(set(cleaned_subqueries))  # 去重
        return subqueries

    def rewrite_query(self, query: str, reason_history: str, llm_function=None) -> List[str]:
        # 提示模板：明确要求生成 3-5 个子查询
        messages = [[
            # {'role': 'system', 'content': "Please answer in English."},
            {"role": "user", "content": prompt_templates['rewrite'].format(query=query)} # , reason_history=reason_history
        ]] # 后续可以尝试batch处理, 这里add for 循环即可。注意要同步修改llm function
        # messages = [prompt_templates['rewrite'].format(query=query)]
        result = llm_function(messages)

        # 处理 LLM 输出格式
        # subqueries = result.split("\n")
        # subqueries = [sq.strip() for sq in subqueries if sq.strip()]  # 过滤空字符串
        # 先按编号分割（如1. 2.），再按换行分割，确保覆盖多种格式
        split_by_number = re.split(r'\b(1|2|3|4|5)\s*[.\s]\s*', result)  # 分割编号与内容
        subqueries = []
        for part in split_by_number:
            # 过滤数字和空值，再按换行分割剩余内容
            if not part.strip() or part.strip().isdigit():
                continue
            subqueries.extend([line.strip() for line in part.split('\n') if line.strip()])

        subqueries = self.clean_subquery(subqueries)  # 清洗子查询

        # 如果生成的子查询数量不足，则通过规则生成额外的子查询
        if len(subqueries) < 3:
            # print("Warning: Generated subqueries are insufficient. Falling back to rule-based generation.")
            subqueries.extend(self._generate_fallback_subqueries(query))

        # 限制最多返回 5 个子查询
        return subqueries[:3]


    def rewrite_query_follow(self, query: str, reason_history: str, llm_function=None) -> List[str]:
        # 提示模板：明确要求生成 follow 个子查询
        messages = [[
#             {'role': 'system', 'content': "Please answer in English."},
            {'role': 'user', 'content': prompt_templates['follow'].format(query=query,reason_history=reason_history)} #
        ]]
        result = llm_function(messages)

        # 处理 LLM 输出格式
        subqueries = result.strip()
        return subqueries

    def _generate_fallback_subqueries(self, query: str) -> List[str]:
        """规则生成备用子查询"""
        fallback_subqueries = []

        # 示例规则：基于关键词生成子查询
        keywords = ["what", "why", "how", "when", "where"]
        for keyword in keywords:
            if keyword in query.lower():
                fallback_subqueries.append(f"{keyword.capitalize()} {query}")

        # 如果仍然不足，则直接拆分查询
        if len(fallback_subqueries) < 3:
            fallback_subqueries.extend(query.split(" and "))
            fallback_subqueries.extend(query.split(" or "))

            # 去重并过滤空字符串
            fallback_subqueries = list(set(fallback_subqueries))
            fallback_subqueries = [sq.strip() for sq in fallback_subqueries if sq.strip()]

        return fallback_subqueries

    def decide_action(self, subquery: str, reason_history: str, confidence_threshold: float = 0.7, llm_function=None) -> str:
        messages = [[
#             {'role': 'system', 'content': "Please answer in English."},

            {'role': 'user', 'content': prompt_templates['decide'].format(subquery=subquery, reason_history=reason_history)}
        ]]

        decision_text = llm_function(messages)

        # print("BBBBBBBB", decision_text)

        # 提取置信度评分
        confidence_score = 1.0
        # if "confidence score:" in decision_text:
        #     try:
        #         confidence_part = decision_text.split("confidence score:")[-1]
        #         confidence_score = float(confidence_part.split()[0]) # 这个定位可能要重写
        #     except (IndexError, ValueError):
        #         confidence_score = 1 # 这里qwen3B，可能根本没有这个数据，因为他理解能力差太多。所以干醋不要

        # 提取决策
        if "yes" in decision_text and confidence_score >= confidence_threshold:
            return "yes", decision_text
        elif "no" in decision_text and confidence_score >= confidence_threshold:
            return "no", decision_text
        else:
            # 如果置信度低于阈值，默认调用 RAG
            return "no", decision_text

    def select_metapaths(self, subquery: str, reason_history: str, llm_function=None) -> List[str]:
        meta_descriptions = "\n".join([
            f"ID: {index}\nMeta_path: {meta['meta-path']}\n"
            for index, meta in self.metapaths.items()
        ])
        # 构建提示消息
        # print("XXXXXXX", prompt_templates.keys())
        messages = [[
            {'role': 'user', 'content': prompt_templates['meta_path'].format(
                subquery=subquery,
                meta_path=meta_descriptions,
                reason_history=reason_history
            )}
        ]]

        # 调用 llm_function 获取选择的 Metapath ID
        response = llm_function(messages)

        # 解析响应，提取 Metapath ID
        selected_metapath_ids = [int(num) for num in re.findall(r'\d+', response) if int(num) < len(self.metapaths)]

        # print("BBBBB", response,selected_metapath_ids , len(self.metapaths))
        # 确保选择的 Metapath 数量不超过比例
        num_selected = round(len(self.metapaths) * self.ratio)
        return selected_metapath_ids[:num_selected]

    def execute_query(self, subquery: str, decision: str, reason_history: str, llm_function=None, topk=1) -> str:
        answer, selected_metapaths_id, selected_entities, rag_summary = '', '', '', ''
        if decision == "no":
            # 直接使用LLM回答
            messages = [[
    #             {'role': 'system', 'content': "Please answer in English."},

                {'role': 'user', 'content': prompt_templates['direct_answer'].format(subquery=subquery, reason_history=reason_history)}
            ]]

            answer = llm_function(messages)
            return answer, selected_metapaths_id, selected_entities, rag_summary
        else:
            # 调用AgentLow进行RAG检索
            selected_metapaths_id = self.select_metapaths(subquery, reason_history, llm_function)
            selected_metapaths = [self.metapaths[str(i)]["raw-meta-path"] for i in selected_metapaths_id] # 输出name
            answer, selected_entities, rag_summary = self.agent_low.run(subquery, selected_metapaths, reason_history, llm_function, topk)

            return answer, selected_metapaths_id, selected_entities, rag_summary

    def evaluate_answer(self, answer: str, groundtruth: str) -> bool:
        """
        评估答案是否正确。这里不重要。
        """
        prompt = PromptTemplate(
            input_variables=["answer", "groundtruth"],
            template="Does the following answer match the ground truth? Respond with 'yes' or 'no'.\n\n"
                     "Answer: {answer}\n"
                     "Ground Truth: {groundtruth}\n\n"
                     "Response:"
        )
        chain = prompt | self.llm
        response = chain.invoke({"answer": answer, "groundtruth": groundtruth}).strip().lower()
        return "yes" in response

    def combine_answers(self, query: str, paths: List[Dict]) -> str:
        """
        整合所有子查询答案生成最终答案。
        """
        subquery_answers = "\n".join([f"{i + 1}. {path['subquery']}\n{path['answer']}" for i, path in enumerate(paths)])
        prompt = PromptTemplate(
            input_variables=["query", "subquery_answers"],
            template="Combine the following answers into a single coherent response for the query.\nQuery: {query}\n{subquery_answers}"
        )
        chain = prompt | self.llm
        final_answer = chain.invoke({"query": query, "subquery_answers": subquery_answers})
        return final_answer



    def predict(self, query: str, llm_function=None, topk=1):
        # 训练时候才启用
        queue = deque()
        # initial_subqueries = [self.rewrite_query_follow(query, '', llm_function)]
        initial_subqueries = self.rewrite_query(query, '', llm_function)

        for subquery in initial_subqueries[:1]:
            subquery = f"Base query: {query} \n\n" + f"Sub query: {subquery}\n"
            queue.append((subquery, []))  # 每个子查询附带当前路径信息
        # print("AAAAAAAA",initial_subqueries)
        iteration_count = 0  # 迭代计数器
        # subquery_hist, subquery_answers, reason_history = [], [], ''
        # reverse_subquery_hist, reverse_subquery_answers, reverse_reason_history = [], [], ''

        golden_paths, reverse_negative_paths = [], []

        while queue and iteration_count < self.max_iterations: # 如果是离线强化学习的话，靠近所谓的golden answer在该步骤的决断，reward更大，反之则更小。但我们这里是在线强化学习。
            if iteration_count == 0: # 其实就是一开始的LLM，是否终止。
                check_prompt = prompt_templates['complete_checking'].format(query=query,
                                                                            reason_history='')  # 会先给出query，然后才给出answer
                messages = [[
                    {'role': 'user', 'content': check_prompt}
                ]]
                decision = llm_function(messages,template='complete_checking')
                if "incomplete" not in decision.lower():
                    return decision, [[]], [[]]  # 提前返回最终答案, 简单的数据。

            current_subquery, current_path = queue.popleft()
            reason_history = current_path[-1]['reason_history'] if current_path else ''
            decision, decision_text = self.decide_action(current_subquery, reason_history, self.confidence_score, llm_function)
            answer, selected_metapaths_id, selected_entities, rag_summary = self.execute_query(current_subquery, decision, reason_history, llm_function, topk)
            source = "RAG" if decision=="yes" else "LLM"

            # 创建hard negative, 相同长度的路径（只是选择不同）， 一般的概率选别的
            all_sources = ['LLM', 'RAG']
            # if random.random() <0.5:
            reverse_decision = ["yes", "no"]
            reverse_decision.remove(decision)
            all_sources.remove(source)
            reverse_source = all_sources[0]
            reverse_answer, reverse_selected_metapaths_id, reverse_selected_entities, reverse_rag_summary = [],[],[],[]# self.execute_query(current_subquery, reverse_decision[0],  reason_history, llm_function, topk)  # LLM 生成, 这里用不用reverse值得商榷，感觉虚构
            # else: # 这里整体可以使用之前的reason PATH
            #     reverse_decision = [decision]
            #     reverse_source = source
            #     reverse_answer,reverse_selected_metapaths_id, reverse_selected_entities, reverse_rag_summary = answer, selected_metapaths_id, selected_entities, rag_summary# self.execute_query(current_subquery, reverse_decision[0],  reason_history, llm_function)

            # 记录当前路径信息
            path_info = {
                "source": source,
                "subquery": current_subquery,
                "subanswer": answer,
                "decision": decision,
                "chosen_metapaths": selected_metapaths_id,
                "chosen_entities": selected_entities,
                "naive_results": rag_summary,
                "rewrite": '\n'.join([f"{i+1}. {subquery}\n" for i, subquery in enumerate(initial_subqueries)]),  # 记录初始子查询
            }
            reverse_path_info = {
                "source": reverse_source,
                "subquery": current_subquery,
                "subanswer": reverse_answer,
                "decision": decision,
                "chosen_metapaths": reverse_selected_metapaths_id,
                "chosen_entities": reverse_selected_entities,
                "naive_results": reverse_rag_summary,
                "rewrite": '\n'.join([f"{i+1}. {subquery}\n" for i, subquery in enumerate(initial_subqueries)]),  # 记录初始子查询
            }
            reason_history = reason_history + "\n" + "{}. {}\nAnswer{}".format(len(current_path), current_subquery, answer)
            reverse_reason_history = reason_history + "\n" + "{}. {}\nAnswer{}".format(len(current_path), current_subquery, reverse_answer)
            path_info['reason_history'], reverse_path_info['reason_history'] = reason_history, reverse_reason_history

            # 检查当前答案是否可以回答主查询
            check_prompt = prompt_templates['complete_checking'].format(query=query, reason_history=reason_history)  # 会先给出query，然后才给出answer
            messages = [[
#             {'role': 'system', 'content': "Please answer in English."},

                {'role': 'user', 'content': check_prompt}
            ]]

            decision = llm_function(messages, template='complete_checking')
            path_info['final_check'] = decision

            if "incomplete" not in decision.lower(): # 有答案了
                path_info['follow'] = ''
                golden_paths.append(current_path + [path_info]) # 到时候让答案离我们的思考过程更近
                reverse_negative_paths.append(current_path + [reverse_path_info])
                return decision, golden_paths, reverse_negative_paths  # 提前返回最终答案
            else: # 继续分解
                reverse_negative_paths.append(current_path + [path_info])
                reverse_negative_paths.append(current_path + [path_info])

            follow = self.rewrite_query_follow(current_subquery, reason_history, llm_function)
            path_info['follow'] = follow
            new_path = current_path + [path_info]
            queue.append((follow, new_path)) # 在PPO阶段只需要一些hard negative，无需完备的路径信息

            iteration_count += 1  # 增加迭代计数器

        # 如果未提前返回，则整合所有子查询答案
        golden_paths.append(current_path)
        # check_prompt = prompt_templates['complete_checking'].format(query=query, reason_history=reason_history)
        messages = [[
#             {'role': 'system', 'content': "Please answer in English."},
            {'role': 'user', 'content': check_prompt}
        ]]
        final_answer = llm_function(messages,template='complete_checking') # str， [[{},{}, {}]], [[{}, {}, {}],[{},{}]]
        return final_answer, golden_paths, reverse_negative_paths # str， [[{},{}, {}]], [[{}, {}, {}],[{},{}]] 让我的golden path比别的path更准，比相同的更短，还是说离答案在语义上更贴近。 也可以和采样时候得到的数据相一致即可。


    #
    # def run(self, query: str, llm_function=None) -> str: # for inference
    #     # return golden path and negative path when training else final answer
    #     queue = deque()
    #     initial_subqueries = self.rewrite_query(query, '', llm_function)
    #     for subquery in initial_subqueries:
    #         queue.append(subquery)
    #     iteration_count = 0  # 迭代计数器
    #
    #     # print("AAAAAA", query, queue)
    #     subquery_hist = []
    #     subquery_answers = []
    #     reason_history = ""
    #
    #     while queue and iteration_count <= self.max_iterations:
    #         print("AAAAAA", query, queue, iteration_count)
    #         current_subquery = queue.popleft()
    #         decision, decision_text = self.decide_action(current_subquery, reason_history, 0.7, llm_function)
    #         answer = self.execute_query(current_subquery, decision, reason_history, llm_function)
    #         print("BBBBB", decision, answer)
    #         subquery_answers.append(answer)
    #         subquery_hist.append(current_subquery)
    #
    #         # 检查当前答案是否可以回答主查询
    #         reason_history = "\n".join(
    #             [f"{i + 1}. {sq}\n{ans}" for i, (sq, ans) in enumerate(zip(subquery_hist, subquery_answers))])
    #         check_prompt = prompt_templates['complete_checking'].format(query=query, reason_history=reason_history)  # 会先给出query，然后才给出answer
    #         messages = [
    #             {'role': 'system', 'content': ""},
    #             {'role': 'user', 'content': check_prompt}
    #         ]
    #         decision = llm_function(messages)
    #         print('CCCCCCC', check_prompt)
    #         print("DDDDDD", decision)
    #         if "no" not in decision:
    #             print('EEEEEEEE')
    #             return decision  # 提前返回最终答案
    #         else:
    #             print('FFFFF')
    #             new_subqueries = self.rewrite_query(current_subquery, reason_history, llm_function)
    #             for new_subquery in new_subqueries:
    #                 queue.append(new_subquery)
    #
    #         iteration_count += 1  # 增加迭代计数器
    #
    #     # 如果未提前返回，则整合所有子查询答案
    #     print("HHHHH")
    #     check_prompt = prompt_templates['complete_checking'].format(query=query, reason_history=reason_history)
    #     messages = [
    #         {'role': 'system', 'content': ""},
    #         {'role': 'user', 'content': check_prompt}
    #     ]
    #     final_answer = llm_function(messages)
    #     return final_answer, reason_history
