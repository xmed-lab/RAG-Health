# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : agent_top.py
# Time       ：14/3/2025 5:31 pm
# Author     ：Any
# version    ：python 
# Description：
"""
import random
import re
from typing import List, Dict, Any, Tuple
# from langchain import LLMChain, PromptTemplate
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import BaseLLM
from collections import deque
from agent_low import AgentLow
from instructions_template import prompt_templates
from langchain.schema.runnable import RunnableConfig
from utils import normalize_answer, normalize_answer_multichoice, extract_numbers, locate_answer


# AgentTop类
class AgentTop:
    def __init__(self, llm: BaseLLM, metapaths: List[Dict[str, str]], agent_low: AgentLow, ratio: float = 0.3,  max_iterations: int = 4, confidence: float = 0.7, config=None):
        self.llm = llm
        self.metapaths = metapaths  # metapaths是一个字典列表，包含name和description
        self.agent_low = agent_low
        self.ratio = ratio
        self.max_iterations = max_iterations
        self.confidence_score = confidence
        self.config = config

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

    def rewrite_query(self, query: str, reason_history: str, run_config:RunnableConfig=RunnableConfig(llm={})) -> List[str]:
        # 提示模板：明确要求生成 3-5 个子查询
        prompt = PromptTemplate(
            input_variables=["query"],
            template=prompt_templates['rewrite']
        )
        chain = prompt | self.llm
        result = chain.invoke({"query": query}, config=run_config) # , "reason_history": reason_history 不要加这个不然会爆炸
        # 处理 LLM 输出格式
        split_by_number = re.split(r'\b(1|2|3|4|5)\s*[.\s]\s*', result)  # 分割编号与内容
        subqueries = []
        for part in split_by_number:
            
            if not part.strip() or part.strip().isdigit():
                continue
            subqueries.extend([line.strip() for line in part.split('\n') if line.strip()])

        subqueries = self.clean_subquery(subqueries)  # 清洗子查询

        if len(subqueries) < 3:
            # print("Warning: Generated subqueries are insufficient. Falling back to rule-based generation.")
            subqueries.extend(self._generate_fallback_subqueries(query))

        # 限制最多返回 n个初始查询
        return subqueries[:3]


    def rewrite_query_follow(self, query: str, reason_history: str, run_config:RunnableConfig=RunnableConfig(llm={})) -> List[str]:
        # 提示模板：明确要求生成 3-5 个子查询
        prompt = PromptTemplate(
            input_variables=["query", "reason_history"],
            template=prompt_templates['follow']
        )
        chain = prompt | self.llm
        result = chain.invoke({"query": query, "reason_history": reason_history}, config=run_config) # 不要加这个不然会爆炸

        # 处理 LLM 输出格式
        followup = result.strip()
        return followup

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

    def decide_action(self, subquery: str, reason_history: str, confidence_threshold: float = 0.7, run_config:RunnableConfig=RunnableConfig(llm={})) -> str:
        prompt = PromptTemplate(
            input_variables=["subquery", "reason_history"],
            template=prompt_templates["decide"]
        )
        chain = prompt | self.llm
        decision_text = chain.invoke({"subquery": subquery, "reason_history": reason_history}, config=run_config)
        decision_text = decision_text.strip().lower()

        # 提取置信度评分
        confidence_score = 1.0
        # 提取决策
        if "yes" in decision_text and confidence_score >= confidence_threshold:
            return "yes", decision_text
        elif "no" in decision_text and confidence_score >= confidence_threshold:
            return "no", decision_text
        else:
            # 如果置信度低于阈值，默认调用 RAG
            return "no", decision_text

    def select_metapaths(self, subquery: str, reason_history: str, run_config:RunnableConfig=RunnableConfig(llm={})) -> List[str]:
        meta_descriptions = "\n".join([
            f"ID: {index}\nMeta_path: {meta['meta-path']}\n"
            for index, meta in self.metapaths.items()
        ])

        prompt = PromptTemplate(
            input_variables=["subquery", "description", "reason_history"],
            template=prompt_templates['meta_path']
        )
        chain = prompt | self.llm  # LLMChain(llm=self.llm, prompt=prompt)
        response = chain.invoke(
            {"subquery": subquery, "meta_path": meta_descriptions, "reason_history": reason_history}, config=run_config)


        # 解析响应，提取 Metapath ID
        selected_metapath_ids = [int(num) for num in re.findall(r'\d+', response) if int(num) < len(self.metapaths)]

        # 确保选择的 Metapath 数量不超过30%
        num_selected = round(len(self.metapaths) * self.ratio)
        return selected_metapath_ids[:num_selected]

    def execute_query(self, subquery: str, decision: str, reason_history: str, run_config:RunnableConfig=RunnableConfig(llm={}), topk=1) -> str:
        answer, selected_metapaths_id, selected_entities, rag_summary = '', '', '', ''
        if decision == "no":
            # 直接使用LLM回答
            prompt = PromptTemplate(
                input_variables=["subquery", "reason_history"],
                template=prompt_templates['direct_answer']
            )
            chain = prompt | self.llm #LLMChain(llm=self.llm, prompt=prompt)
            answer = chain.invoke({"subquery":subquery, "reason_history":reason_history}, config=run_config)
            return answer, selected_metapaths_id, selected_entities, rag_summary
        else:
            # 调用AgentLow进行RAG检索
            selected_metapaths_id = self.select_metapaths(subquery, reason_history)
            selected_metapaths = [self.metapaths[str(i)]["raw-meta-path"] for i in selected_metapaths_id] # 输出name
            answer, selected_entities, rag_summary = self.agent_low.run(subquery, selected_metapaths, reason_history, run_config, topk=topk) # 这里的rag-summary就是naive rag
            return answer, selected_metapaths_id, selected_entities, rag_summary # answer is intermediate answer；select entities是KG, rag_summary是naive

    def evaluate_answer(self, answer: str, groundtruth: str, token_overlap_threshold=0.2):
        """
        评估答案是否正确。支持判断题、选择题和开放式问题； 这里是粗粒度的
        """
        answer =  locate_answer(answer) # 遵循和evaluation一样的步骤
        # 单选择题，判断
        if self.config['TASK'] in ['MOR', 'REA', 'IHM']:
            # normalize
            answer = normalize_answer(answer)
            groundtruth = normalize_answer(groundtruth)
            if groundtruth.lower() in answer.lower():
                return True
            else:
                return False
        elif self.config['TASK'] in ['LOS']:
            answer = normalize_answer(answer)
            groundtruth = normalize_answer(groundtruth)
            scalr_gold = extract_numbers(groundtruth)[
                0]  # list(map(lambda gold: extract_numbers(golds)[0]), golds) # 第一个数字
            try:
                scalr_pred = extract_numbers(answer)[-1]  # list(map(lambda pred: extract_numbers(pred)[0]), preds)
                return scalr_gold == scalr_pred
            except:  # 没有数字
                return False
        elif self.config['TASK'] in ['SINGLE']:
            answer = normalize_answer_multichoice(answer)
            groundtruth = normalize_answer_multichoice(groundtruth)
            # 直接文本匹配（处理判断题）
            return answer == groundtruth
        # 开放式问题（基于token重叠）
        elif self.config['TASK'] in ['SUMMARY']:
            # 计算共同token数量
            # normalize
            answer = normalize_answer(answer)
            groundtruth = normalize_answer(groundtruth)

            common_tokens = set(answer) & set(groundtruth)
            overlap_ratio = len(common_tokens) / max(len(set(groundtruth)), 1)

            return overlap_ratio >= token_overlap_threshold

        elif self.config['TASK'] in ['MULTIPLE']:
            raise NotImplementedError("Multiple choice evaluation is not implemented yet.")
        elif self.config['TASK'] in ['DIAG', 'REC', 'PHE']:
            raise NotImplementedError("Diagnosis, Recommendation, and Phenotyping evaluation is not implemented yet.")



    def combine_answers(self, query: str, paths: List[Dict], run_config:RunnableConfig=RunnableConfig(llm={})) -> str:
        """
        整合所有子查询答案生成最终答案。
        """
        subquery_answers = "\n".join([f"{i + 1}. {path['subquery']}\n{path['answer']}" for i, path in enumerate(paths)])
        prompt = PromptTemplate(
            input_variables=["query", "subquery_answers"],
            template="Combine the following answers into a single coherent response for the query.\nQuery: {query}\n{subquery_answers}"
        )
        chain = prompt | self.llm
        final_answer = chain.invoke({"query": query, "subquery_answers": subquery_answers}, config=run_config)
        return final_answer

    def extract_key_content_with_re(self, text):
        # 正则模式：匹配"Follow these rules:"之后、"Now"之前的内容
        # 考虑引号、换行等格式，使用非贪婪匹配
        pattern = r'Follow these rules:(.*?)Now'
        # 使用DOTALL模式让.匹配包括换行符在内的所有字符
        match = re.search(pattern, text, re.DOTALL)

        if match:
            # 提取匹配到的内容并清理前后空白和多余引号
            extracted = match.group(1).strip().replace('"', '').strip()
            return extracted
        return ""

    def sample_path(self,  query: str, groundtruth: str, run_config:RunnableConfig=RunnableConfig(llm={}), topk=1):
        # 筛选Qwen72B生成数据
        # 训练时候才启用
        queue = deque()
        initial_subqueries = self.rewrite_query(query, '', run_config=run_config) # [(sub,''), (sub,'')]
        for subquery in initial_subqueries[:2]: # 只用2个
            subquery = f"Base query: {query} \n\n" + f"Sub query: {subquery}\n"
            queue.append((subquery, []))  # 每个子查询附带当前路径信息

        iteration_count = 0  # 迭代计数器
        golden_paths, negative_paths = [], []
        while queue and iteration_count < self.max_iterations:
            current_subquery, current_path = queue.popleft() # query, [{}]
            reason_history = current_path[-1]['reason_history'] if current_path else ''
            # decision, decision_text = self.decide_action(current_subquery, reason_history, self.confidence_score)
            answer, selected_metapaths_id, selected_entities, rag_summary = self.execute_query(current_subquery, "no", reason_history,run_config=run_config, topk=topk) # answer, '','',''; 能自己回答；
            reverse_answer, reverse_selected_metapaths_id, reverse_selected_entities, reverse_rag_summary = self.execute_query(current_subquery, "yes", reason_history, run_config=run_config, topk=topk) # answer, ['1'], ['entity'], 'summary'

            # 记录当前路径信息
            path_info = {
                "source": "LLM",
                "subquery": current_subquery,
                "subanswer": answer,
                "decision": "no", # yes应该是RAG,草了。
                "chosen_metapaths": selected_metapaths_id,
                "chosen_entities": selected_entities,
                "naive_results": rag_summary,
                "rewrite": '\n'.join([f"{i+1}. {subquery}\n" for i, subquery in enumerate(initial_subqueries)]),  # 记录初始子查询
            }

            reverse_path_info = {
                "source": "RAG",
                "subquery": current_subquery,
                "subanswer": reverse_answer,
                "decision": "yes",
                "chosen_metapaths": reverse_selected_metapaths_id,
                "chosen_entities": reverse_selected_entities,
                "naive_results": reverse_rag_summary,
                "rewrite": '\n'.join([f"{i+1}. {subquery}\n" for i, subquery in enumerate(initial_subqueries)]),  # 记录初始子查询
            }

            # 更新reason_history
            reason_history = reason_history + "\n" + "{}. {}\nAnswer{}".format(len(current_path), current_subquery, answer)
            reverse_reason_history = reason_history + "\n" + "{}. {}\nAnswer{}".format(len(current_path), current_subquery, reverse_answer)
            path_info['reason_history'], reverse_path_info['reason_history'] = reason_history, reverse_reason_history

            # 检查当前答案是否可以回答主查询
            prompt = PromptTemplate(
                input_variables=["query", "reason_history"],
                template= prompt_templates['complete_checking']
            )
            chain = prompt | self.llm
            decision = chain.invoke({"query": query, "reason_history": reason_history}, config=run_config) # 会先给出query，然后才给出answer
            reverse_decision = chain.invoke({"query": query, "reason_history": reverse_reason_history}, config=run_config) # 会先给出query，然后才给出answer
            llm_is_correct = self.evaluate_answer(decision, groundtruth)
            rag_is_correct = self.evaluate_answer(reverse_decision, groundtruth)
            path_info['final_check'] = decision
            reverse_path_info['final_check'] = reverse_decision

            if llm_is_correct:
                path_info['follow'], reverse_path_info['follow'] = '', ''
                golden_paths.append(current_path + [path_info]) # 找到正确答案提前退出，偏爱LLM [{},{},{}], 如果要看reason path，请选-1,感觉也可以加入一些别的好玩的DPO
                negative_paths.append(current_path + [reverse_path_info])
                # if iteration_count >2: # 检测有无更高的iteration正确的。
                #     print("AAAAAAAA")
                print("*************Searched for this question (LLM)**************")
                break
            elif rag_is_correct:
                path_info['follow'], reverse_path_info['follow'] = '', ''
                golden_paths.append(current_path + [reverse_path_info]) # 找到正确答案提前退出 [{}]
                negative_paths.append(current_path + [path_info])
                # if iteration_count >2: # 检测有无更高的iteration正确的。
                #     print("BBBBBBB")
                print("*************Searched for this question (RAG)**************")
                break
            else:
                negative_paths.append(current_path + [reverse_path_info])
                negative_paths.append(current_path + [path_info])# [{},{}], 但是golden为空

            # 如果没有, 生成新的子查询并加入队列
            follow = self.rewrite_query_follow(current_subquery, reason_history, run_config=run_config) # 'sub'
            path_info['follow'] = follow
            new_path = current_path + [path_info] # [] + [{}]=> [{}]
            queue.append((follow, new_path))
            reverse_follow = self.rewrite_query_follow(current_subquery, reverse_reason_history, run_config=run_config) # 'sub'
            reverse_path_info['follow'] = reverse_follow

            new_path = current_path + [reverse_path_info] # [(sub, []), (sub2, [{llm}]), (sub2, [{rag}])]
            queue.append((reverse_follow, new_path))

            iteration_count += 1  # 增加迭代计数器
        if golden_paths==[]:
            print("*************Not Searched for this question**************")
        return golden_paths, negative_paths, query, groundtruth


    def predict(self, query: str, run_config:RunnableConfig=RunnableConfig(llm={}), topk=1):
        # 训练时候才启用
        queue = deque()
        #initial_subqueries =  [self.rewrite_query_follow(query, '', run_config=run_config)] # 不再需要多样性。这个是能够获得RAG选项的
        initial_subqueries =  self.rewrite_query(query, '', run_config=run_config) # 不再需要多样性。这个是能够获得RAG选项的

        for subquery in initial_subqueries[:1]: # 注意，这里只用一个，上面是为了多样性，这里便于有多个过程。
            subquery = f"Base query: {query} \n\n" + f"Sub query: {subquery}\n"
            queue.append((subquery, []))  # 每个子查询附带当前路径信息
        queue.append((query, []))
        iteration_count = 0  # 迭代计数器
        golden_paths, reverse_negative_paths = [], []

        while queue and iteration_count < self.max_iterations: #
            if iteration_count == 0: # 其实就是一开始的LLM，是否终止。
                prompt = PromptTemplate( # 终止
                    input_variables=["query", "reason_history"],
                    template=prompt_templates['complete_checking']
                )
                chain = prompt | self.llm
                decision = chain.invoke({"query": query, "reason_history": ''}, # 因为猜不对的很多，所以模型可能倾向于直接给出答案
                                        config=run_config)  # 会先给出query，然后才给出answer
                if "incomplete" not in decision.lower():
                    return decision, [], []  # 提前返回最终答案, 简单的数据。

            current_subquery, current_path = queue.popleft()
            reason_history = current_path[-1]['reason_history'] if current_path else ''
            decision, decision_text = self.decide_action(current_subquery, reason_history, self.confidence_score, run_config=run_config)
            answer, selected_metapaths_id, selected_entities, rag_summary = self.execute_query(current_subquery, decision, reason_history, run_config=run_config, topk=topk)
            source = "RAG" if decision=="yes" else "LLM"
            # 创建hard negative, 相同长度的路径（只是选择不同）， 一般的概率选别的; 这里为了让正负看起来不一样；
            all_sources = ['LLM', 'RAG']
            if random.random() <0.5:
                reverse_decision = ["yes", "no"]
                reverse_decision.remove(decision)
                all_sources.remove(source)
                reverse_source = all_sources[0]
                reverse_answer, reverse_selected_metapaths_id, reverse_selected_entities, reverse_rag_summary = [],[], [], []#case的时候打开,  self.execute_query(current_subquery, reverse_decision[0],  reason_history, run_config=run_config, topk=topk)  # LLM 生成, 这里用不用reverse值得商榷，感觉虚构
            else: # 这里整体可以使用之前的reason PATH
                reverse_decision = [decision]
                reverse_source = source
                reverse_answer,reverse_selected_metapaths_id, reverse_selected_entities, reverse_rag_summary = answer, selected_metapaths_id, selected_entities, rag_summary# self.execute_query(current_subquery, reverse_decision[0],  reason_history, llm_function)

            # 记录当前路径信息
            path_info = {
                "source": source,
                "subquery": current_subquery,
                "subanswer": answer,
                "decision": decision,
                "chosen_metapaths": selected_metapaths_id,
                "chosen_entities": selected_entities,
                "naive_results": rag_summary,
                "rewrite": '\n'.join([f"{i+1}. {subquery}\n" for i, subquery in enumerate(initial_subqueries)]),
            }
            reverse_path_info = {
                "source": reverse_source,
                "subquery": current_subquery,
                "subanswer": reverse_answer,
                "decision": decision,
                "chosen_metapaths": reverse_selected_metapaths_id,
                "chosen_entities": reverse_selected_entities,
                "naive_results": reverse_rag_summary,
                "rewrite": '\n'.join([f"{i+1}. {subquery}\n" for i, subquery in enumerate(initial_subqueries)]),
            }
            reason_history = reason_history + "\n" + "{}. {}\nAnswer{}".format(len(current_path), current_subquery, answer)
            reverse_reason_history = reason_history + "\n" + "{}. {}\nAnswer{}".format(len(current_path), current_subquery, reverse_answer)
            path_info['reason_history'], reverse_path_info['reason_history'] = reason_history, reverse_reason_history
            prompt = PromptTemplate(
                input_variables=["query", "reason_history"],
                template= prompt_templates['complete_checking']
            )
            chain = prompt | self.llm
            decision = chain.invoke({"query": query, "reason_history": reason_history}, config=run_config) # 会先给出query，然后才给出answer
            path_info['final_check'] = decision

            if "incomplete" not in decision.lower():
                path_info['follow'] = ''
                golden_paths.append(current_path + [path_info]) #
                reverse_negative_paths.append(current_path + [reverse_path_info])
                return decision, golden_paths, reverse_negative_paths  # 提前返回最终答案
            else: # 继续分解
                reverse_negative_paths.append(current_path + [path_info])
                reverse_negative_paths.append(current_path + [path_info])

            follow = self.rewrite_query_follow(current_subquery, reason_history, run_config=run_config)
            path_info['follow'] = follow
            new_path = current_path + [path_info]
            queue.append((follow, new_path)) # 在PPO阶段只需要一些hard negative，无需完备的路径信息

            iteration_count += 1  # 增加迭代计数器

        golden_paths.append(current_path)
        # check_prompt = prompt_templates['complete_checking'].format(query=query, reason_history=reason_history)
        prompt = PromptTemplate(
            input_variables=["query", "reason_history"],
            template=prompt_templates['complete_checking']
        )
        chain = prompt | self.llm
        final_answer = chain.invoke({"query": query, "reason_history": reason_history}, config=run_config)

        return final_answer, golden_paths, reverse_negative_paths 
