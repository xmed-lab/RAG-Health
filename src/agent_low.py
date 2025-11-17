# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : agent_low.py
# Time       ：14/3/2025 4:56 pm
# Author     ：Any
# version    ：python 
# Description：
"""
import threading
import requests

import os.path
import gc
import random
import dgl
import torch
import multiprocessing
from collections import defaultdict
import numpy as np
from typing import Dict, List, Tuple, Optional, Set, Any
# from langchain.embeddings import HuggingFaceEmbeddings
# from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain.schema import Document
from tqdm import tqdm
from instructions_template import prompt_templates
from utils import create_heterogeneous_graph, generate_node_text, generate_edge_text
from concurrent.futures import ThreadPoolExecutor
from langchain.schema.runnable import RunnableConfig


class AgentLow:
    """图向量数据库系统，用于DGL异构图的GraphRAG检索"""

    def __init__(self, embedding_model, llm: Any, TOPK : int=3, dim: int = 768):
        """
        初始化GraphRAG数据库

        Args:
            embedding_model: 文本向量化模型
            dim: 向量维度
        """
        self.graph = None
        self.graph_node_types = None # for subgraph
        self.llm = llm
        self.embed_model = embedding_model
        self.dim = dim

        # 节点和边向量存储
        self.node_stores = {}
        self.edge_stores = {}

        # 存储节点和边的映射及属性
        self.node_mapping = {}
        self.edge_mapping = {}
        self.node_text_dict = {}
        self.edge_text_dict = {}
        # Naive RAG 文本块存储
        self.text_chunks = []  # 存储所有文本块
        self.text_chunk_store = FAISS.from_texts(['dummy text'], self.embed_model, metadatas=[{"id": -1}])  # 初始化空的 FAISS 索引

        # 缓存 PageRank 分数
        self.pagerank_scores = None

        self.TOPK = TOPK

        self.root_path = '/hpc2hdd/home/xxxs349/xxxc/RAGHealth/data/ready/'  # 设置根路径
        self.vector_store_lock = threading.Lock() # double free现象

    def add_text_chunks(self, chunks: List[str]):
        if os.path.exists(self.root_path + "/faiss_index"):
            # 如果索引已经存在，直接加载
            print("Load existing FAISS index from local storage. If you want to update the index, please delete the existing index first.")
            self.text_chunk_store = FAISS.load_local(self.root_path + "/faiss_index", self.embed_model, allow_dangerous_deserialization=True)
            document_num = len(self.text_chunk_store.docstore._dict)
        else:
            print("Create FAISS index from scratch. This may take a while for large datasets.")
            # 生成文档对象
            documents = [
                Document(page_content=chunk, metadata={"id": i})
                for i, chunk in enumerate(chunks)
            ]
            document_num = len(documents)

            # 添加到向量存储
            # self.text_chunk_store.add_documents(documents)
            self._add_documents_in_batches(self.text_chunk_store, documents)


            # 持久化，不然一次时间巨长
            self.text_chunk_store.save_local(self.root_path + "/faiss_index")  # 保存索引到本地

        self.text_chunks.extend(chunks)
        return document_num

    def naive_rag_retrieval(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        # 查询向量存储
        retrieved = self.text_chunk_store.similarity_search_with_score(query, k=top_k)
        # 格式化结果
        results = []
        for doc, score in retrieved:
            # 确保 metadata 包含 "id"，否则使用默认值
            chunk_id = doc.metadata.get("id", -1)  # 如果 "id" 不存在，默认值为 -1
            if chunk_id == -1:
                continue  # 跳过 id 为 -1 的结果
            text = self.text_chunks[chunk_id] if chunk_id != -1 else doc.page_content
            results.append({
                "text": text,
                "score": float(score),
                "chunk_id": chunk_id
            })

        return results

    def load_graph(self, g: dgl.DGLGraph):
        """加载DGL异构图, 创建index"""
        self.graph = g

        self.graph_node_types = np.array(self.graph.ntypes)

        # 为每种节点和边类型初始化存储
        for ntype in g.ntypes:
            if ntype not in self.node_stores:
                # 使用 FAISS 初始化空的向量存储
                index = FAISS.from_texts(
                    texts=["dummy_text"],  # 添加一个虚拟文本
                    embedding=self.embed_model,  # 嵌入模型
                    metadatas=[{"id": -1}]  # 虚拟元数据
                )
                self.node_stores[ntype] = index
                self.node_mapping[ntype] = {}
                self.node_text_dict[ntype] = {}

        for etype in g.canonical_etypes:
            if etype not in self.edge_stores:
                # 使用 FAISS 初始化空的向量存储
                index = FAISS.from_texts(
                    texts=["dummy_text"],  # 添加一个虚拟文本
                    embedding=self.embed_model,  # 嵌入模型
                    metadatas=[{"id": -1}]  # 虚拟元数据
                )
                self.edge_stores[etype] = index
                self.edge_mapping[etype] = {}
                self.edge_text_dict[etype] = {}

    def index_nodes(self, node_type: str, node_ids: List[int], texts: List[str]):
        """索引节点文本"""
        assert len(node_ids) == len(texts), "The node ID and text quantity must be consistent."
        if os.path.exists(self.root_path + '/nodes_index' + f"/{node_type}_index"):
            # 如果索引已经存在，直接加载
            print("Load existing Node index from local storage. If you want to update the index, please delete the existing index first.")
            self.node_stores[node_type] = FAISS.load_local(self.root_path + '/nodes_index' + f"/{node_type}_index", self.embed_model, allow_dangerous_deserialization=True)
            document_num = len(self.node_stores[node_type].docstore._dict)
        else:
            print("Create Node Index. ")
            # 生成文档对象
            documents = [
                Document(page_content=text, metadata={"id": node_id})
                for node_id, text in zip(node_ids, texts)
            ]
            document_num = len(documents)

            self._add_documents_in_batches(self.node_stores[node_type], documents)

            # 持久化，不然一次时间巨长
            self.node_stores[node_type].save_local(self.root_path + '/nodes_index' + f"/{node_type}_index")  # 保存索引到本地


        # 更新映射
        for i, node_id in enumerate(node_ids):
            self.node_mapping[node_type][i] = node_id  # 字典形式存储
            self.node_text_dict[node_type][node_id] = texts[i]


        return document_num

    def index_edges(self, edge_type: Tuple[str, str, str], edge_ids: List[int], texts: List[str]):
        """索引边文本"""
        assert len(edge_ids) == len(texts), "The edge ID and text quantity must be consistent."
        safe_edge_type = '_'.join(edge_type).replace('/', '_') # 有的元路径有/
        print(safe_edge_type)
        if safe_edge_type in ['drug_drug_effect_effect_phenotype_index','drug_drug_drug_drug','effect_phenotype_disease_phenotype_positive_disease','gene_protein_disease_protein_disease']:  # 太鸡儿大了
            document_num =0
            return 0
        if os.path.exists(self.root_path + '/meta_path_index' + f"/{safe_edge_type}_index"):
            # 如果索引已经存在，直接加载
            print("Load existing Edge index from local storage. If you want to update the index, please delete the existing index first.")

            self.edge_stores[edge_type] = FAISS.load_local(self.root_path + '/meta_path_index' + f"/{safe_edge_type}_index", self.embed_model, allow_dangerous_deserialization=True)
            document_num = len(self.edge_stores[edge_type].docstore._dict)
        else:
            print("Create Edge Index.")
            batch_size = 100  # 可以根据实际情况调整批次大小
            document_num = len(edge_ids)
            for i in range(0, len(edge_ids), batch_size):
                # 生成当前批次的文档对象
                batch_documents = [
                    Document(page_content=text, metadata={"id": edge_id})
                    for edge_id, text in zip(edge_ids[i:i + batch_size], texts[i:i + batch_size])
                ]
                # 添加当前批次的文档到向量存储
                self.edge_stores[edge_type].add_documents(batch_documents)
                # 释放当前批次的内存
                del batch_documents
                gc.collect()

                # 定期保存索引
                if (i // batch_size + 1) % 10 == 0:  # 每处理10个批次保存一次索引
                    self.edge_stores[edge_type].save_local(
                        self.root_path + '/meta_path_index' + f"/{safe_edge_type}_index")

            # 最后再保存一次索引，确保所有文档都已保存
            self.edge_stores[edge_type].save_local(self.root_path + '/meta_path_index' + f"/{safe_edge_type}_index")

        # 更新映射
        for i, edge_id in enumerate(edge_ids):
            self.edge_mapping[edge_type][i] = edge_id
            self.edge_text_dict[edge_type][edge_id] = texts[i]

        return document_num



    def _add_documents_in_batches(self, vector_store, documents, batch_size=1):
        with ThreadPoolExecutor() as executor:
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                executor.submit(self._add_batch_safely, vector_store, batch)

    def _add_batch_safely(self, vector_store, batch):
        with self.vector_store_lock:
            vector_store.add_documents(batch)

    def node_retrieval(self, query: str, node_types: Optional[List[str]] = None, top_k: int = 1) -> Dict[
        str, List[Dict]]:
        results = {} # []
        node_types = node_types or list(self.node_stores.keys())

        for ntype in node_types:
            if ntype not in self.node_stores:
                continue

            # 查询向量存储
            retrieved = self.node_stores[ntype].similarity_search(query, k=top_k)

            # 格式化结果
            type_results = []
            for doc in retrieved:
                node_id = doc.metadata["id"]
                if node_id == -1:  # 过滤掉虚拟数据
                    continue
                text = self.node_text_dict[ntype].get(node_id, doc.page_content)
                type_results.append({
                    "id": node_id,
                    "text": text
                })
                # type_results.append(text)
            if type_results:
                # results.append(type_results)
                results[ntype] = type_results
        #
        return results

    def edge_retrieval(self, query: str, edge_types: Optional[List[Tuple[str, str, str]]] = None, top_k: int = 1) -> \
    Dict[Tuple[str, str, str], List[Dict]]:
        results = []
        edge_types = edge_types or list(self.edge_stores.keys())

        for etype in edge_types:
            if etype not in self.edge_stores:
                continue

            # 查询向量存储
            retrieved = self.edge_stores[etype].similarity_search(query, k=top_k)

            # 格式化结果
            type_results = []
            for doc in retrieved:
                edge_id = doc.metadata["id"]
                if edge_id == -1:  # 过滤掉虚拟数据
                    continue
                text = self.edge_text_dict[etype].get(edge_id, doc.page_content)
                type_results.append(text)
            if type_results:
                # results[etype] = type_results
                results.append(type_results)
        return results


    def subgraph_retrieval(self, query: str, max_nodes: int = 50, num_walks: int = 2, walk_length: int = 5,
                           metapaths: List[List[str]] = None) -> dgl.DGLGraph:
        """
        基于随机游走检索与查询相关的子图，支持多种元路径。

        Args:
            query: 查询文本
            max_nodes: 返回子图的最大节点数
            num_walks: 随机游走的次数
            walk_length: 每次随机游走的长度
            metapaths: 元路径列表，例如 [["user", "buys", "product"], ["product", "belongs_to", "category"]]

        Returns:
            包含相关节点和边的DGL子图
        """
        if metapaths is None:
            metapaths = [["disease", "disease-disease", "disease"], ["disease", "indication", "drug"]]  # 默认元路径

        # 获取种子节点
        seed_nodes_results = self.node_retrieval(query, top_k=self.TOPK)  # 只选种子节点进行扩展
        seed_nodes = {
            ntype: [item["id"] for item in nodes]
            for ntype, nodes in seed_nodes_results.items()
        } # {'user': [0, 1], 'product': [0, 1], 'category': [0, 1]}

        # 初始化节点集合
        nodes_dict = {ntype: set() for ntype in self.graph.ntypes}
        # print('BBBBB3333', seed_nodes, metapaths)

        # 对每种元路径进行随机游走
        for metapath in metapaths: # 并行meta-path采样
            # 获取元路径的起始节点类型
            start_ntype = metapath[0]

            # 获取起始节点类型的种子节点
            if start_ntype not in seed_nodes or not seed_nodes[start_ntype]:
                continue  # 如果没有种子节点，跳过该元路径

            # print('BBBBB4445554', start_ntype, seed_nodes, metapath)


            # 将种子节点转换为张量
            start_nodes_tensor = torch.tensor(seed_nodes[start_ntype], dtype=torch.int64)

            # 进行随机游走
            # for _ in range():
            walks, types = dgl.sampling.random_walk(
                self.graph,  # 输入的图
                start_nodes_tensor.repeat(num_walks),  # 种子节点列表
                metapath=[metapath] * walk_length  # 元路径, 因为只有1跳，所以直接重复
            )  # 提取游走路径
            # print('BB5555', walks, types, self.graph.ntypes)
            node_types = self.graph_node_types[types.tolist()]  # 获取节点类型
            # 更新节点集合
            for index in range(walks.shape[1]):  # 遍历每条游走路径
                node_id = walks[:, index]
                node_type = node_types[index]

                # 过滤有效节点并更新
                valid_nodes = node_id[node_id != -1].unique()  # 过滤无效节点
                nodes_dict[node_type].update(valid_nodes.tolist())


        if sum(len(nodes) for nodes in nodes_dict.values()) >= max_nodes:
            # 构建子图, 随机采样max_nodes
            all_samples = [(key, value) for key, values in nodes_dict.items() for value in values]

            # 随机选择 3 个样本
            samples = random.sample(all_samples, 3)

            result = defaultdict(list)
            _ = [result[key].append(value) for key, value in samples]

            # 转换为普通字典
            nodes_dict = dict(result)

        # 构建子图
        induced_nodes = {ntype: torch.tensor(list(nodes), dtype=torch.int64) for ntype, nodes in nodes_dict.items() if
                         nodes}
        # print("KKKKKK,", induced_nodes)
        subgraph = dgl.node_subgraph(self.graph, induced_nodes) # 扩展了其实
        subgraph_dic = {ntype:subgraph.nodes(ntype) for ntype in subgraph.ntypes if subgraph.number_of_nodes(ntype) > 0} # 过滤掉没有节点的类型
        subgraph = {ntype: list(map(self.node_text_dict[ntype].__getitem__, val.tolist())) for ntype, val in subgraph_dic.items()}#dgl.node_subgraph(self.graph, induced_nodes)  # 构建子图
        return subgraph

    def query_graph(self, query: str, selected_metapaths: List[str]) -> Dict[str, Any]:
        """
        综合查询接口，同时返回相关节点、边和子图. 这里得重写，因为我们需要元路径在不同的数据库里面检索。

        Args:
            query: 查询文本
            selected_metapaths: 选定的元路径列表

        Returns:
            包含节点、边和子图的结果字典
        """
        # 根据选定的元路径过滤节点和边类型
        node_types = list({node for path in selected_metapaths for node in (path[0], path[2])}) # list(set([mp.split("->")[0] for mp in selected_metapaths if "->" in mp]))
        edge_types = list({path[1] for path in selected_metapaths}) # list(set([tuple(mp.split("->")) for mp in selected_metapaths if "->" in mp]))

        result = {
            "query": query,
            "nodes": self.node_retrieval(query, node_types=node_types, top_k=self.TOPK),
            "edges": self.edge_retrieval(query, edge_types=selected_metapaths, top_k=self.TOPK), # 这里必须是meta_path
        }


        # 添加子图
        try:
            subgraph = self.subgraph_retrieval(query, walk_length=1, num_walks=2, max_nodes=30, metapaths=selected_metapaths) # 当完全没有的时候，可能会报错，所以限制性下walk length
            result["subgraph"] = subgraph
        except Exception as e:
            print(f"子图检索失败: {e}")
            result["subgraph"] = ''

        return result

    def summarize_query_with_naive_rag(self, query: str, naive_results: List[Dict[str, Any]]) -> str:
        # 构建 prompt 的头部
        prompt = f"User Query: {query}\n\n"

        # 添加检索到的文本片段
        prompt += "Retrieved Text Chunks:\n"
        for i, result in enumerate(naive_results):
            prompt += f"- Chunk {i + 1} (Score: {result['score']}):\n"
            prompt += f"  {result['text']}\n"

        # 添加总结指令
        prompt += "\nBased on the above information, please summarize the user query and related text chunks:\n"

        # 调用语言模型生成总结（这里假设使用 OpenAI GPT）
        summary = self.llm(prompt)
        return summary


    def summarize_query_with_kg(self, query: str, kg_results: Dict[str, Any]) -> str:

        # 提取检索到的节点、边和子图信息
        nodes = kg_results.get("nodes", {})
        edges = kg_results.get("edges", {})
        subgraph = kg_results.get("subgraph", '')

        # 构建 prompt 的头部
        prompt = f"User Query: {query}\n\n"

        # 添加节点信息
        prompt += "Retrieved Nodes:\n"
        for ntype, node_list in nodes.items():
            prompt += f"- Node Type: {ntype}\n"
            for node in node_list:
                prompt += f"  - Node ID: {node['id']}, Text: {node['text']}\n"

        # 添加边信息
        prompt += "\nRetrieved Edges:\n"
        for etype, edge_list in edges.items():
            src_type, edge_type, dst_type = etype
            prompt += f"- Edge Type: {src_type} -> {edge_type} -> {dst_type}\n"
            for edge in edge_list:
                prompt += f"  - Edge ID: {edge['id']}, Text: {edge['text']}\n"

        # 添加子图信息
        if subgraph:
            prompt += "\nRetrieved Subgraphs:\n"
            prompt += f"- Subgraphs Nodes Count: {sum(subgraph.num_nodes(ntype) for ntype in subgraph.ntypes)}\n"
            prompt += f"- Subgraphs Edges Count: {sum(subgraph.num_edges(etype) for etype in subgraph.canonical_etypes)}\n"
            
            # 添加子图的节点信息
            prompt += "- Subgraph Nodes:\n"
            for ntype in subgraph.ntypes:
                nodes_data = subgraph.nodes[ntype].data  # 获取节点数据
                for node_id in range(subgraph.num_nodes(ntype)):
                    node_text = nodes_data.get("text", [""])[node_id]  # 获取节点文本
                    prompt += f"  - Node Type: {ntype}, Node ID: {node_id}, Text: {node_text}\n"
    
            # 添加子图的边信息
            prompt += "- Subgraph Edges:\n"
            for etype in subgraph.canonical_etypes:
                src_type, edge_type, dst_type = etype
                edges_data = subgraph.edges[etype].data  # 获取边数据
                for edge_id in range(subgraph.num_edges(etype)):
                    edge_text = edges_data.get("text", [""])[edge_id]  # 获取边文本
                    prompt += f"  - Edge Type: {src_type} -> {edge_type} -> {dst_type}, Edge ID: {edge_id}, Text: {edge_text}\n"
        else:
            prompt += "\n No related Subgraphs\n"

        # 添加总结指令
        prompt += "\nBased on the above information, please summarize the user query and related knowledge graph content:\n"

        # 调用语言模型生成总结（这里假设使用 OpenAI GPT）
        # 调用外部传入的 LLM 生成总结
        summary = self.llm(prompt)
        return summary

    def get_final_answer(self, query: str, summary: str) -> str:
        # 构建 prompt
        prompt = PromptTemplate(
            input_variables=["query", "summary"],
            template=prompt_templates['final_answer'])
        # 调用语言模型生成最终答案（这里假设使用 OpenAI GPT）
        # 调用外部传入的 LLM 生成总结
        chain = prompt | self.llm
        final_answer = chain.invoke({"query": query, "summary": summary})
        return final_answer

    def request_flask(self, data, url='http://localhost:5000/api/graph-retrieval'):
        # 使用 json 参数发送 JSON 数据
        response = requests.post(url, json=data)
        return response.text

    # 改进后的代码片段
    def run(self, subquery: str, selected_metapaths: List[str], reason_history: str, run_config:RunnableConfig=RunnableConfig(llm={}), topk=1) -> str:
        selected_metapaths = [list(i) for i in selected_metapaths] # 保证json request
        kg_results = self.request_flask({"subquery": subquery, "selected_metapaths": selected_metapaths, "top_k": topk})# self.query_graph(subquery, selected_metapaths)
        naive_results = ''# self.request_flask({"subquery": subquery, "top_k": topk}, url='http://localhost:5000/api/naive-retrieval')#self.naive_rag_retrieval(subquery, top_k=self.TOPK)

        # 2. 统一生成总结
        # combined_prompt = prompt_templates['combined_summary']
        combined_prompt = PromptTemplate(
            input_variables=["subquery", "kg_results", "naive_results"],
            template=prompt_templates['combined_prompt'])
        chain = combined_prompt | self.llm
        final_answer = chain.invoke({"subquery": subquery, "kg_results": kg_results, "naive_results": naive_results}, config=run_config)
        # 3. 生成最终答案
        # final_answer = self.get_final_answer(subquery, combined_summary)
        return final_answer, kg_results, naive_results









