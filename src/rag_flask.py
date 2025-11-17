import os
import faiss

import asyncio
from langchain.chains import LLMChain
import gc
import threading
import json
import pickle
import torch
import dgl
import random
from typing import List, Optional, Dict, Tuple, Any
from collections import defaultdict
from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

import numpy as np
from dataset import load_graph_data, get_emb_model, load_text_chunk_data, load_user_chunk_data
from concurrent.futures import ThreadPoolExecutor
from config import config
from instructions_template import kare_prompt, medrag_prompt # 这个需要提前准备
from langchain_core.prompts import PromptTemplate
from utils import get_llm_model
from itertools import chain


class AgentLow:
    """单独把index拿出来做服务"""

    def __init__(self, embedding_model, TOPK : int=3):
        """
        初始化GraphRAG数据库

        Args:
            embedding_model: 文本向量化模型
            dim: 向量维度
        """
        self.graph = None
        self.graph_node_types = None # for subgraph
        self.embed_model = embedding_model

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
        # user RAG
        self.user_chunks = []  # 存储所有文本块
        self.user_chunk_store = FAISS.from_texts(['dummy text'], self.embed_model, metadatas=[{"id": -1}])  # 初始化空的 FAISS 索引

        # commuity RAG
        self.com_chunks = []  # 存储所有文本块
        self.com_chunk_store = FAISS.from_texts(['dummy text'], self.embed_model, metadatas=[{"id": -1}])  # 初始化空的 FAISS 索引

        # disease RAG
        self.disease_chunks = []  # 存储所有文本块
        self.disease_chunk_store = FAISS.from_texts(['dummy text'], self.embed_model, metadatas=[{"id": -1}])  # 初始化空的 FAISS 索引


        self.TOPK = TOPK

        self.root_path = '/hpc2hdd/home/xxxs349/xxxc/RAGHealth/data/ready/'  # 设置根路径
        self.vector_store_lock = threading.Lock() # double free现象


    def add_user_chunks(self, chunks: List[str]):
        """
        添加文本块到 Naive RAG 存储中。

        Args:
            chunks: 文本块列表。
        """
        if os.path.exists(self.root_path + "/user_index"):
            # 如果索引已经存在，直接加载
            print("Load existing FAISS index from local storage. If you want to update the index, please delete the existing index first.")
            self.user_chunk_store = FAISS.load_local(self.root_path + "/user_index", self.embed_model, allow_dangerous_deserialization=True)
            document_num = len(self.user_chunk_store.docstore._dict)
        else:
            print("Create FAISS index from scratch. This may take a while for large datasets.")
            # 生成文档对象
            batch_size = 100  # 可以根据实际情况调整批次大小
            document_num = len(chunks)
            for i in range(0, len(chunks), batch_size):
                # 生成当前批次的文档对象
                batch_documents = [
                    Document(page_content=chunk, metadata={"id": i})
                    for i,  chunk in enumerate(chunks[i:i + batch_size])
                ]
                # 添加当前批次的文档到向量存储
                self.user_chunk_store.add_documents(batch_documents)
                # 释放当前批次的内存
                del batch_documents
                gc.collect()

                # 定期保存索引
                if (i // batch_size + 1) % 10 == 0:  # 每处理10个批次保存一次索引
                    self.user_chunk_store.save_local(self.root_path + "/user_index")

            # 最后再保存一次索引，确保所有文档都已保存
            self.user_chunk_store.save_local(self.root_path + "/user_index")

        self.user_chunks.extend(chunks) # no matter how to save; index
        return document_num


    def add_text_chunks(self, chunks: List[str]):
        """
        添加文本块到 Naive RAG 存储中。

        Args:
            chunks: 文本块列表。
        """
        if os.path.exists(self.root_path + "/faiss_index"):
            # 如果索引已经存在，直接加载
            print("Load existing Text index from local storage. If you want to update the index, please delete the existing index first.")
            self.text_chunk_store = FAISS.load_local(self.root_path + "/faiss_index", self.embed_model, allow_dangerous_deserialization=True)
            document_num = len(self.text_chunk_store.docstore._dict)
        else:
            print("Create FAISS index from scratch. This may take a while for large datasets.")
            # 生成文档对象
            batch_size = 100  # 可以根据实际情况调整批次大小
            document_num = len(chunks)
            for i in range(0, len(chunks), batch_size):
                # 生成当前批次的文档对象
                batch_documents = [
                    Document(page_content=chunk, metadata={"id": i})
                    for i,  chunk in enumerate(chunks[i:i + batch_size])
                ]
                # 添加当前批次的文档到向量存储
                self.text_chunk_store.add_documents(batch_documents)
                # 释放当前批次的内存
                del batch_documents
                gc.collect()

                # 定期保存索引
                if (i // batch_size + 1) % 10 == 0:  # 每处理10个批次保存一次索引
                    self.text_chunk_store.save_local(self.root_path + "/faiss_index")

            # 最后再保存一次索引，确保所有文档都已保存
            self.text_chunk_store.save_local(self.root_path + "/faiss_index")

        self.text_chunks.extend(chunks) # no matter how to save; index
        return document_num


    def user_rag_retrieval(self, query: str, top_k: int = 1) -> List[Dict[str, Any]]:
        """
        从文本块中检索与查询相关的片段（Naive RAG）。

        Args:
            query: 查询文本。
            top_k: 返回的最大结果数。

        Returns:
            检索到的文本片段列表，包含文本和相似度分数。
        """
        # 查询向量存储
        retrieved = self.user_chunk_store.similarity_search_with_score(query, k=top_k)
        # 格式化结果
        results = []
        for doc, score in retrieved:
            # 确保 metadata 包含 "id"，否则使用默认值
            chunk_id = doc.metadata.get("id", -1)  # 如果 "id" 不存在，默认值为 -1
            if chunk_id == -1:
                continue  # 跳过 id 为 -1 的结果
            text = self.user_chunks[chunk_id] if chunk_id != -1 else doc.page_content
            results.append({
                "text": text,
                "score": float(score),
                "chunk_id": chunk_id
            })

        return results

    def naive_rag_retrieval(self, query: str, top_k: int = 1) -> List[Dict[str, Any]]:
        """
        从文本块中检索与查询相关的片段（Naive RAG）。

        Args:
            query: 查询文本。
            top_k: 返回的最大结果数。

        Returns:
            检索到的文本片段列表，包含文本和相似度分数。
        """
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


    def com_retrieval(self, query: str, top_k: int = 1) -> List[Dict[str, Any]]:
        """
        从文本块中检索与查询相关的片段（Naive RAG）。

        Args:
            query: 查询文本。
            top_k: 返回的最大结果数。

        Returns:
            检索到的文本片段列表，包含文本和相似度分数。
        """
        # 查询向量存储
        retrieved = self.com_chunk_store.similarity_search_with_score(query, k=top_k)
        # 格式化结果
        results = []
        for doc, score in retrieved:
            # 确保 metadata 包含 "id"，否则使用默认值
            chunk_id = doc.metadata.get("id", -1)  # 如果 "id" 不存在，默认值为 -1
            if chunk_id == -1:
                continue  # 跳过 id 为 -1 的结果
            text = self.com_chunks[chunk_id] if chunk_id != -1 else doc.page_content
            results.append({
                "text": text,
                "score": float(score),
                "chunk_id": chunk_id
            })

        return results



    def disease_retrieval(self, query: str, top_k: int = 1) -> List[Dict[str, Any]]:
        """
        从文本块中检索与查询相关的片段（Naive RAG）。

        Args:
            query: 查询文本。
            top_k: 返回的最大结果数。

        Returns:
            检索到的文本片段列表，包含文本和相似度分数。
        """
        # 查询向量存储
        retrieved = self.disease_chunk_store.similarity_search_with_score(query, k=top_k)
        # 格式化结果
        results = []
        for doc, score in retrieved:
            # 确保 metadata 包含 "id"，否则使用默认值
            chunk_id = doc.metadata.get("id", -1)  # 如果 "id" 不存在，默认值为 -1
            if chunk_id == -1:
                continue  # 跳过 id 为 -1 的结果
            text = self.disease_chunks[chunk_id] if chunk_id != -1 else doc.page_content
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


    def cluster_comm(self, texts, embs, k_community=300): # 1000个社区， 这里暂时用node吧
        llm = get_llm_model(config, config['T_LLM'], api_base=None, path=config['T_LLM_PATH']) # 最好使用相似的

        d = embs.shape[1]  # 向量维度
        kmeans = faiss.Kmeans(d, k_community, niter=20, verbose=False)
        kmeans.train(embs)
        print("cluster XXXXX",embs.shape, len(texts))
        _, labels = kmeans.index.search(embs, 1)
        labels = labels.flatten()

        # 3. 分组聚合文本（按聚类标签分组）
        clusters = defaultdict(list)
        for label, text in zip(labels, texts):
            clusters[label].append(text)

        # 4. 生成社区描述（基于LLM）
        community_descriptions = []
        for cluster_texts in clusters.values():
            # 合并簇内文本为段落
            # 如果 cluster_texts 的文本数量大于 1000，进行拆分
            segments = []

            if len(cluster_texts) > 1000:
                # 将 cluster_texts 拆分为多个部分
                for i in range(0, len(cluster_texts), 1000):
                    segments.append(cluster_texts[i:i + 1000])
            else:
                segments.append(cluster_texts)  # 保持原样

            # 生成描述
            print("All segments number: ", len(segments))
            for segment in segments:
                cluster_paragraph = ', '.join(segment)
                combined_prompt = PromptTemplate(
                    input_variables=['cluster_paragraph'],
                    template=kare_prompt['community'])
                chain = combined_prompt | llm
                description = chain.invoke({"cluster_paragraph": cluster_paragraph})
                community_descriptions.append(description)

        return community_descriptions


    def index_community(self, k_community=300):
        # 生成community描述
        if os.path.exists(self.root_path + "/community_index"):
            # 如果索引已经存在，直接加载
            print(
                "Load existing Community index from local storage. If you want to update the index, please delete the existing index first.")
            self.com_chunk_store = FAISS.load_local(self.root_path + "/community_index", self.embed_model,
                                                     allow_dangerous_deserialization=True)
            with open(self.root_path +'com_descr.pkl', 'rb') as f:
                data = pickle.load(f)
            self.com_chunks.extend(data)
            document_num = len(self.com_chunk_store.docstore._dict)
        else:
            print("Create FAISS index from scratch. This may take a while for large datasets.")
            all_text = []
            all_embs = []
            for node_type, texts in self.node_text_dict.items():
                all_text.extend(texts.values())  # 将所有节点的文本合并
                embs = self.embed_model.embed_documents(texts.values())  # 获取所有节点的嵌入向量
                print("Process embds  ",len(texts), len(embs))
                all_embs.extend(embs)  # 将所有节点的嵌入向量合并
            all_embs = np.array(all_embs)

            # 生成相似对象
            descriptions = self.cluster_comm(all_text, all_embs, k_community)
            with open(self.root_path + 'com_descr.pkl', 'wb') as f:
                pickle.dump(descriptions, f)

            # 生成文档对象
            documents = [
                Document(page_content=text, metadata={"id": id})
                for id, text in enumerate(descriptions)
            ]
            document_num = len(documents)
            self._add_documents_in_batches(self.com_chunk_store, documents)
            # 持久化，不然一次时间巨长
            self.com_chunk_store.save_local(self.root_path + "/community_index")  # 保存索引到本地

            self.com_chunks.extend(descriptions)
        return document_num

    async def process_diseases_batch(self, llm, diseases, template, batch_size=10, max_retries=3):
        async def process_single_disease(disease):
            """处理单个疾病描述的协程函数"""
            for attempt in range(max_retries):
                try:
                    prompt = PromptTemplate(input_variables=["disease"], template=template)
                    chain = LLMChain(llm=llm, prompt=prompt)
                    return await chain.arun(disease=disease)
                except Exception as e:
                    if attempt == max_retries - 1:
                        print(f"疾病 '{disease}' 处理失败: {str(e)}")
                        return None  # 或返回默认值
                    await asyncio.sleep(1 + attempt * 2)  # 指数退避重试

        async def process_batch(batch_diseases):
            """处理一个批次的疾病描述"""
            tasks = [process_single_disease(d) for d in batch_diseases]
            return await asyncio.gather(*tasks)

        # 分批处理所有疾病
        all_descriptions = []
        for i in range(0, len(diseases), batch_size):
            batch = diseases[i:i + batch_size]
            batch_results = await process_batch(batch)
            all_descriptions.extend(batch_results)
            if i % 100 ==0:
                print("XXXXXXX", i)

        return all_descriptions


    def index_disease(self):
        # 生成community描述
        if os.path.exists(self.root_path + "/disease_index"):
            # 如果索引已经存在，直接加载
            print(
                "Load existing Disease index from local storage. If you want to update the index, please delete the existing index first.")
            self.disease_chunk_store = FAISS.load_local(self.root_path + "/disease_index", self.embed_model,
                                                     allow_dangerous_deserialization=True)
            with open(self.root_path +'disease_descr.pkl', 'rb') as f:
                data = pickle.load(f)
            self.disease_chunks.extend(data)
            document_num = len(self.disease_chunk_store.docstore._dict)
        else:
            print("Create FAISS index from scratch. This may take a while for large datasets.")
            all_text = []

            for node_type, texts in self.node_text_dict.items():
                if node_type == 'disease':
                    all_text.extend(texts.values())  # 将所有节点的文本合并

            print("We have {} disease!".format(len(all_text)))
            # 后续改进，可以检索DB中对应的含义description，而不仅仅是使用名字。
            descriptions = []
            llm = get_llm_model(config, config['T_LLM'], api_base=None, path=config['T_LLM_PATH'])  # 最好使用相似的
            # combined_prompt = PromptTemplate(
            #     input_variables=['disease'],
            #     template=medrag_prompt['disease_gen'])
            # chain = combined_prompt | llm
            # for disease in all_text:
            #     description = chain.invoke({"disease": disease})
            #     descriptions.append(description)
            descriptions = asyncio.run(self.process_diseases_batch(llm, all_text, medrag_prompt['disease_gen']))
            descriptions = [d for d in descriptions if d is not None]

            print("Description generate Done!")

            with open(self.root_path + 'disease_descr.pkl', 'wb') as f:
                pickle.dump(descriptions, f)

            # 生成文档对象
            documents = [
                Document(page_content=text, metadata={"id": id})
                for id, text in enumerate(descriptions)
            ]
            document_num = len(documents)
            self._add_documents_in_batches(self.disease_chunk_store, documents)
            # 持久化，不然一次时间巨长
            self.disease_chunk_store.save_local(self.root_path + "/disease_index")  # 保存索引到本地

            self.disease_chunks.extend(descriptions)
        return document_num



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
                Document(page_content=text, metadata={"id": node_id}) # 这里的ID
                for node_id, text in zip(node_ids, texts)
            ]
            document_num = len(documents)

            # 添加到向量存储
            # self.node_stores[node_type].add_documents(documents) # node_type: [index-name pair]
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

    def naive_rag_retrieval(self, query: str, top_k: int = 1) -> List[Dict[str, Any]]:
        """
        从文本块中检索与查询相关的片段（Naive RAG）。

        Args:
            query: 查询文本。
            top_k: 返回的最大结果数。

        Returns:
            检索到的文本片段列表，包含文本和相似度分数。
        """
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


    def node_retrieval(self, query: str, node_types: Optional[List[str]] = None, top_k: int = 1) -> Dict[
        str, List[Dict]]:
        """
        检索与查询语义相关的节点

        Args:
            query: 查询文本
            node_types: 节点类型列表(为None时检索所有类型)
            top_k: 每种类型返回的最大结果数

        Returns:
            按节点类型组织的检索结果（当然也可以完全检索，regardless of ntype）
        """
        results =[]#= {} # []
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
                # type_results.append({
                #     "id": node_id,
                #     "text": text
                # })
                type_results.append(text)
            if type_results:
                results.append(type_results)
                # results[ntype] = type_results
        #
        return results

    def edge_retrieval(self, query: str, edge_types: Optional[List[Tuple[str, str, str]]] = None, top_k: int = 1) -> \
    Dict[Tuple[str, str, str], List[Dict]]:
        """
        检索与查询语义相关的边

        Args:
            query: 查询文本
            edge_types: 边类型列表(为None时检索所有类型)
            top_k: 每种类型返回的最大结果数

        Returns:
            按边类型组织的检索结果
        """
        results = []#{}, 不然会有tuple error
        scores = []
        edge_types = edge_types or list(self.edge_stores.keys())

        for etype in edge_types:
            if etype not in self.edge_stores: # 保证存在
                continue

            # 查询向量存储
            retrieved = self.edge_stores[etype].similarity_search_with_score(query, k=top_k)

            # 格式化结果
            type_results = []
            type_scores = []
            for doc, score in retrieved:
                edge_id = doc.metadata["id"]
                if edge_id == -1:  # 过滤掉虚拟数据
                    continue
                text = self.edge_text_dict[etype].get(edge_id, doc.page_content)
                # type_results.append({
                    # "id": edge_id,
                    # "text": text,
                    # "src_type": etype[0],
                    # "edge_type": etype[1],
                    # "dst_type": etype[2]
                # })
                type_results.append(text)
                type_scores.append(score)
            if type_results:
                # results[etype] = type_results
                results.append(type_results)
                scores.append(type_scores)
        flattened_results = list(chain.from_iterable(results))
        flattened_scores = list(chain.from_iterable(scores))  # 注意这里使用 scores 而非 results
        if config['MODEL'] != 'ours':
            top_k_indices = np.argsort(flattened_scores)[-int(top_k*len(edge_types)* config['META_RATIO']):][::-1] # 保证retrieval个数一致
            results = [flattened_results[i] for i in top_k_indices]
        else:
            results = flattened_results

        return results


    def subgraph_retrieval(self, query: str, max_nodes: int = 20, num_walks: int = 2, walk_length: int = 3,
                           metapaths: List[List[str]] = None, topk=1) -> dgl.DGLGraph:
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
        seed_nodes_results = self.node_retrieval(query, top_k=topk)  # 只选种子节点进行扩展
        seed_nodes = {
            ntype: [item["id"] for item in nodes] # 这里是真的图ID么
            for ntype, nodes in seed_nodes_results.items()
        } # {'user': [0, 1], 'product': [0, 1], 'category': [0, 1]}

        # 初始化节点集合
        nodes_dict = {ntype: set() for ntype in self.graph.ntypes}

        # 获取各类型节点的最大有效 ID
        node_type_max_ids = {ntype: self.graph.number_of_nodes(ntype) - 1 for ntype in self.graph.ntypes}
        # 俺也不知道，为什么会retreival 过大的值,这里要保证小。
        seed_nodes = {ntype: [node for node in nodes if node <= node_type_max_ids[ntype]] for ntype, nodes in seed_nodes.items()}


        print("Type max nodes number", node_type_max_ids)
        print("Cur seed nodes", seed_nodes)

        # 对每种元路径进行随机游走
        for metapath in metapaths: # 并行meta-path采样
            # 获取元路径的起始节点类型
            start_ntype = metapath[0]

            # 获取起始节点类型的种子节点
            if start_ntype not in seed_nodes or not seed_nodes[start_ntype]:
                continue  # 如果没有种子节点，跳过该元路径

            # 将种子节点转换为张量
            start_nodes_tensor = torch.tensor(seed_nodes[start_ntype], dtype=torch.int64)

            # 进行随机游走
            # for _ in range():
            walks, types = dgl.sampling.random_walk(
                self.graph,  # 输入的图
                start_nodes_tensor.repeat(num_walks),  # 种子节点列表
                metapath=[metapath] * walk_length  # 元路径, 因为只有1跳，所以直接重复
            )  # 提取游走路径
            node_types = self.graph_node_types[types.tolist()]  # 获取节点类型
            # 更新节点集合
            for index in range(walks.shape[1]):  # 遍历每条游走路径
                node_id = walks[:, index]
                node_type = node_types[index]

                # 过滤有效节点并更新
                valid_nodes = node_id[node_id != -1].unique()  # 过滤无效节点
                nodes_dict[node_type].update(valid_nodes.tolist())


        # 如果节点数达到 max_nodes，直接返回
        if sum(len(nodes) for nodes in nodes_dict.values()) >= max_nodes:
            # 构建子图, 随机采样max_nodes
            # 将所有键值对展开成一个列表
            all_samples = [(key, value) for key, values in nodes_dict.items() for value in values]

            # 随机选择 3 个样本
            samples = random.sample(all_samples, 3)

            # 使用 defaultdict 和 itertools.groupby 合并相同的 key
            result = defaultdict(list)
            _ = [result[key].append(value) for key, value in samples]

            # 转换为普通字典
            nodes_dict = dict(result)

        # 构建子图
        induced_nodes = {ntype: torch.tensor(list(nodes), dtype=torch.int64) for ntype, nodes in nodes_dict.items() if
                         nodes}
        subgraph = dgl.node_subgraph(self.graph, induced_nodes) # 扩展了其实
        subgraph_dic = {ntype:subgraph.nodes(ntype) for ntype in subgraph.ntypes if subgraph.number_of_nodes(ntype) > 0} # 过滤掉没有节点的类型
        subgraph = {ntype: list(map(self.node_text_dict[ntype].__getitem__, val.tolist())) for ntype, val in subgraph_dic.items()}#dgl.node_subgraph(self.graph, induced_nodes)  # 构建子图
        return subgraph

    def query_graph(self, query: str, selected_metapaths: List[str], topk:int=1) -> Dict[str, Any]:
        # 根据选定的元路径过滤节点和边类型
        node_types = list({node for path in selected_metapaths for node in (path[0], path[2])}) # list(set([mp.split("->")[0] for mp in selected_metapaths if "->" in mp]))
        edge_types = list({path[1] for path in selected_metapaths}) # list(set([tuple(mp.split("->")) for mp in selected_metapaths if "->" in mp]))

        result = {
            # "query": query,
            "nodes": self.node_retrieval(query, node_types=node_types, top_k=topk),
            "edges": self.edge_retrieval(query, edge_types=selected_metapaths, top_k=topk), # 这里必须是meta_path
        }        # 添加子图

        # try: 巨耗时
        #     subgraph = self.subgraph_retrieval(query, walk_length=1, num_walks=2, max_nodes=30, metapaths=selected_metapaths) # 当完全没有的时候，可能会报错，所以限制性下walk length
        #     result["subgraph"] = subgraph # 这个可能暂时用不到，感觉抽取社区会好点。
        # except Exception as e:
        #     print(f"子图检索失败: {e}")
        #     result["subgraph"] = ''
        return result

def index_graph(config):
    embedding_model = get_emb_model(config)
    agent_low = AgentLow(embedding_model)

    # 2. 创建异构图 & text chunk
    root_to_dir = "/hpc2hdd/home/xxxs349/xxxc/RAGHealth/data/ready"  # 这个路径都是公用的
    node_data, graph, metapaths_dic = load_graph_data(root_to_dir)
    meta_paths = [metapaths_dic[i]['raw-meta-path'] for i in metapaths_dic]

    agent_low.load_graph(graph) # 创建vector store

    print('Agent Load Graph Done!')

    # 5. index 节点和边文本 150G memory~1天 for edges
    node_name_list_by_type = (
        node_data.groupby('node_type')
        .apply(lambda x: x.sort_values(by='new_index')['node_name'].tolist())
        .to_dict()
    )  # {nodetype: [node_name]} ,sort by new_index
    for node_type in node_name_list_by_type:
        chosen_lines = node_data[node_data['node_type'] == node_type]['node_name'].tolist()
        agent_low.index_nodes(node_type=node_type, node_ids=list(range(len(chosen_lines))),
                                   texts=chosen_lines)

    # # index edges
    for path in meta_paths:
        edge_ids = graph.edges(etype=path, form='eid')  # tensor
        rel = path[1]
        src_id, tgt_id = graph.edges(etype=path)  # 获取node_id, tensor list
        src_name = np.array(node_name_list_by_type[path[0]])[src_id].tolist()
        tgt_name = np.array(node_name_list_by_type[path[2]])[tgt_id].tolist()
        src_rel_tgt = [f"{src} {rel} {tgt}" for src, tgt in zip(src_name, tgt_name)]
        agent_low.index_edges(edge_type=path, edge_ids=edge_ids, texts=src_rel_tgt)


    # index text chunk,这个可加可不加，生成的时候太慢了。花钱！！！！
    text_chunks = load_text_chunk_data(root_to_dir)
    agent_low.add_text_chunks(text_chunks)


    # index_user
    user_chunks = load_user_chunk_data(root_to_dir)
    agent_low.add_user_chunks(user_chunks)


    # index community # 只对KARE使用, 我傻逼了。
    # agent_low.index_community()


    # index community # 只对medrag使用
    agent_low.index_disease()


    print('Agent Graph initialize Done!')
    return agent_low


# 创建Flask应用
app = Flask(__name__)
CORS(app)  # 启用CORS，允许跨域请求
# config['GPU'] = '5'
rag_service = index_graph(config)
# print("AAAAAAAAAAAAAA") # juest for test
# query = 'diabetes'
# meta_path = [['anatomy', 'anatomy_anatomy', 'anatomy'], ['anatomy', 'anatomy_anatomy', 'anatomy']]
# meta_path = [tuple(i) for i in meta_path]
# a = rag_service.query_graph(query, meta_path)
# print("BBBBBBBB", a)
# print("AAAAAAAAAAAAAA")



@app.route('/api/graph-retrieval', methods=['POST'])
def api_graph_retrieval():
    """API端点：子图检索"""
    try:
        data = request.json
        query = data.get('subquery')
        top_k = data.get('top_k', 1)

        if not query:
            return jsonify({"error": "Query parameter is required"}), 400

        selected_metapaths = data.get('selected_metapaths', '')
        # # selected_metapaths = eval(selected_metapaths)
        # app.logger.info("Selected Metapaths: %s", selected_metapaths) # 要记得重启,不然看不到
        # app.logger.info("Selected type Metapaths: %s", type(selected_metapaths))
        # app.logger.info("Selected len Metapaths: %d", len(selected_metapaths))
        # app.logger.info("Selected len Metapaths: %d", selected_metapaths[1])

        selected_metapaths = [tuple(i) for i in selected_metapaths]
        # app.logger.debug("tupe: %s", selected_metapaths==str(selected_metapaths)) # 重新运行的时候才有效

        results = rag_service.query_graph(
            query, selected_metapaths, top_k)

        return results

    except Exception as e:
        return jsonify({"error": str(e)}), 500



# 在rag_flask.py中添加健康检查路由
@app.route('/health')
def health():
    return "OK", 200


@app.route('/api/naive-retrieval', methods=['POST'])
def api_naive_retrieval():
    """API端点：子图检索"""
    try:
        data = request.json
        query = data.get('subquery')
        if not query:
            return jsonify({"error": "Query parameter is required"}), 400

        top_k = data.get('top_k', 1)

        # app.logger.debug("tupe: %s", selected_metapaths==str(selected_metapaths)) # 重新运行的时候才有效

        results = rag_service.naive_rag_retrieval(
            query, top_k)

        return results

    except Exception as e:
        return jsonify({"error": str(e)}), 500



@app.route('/api/user-retrieval', methods=['POST'])
def api_user_retrieval():
    """API端点：子图检索"""
    try:
        data = request.json
        query = data.get('subquery')

        if not query:
            return jsonify({"error": "Query parameter is required"}), 400

        top_k = data.get('top_k', 1)

        # app.logger.debug("tupe: %s", selected_metapaths==str(selected_metapaths)) # 重新运行的时候才有效

        results = rag_service.user_rag_retrieval(
            query, top_k)

        return results

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/com-retrieval', methods=['POST'])
def api_com_retrieval():
    """API端点：子图检索"""
    try:
        data = request.json
        query = data.get('subquery')
        top_k = data.get('top_k', 1)

        if not query:
            return jsonify({"error": "Query parameter is required"}), 400

        top_k = data.get('top_k', top_k)

        results = rag_service.com_retrieval(
            query, top_k)

        return results

    except Exception as e:
        return jsonify({"error": str(e)}), 500



@app.route('/api/disease-retrieval', methods=['POST'])
def api_disease_retrieval():
    """API端点：子图检索"""
    try:
        data = request.json
        query = data.get('subquery')
        topk = data.get('topk')
        if not query:
            return jsonify({"error": "Query parameter is required"}), 400

        top_k = data.get('top_k', topk)

        results = rag_service.disease_retrieval(
            query, top_k)

        return results

    except Exception as e:
        return jsonify({"error": str(e)}), 500





# 主函数
if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000) # debug==Flase就不会运行两次
