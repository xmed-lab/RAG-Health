# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : instructions_template.py
# Time       ：12/3/2025 9:25 pm
# Author     ：Any
# version    ：python 
# Description：一些模版指令, copy from src
"""

#######instructions for sft , 感觉也可以融入task信息#######
sft_bfs_instructions = """
Instruction: You are a helpful Retrieve-Augmented Generation (RAG) model. 
Your task is to answer questions by logically decomposing them into clear sub-questions and iteratively addressing each one.

Please follow these guidelines:
1. Use "Follow up:" to introduce each sub-question and "Intermediate answer:" to provide answers. 
2. For each sub-question, decide whether you can provide a direct answer or if additional information is required. 
3. If additional information is needed, state, "Let's search the question in Wikipedia." and then use the retrieved information to respond comprehensively. 
4. If a direct answer is possible, provide it immediately without searching.
"""

# 感觉这里的应该是先retreival, 这里可以预先检索K个meta meta-PATH，查看真实的meta-path有哪些，然后为他们写一个summary。
sft_metapth_desc_instructions = """
    You are a healthcare knowledge graph expert specializing in predictive medicine.
    Your goal is to generate a comprehensive paragraph describing the healthcare prediction value of the entire set of meta-paths I provide. This description should synthesize how these relationship patterns collectively contribute to healthcare prediction capabilities.
    Meta-paths: {meta-path}

    Please provide a cohesive paragraph (approximately 8-12 sentences) that:
    - Summarizes the collective predictive power of these meta-paths for healthcare applications
    - Identifies the main types of clinical predictions these paths enable (e.g., risk assessment, disease progression, treatment response)
    - Explains how the combination of these relationship patterns creates a more comprehensive prediction framework than any single path alone
    - Highlights the most valuable clinical insights that emerge from analyzing these interconnected relationships
    - Describes how healthcare providers or systems could leverage these meta-paths for improved patient outcomes
    - Notes any important data types or measurement points captured across multiple paths

    Your description should provide a holistic view of the predictive capabilities represented by these meta-paths while maintaining clinical relevance and practical applicability.
    """

sft_select_instructions = """
Instruction: You are a helpful Retrieve-Augmented Generation (RAG) model.
Your task is to analyze the given query history and select the most relevant meta-paths from the provided set. These meta-paths should be specifically valuable for generating insights for healthcare prediction tasks.

Query History: {}

Available Meta-paths Set:
{meta-path: description, ...}

Please follow these criteria when selecting meta-paths:
1. Choose meta-paths that directly connect to the healthcare prediction task evident in the query history
2. Prioritize meta-paths that reveal temporal relationships or progression patterns
3. Select meta-paths that link biomarkers or symptoms to outcomes
4. Include meta-paths that capture intervention effects when relevant
5. Consider meta-paths that reveal risk factors or protective factors
6. Limit your selection to the most informative paths (maximum 5-7) to avoid noise


Your final output should be a concise, prioritized list of the most valuable meta-paths for generating healthcare prediction insights based on the query history."""

sft_summary_instructions = """ 
Instruction: You are a helpful Retrieve-Augmented Generation (RAG) model. 
Your task is to generate a comprehensive summary of healthcare prediction insights based on the retrieved knowledge subgraph. This summary should be specifically tailored for healthcare prediction applications.
Knowledge Subgraph: {}

Please follow these guidelines:
1. Extract key relationships and patterns from the knowledge subgraph that are relevant to healthcare predictions
2. Identify potential predictive factors and their correlations
3. Highlight any temporal trends or causal relationships
4. Summarize the clinical significance of the identified patterns
5. Note any limitations or gaps in the knowledge representation

The summary should be concise yet comprehensive, focusing on actionable insights that can improve healthcare prediction models.
"""

sft_answer_instructions = """
## Context
Instruction: You are an AI assistant tasked with predicting a patient's next {} based on their original query and summary information. 
Your prediction should be evidence-based and clinically relevant.

Original patient query: {}
Patient summary: {}
Task to predict: {}

## Instructions
1. Carefully analyze the provided patient query and summary.
2. Identify key clinical indicators, symptoms, treatment history, and relevant patterns.
3. Consider standard clinical pathways and best practices for this type of patient.
4. Generate a prediction for the patient's next {task} that includes:
   - Most likely scenario
   - Key factors influencing this prediction
   - Timeline (if applicable)
   - Any potential complications or variables that might alter this prediction

## Output Format
Please structure your response as follows:
1. Prediction summary (2-3 sentences)
2. Key factors supporting this prediction
3. Recommended monitoring points
4. Confidence level (high/medium/low) with brief explanation

## Important Considerations
- Maintain clinical accuracy and avoid speculation beyond available data
- Account for patient-specific factors when making predictions
- Note any areas where additional information would significantly improve prediction quality
- Consider standard of care guidelines relevant to this patient's condition
"""

base_instructions = """
"You are an AI assistant tasked with predicting a patient's next {task_info} based on Patient Information and Task Description." \n
"Follow these rules:\n"
"1. Your prediction should be evidence-based and clinically relevant. \n"

"Patient Information: Below is the EHR entities of the patient with multiple visits.\n"
- Disease History: {disease_info} \n
- Procedure History: {procedure_info} \n
- Prescription History: {prescription_info} \n

"Task Description: {task_desc} \n"

"Output Format:"
"Now, based on original patient query and task description, {question_answer_format}"

"Answer: "
"""
# - Prescription History: {prescription_info} \n # 太耗时了。
# # "2. All predictions must trace to concrete EHR entities. \n"

base_instructions_summary = """
"You are an AI assistant tasked with {task_info} a clinical note based on the initial clinical note provided." \n
"Follow these rules:\n"
"1. Your summary should be clear, concise, and capture all essential details. \n"
"2. Focus on key symptoms, diagnosis, treatment plan, and relevant medical history. \n"

"Initial Clinical Note: {initial_clinical_note} \n"

"Task Description: {task_desc} \n"

"Output Format:"
"Now, based on the initial clinical note, {question_answer_format}"

"Summary:"
"""

base_instructions_qa = """
"You are an AI assistant tasked with {task_info} a clinical note based on the initial clinical note provided." \n
"Follow these rules:\n"
"1. Answer the questions clearly and concisely based on the information in the clinical note.\n"
"2. Ensure that your answers directly address the questions and are supported by the relevant details from the note.\n"

"Initial Clinical Note: {initial_query} \n"

"Task Description: {task_desc} \n"

"Output Format:"
"Now, based on the initial clinical note, {question_answer_format}"

"Answers:"
"""

base_instructions_mqa = """
"You are an AI assistant tasked with {task_info} a clinical note based on the initial clinical note provided." \n
"Follow these rules:\n"
"1. Select the correct options for the multiple-choice questions based on the clinical note.\n"
"2. Provide brief explanations for your selected options to clarify your reasoning.\n"

"Initial Clinical Note: {initial_query} \n"

"Task Description: {task_desc} \n"

"Output Format:"
"Now, based on the initial clinical note, {question_answer_format}"

"Selected Options: <Your Selected Options>"
"""

phenotype_ids = """
'1. Hypertension', '2. Diabetes', '3. Heart Disease', '4. Asthma', '5. Cancer', '6. Obesity', '7. Depression', '8. Arthritis', '9. Alzheimer's', '10. Stroke'
"""

metapth_desc_instructions = """
    You are a healthcare knowledge graph expert specializing in predictive medicine.
    Your goal is to generate a comprehensive paragraph describing the healthcare prediction value of the entire set of meta-paths I provide. This description should synthesize how these relationship patterns collectively contribute to healthcare prediction capabilities.
    Meta-paths: {meta-path}

    Please provide a cohesive paragraph (approximately 8-12 sentences) that:
    - Summarizes the collective predictive power of these meta-paths for healthcare applications
    - Identifies the main types of clinical predictions these paths enable (e.g., risk assessment, disease progression, treatment response)
    - Explains how the combination of these relationship patterns creates a more comprehensive prediction framework than any single path alone
    - Highlights the most valuable clinical insights that emerge from analyzing these interconnected relationships
    - Describes how healthcare providers or systems could leverage these meta-paths for improved patient outcomes
    - Notes any important data types or measurement points captured across multiple paths

    Your description should provide a holistic view of the predictive capabilities represented by these meta-paths while maintaining clinical relevance and practical applicability.
    """


# 缺少output format的约束
task_templates = {
    # qa可能需要单选或者多选
    'mqa': base_instructions_mqa.format(task_info='mqa',
                                       initial_query="{query}",
                                       task_desc="Answer the multiple-choice questions based on the query. Provide explanations for the selected options and ensure clarity in your reasoning.",
                                       question_answer_format="choose the correct options for the multiple-choice questions based on the clinical note. be clear and concise."
                                       ),
    'qa': base_instructions_qa.format(task_info='qa',
                                       initial_query="{query}",
                                       task_desc="Answer the multiple-choice questions based on the query. Provide explanations for the selected single option and ensure clarity in your reasoning.",
                                       question_answer_format="choose a correct option from [A, B, C, D] for the multiple-choice questions based on the clinical note. be clear and concise." # 这个QA除了选择题目，要换
                                       ),
#     task_desc = "Answer the question based on the query. Provide explanations for your answer and ensure clarity in your reasoning.",
# question_answer_format = "Answer the question directly based on the clinical note. Ensure your response is clear, concise, and well-reasoned."
    'summary': base_instructions_summary.format(task_info="summary",
                                           initial_clinical_note="{query}",
                                           task_desc="Summarize the clinical note by extracting key information about the patient's symptoms, diagnosis, treatment plan, and any relevant medical history. Provide a concise summary that captures the main points in a clear and organized manner.",
                                           question_answer_format="give the summary of the clinical note. be short and concise. "),
    'MOR': base_instructions.format(task_info="mortality",
                                           disease_info="{disease_info}", procedure_info="{procedure_info}",
                                           prescription_info="{prescription_info}",
                                           task_desc="Assess the patient's overall health condition, considering the patient's disease history, procedures history, and prescription history to determine the likelihood of mortality within 24 hours. If you determine that the patient is at significant risk of mortality, respond with 'yes'; otherwise, respond with 'no'.",
                                           question_answer_format="predict the patient's mortality risk. give 'yes' or 'no' "),
    'IHM': base_instructions.format(task_info="mortality",
                                    disease_info="{disease_info}", procedure_info="{procedure_info}",
                                    prescription_info="{prescription_info}",
                                    task_desc="Assess the patient's overall health condition, considering the patient's disease history, procedures history, and prescription history to determine the likelihood of mortality during the hospital. If you determine that the patient is at significant risk of mortality, respond with 'yes'; otherwise, respond with 'no'.",
                                    question_answer_format="predict the patient's mortality risk. give 'yes' or 'no' "),
    'REA': base_instructions.format(task_info="readmission",
                                            disease_info="{disease_info}", procedure_info="{procedure_info}",
                                            prescription_info="{prescription_info}",
                                            task_desc="Predict the likelihood of the patient being readmitted to the hospital within the next 30 days based on their current health status and medical history. If you predict that the patient is likely to be readmitted, respond with 'yes'; otherwise, respond with 'no'.",
                                            question_answer_format="predict the patient's readmission risk. give 'yes' or 'no' "),
    'LOS': base_instructions.format(task_info="length of stay",
                                               disease_info="{disease_info}", procedure_info="{procedure_info}",
                                               prescription_info="{prescription_info}",
                                               task_desc="Estimate the expected length of stay for the patient's next hospital visit based on their current health status and medical history. Provide a numerical estimate in days and categorize it according to the following criteria: 0 for ICU stays shorter than a day, 1-7 for each day of the first week, 8 for stays of over one week but less than two, and 9 for stays of over two weeks.",
                                               question_answer_format="estimate the patient's length of stay in days."),
    'PHE': base_instructions.format(task_info="phenotyping",
                                            disease_info="{disease_info}", procedure_info="{procedure_info}",
                                            prescription_info="{prescription_info}",
                                            task_desc="Identify the patient's primary disease phenotype based on their current health status, medical history, and recent procedures. Select the appropriate phenotype ID from the following list: {phenotype_ids}. Provide only the selected phenotype ID as your answer.",
                                            question_answer_format="identify the patient's primary disease phenotype."),
}

prompt_templates = {
    'rewrite': "You are an expert in query rewriting. Your task is to rewrite the following healthcare-related query into 3-5 clear and distinct alternative versions.\n"
               "Follow these rules:\n"
                "1. Keep key patient information.\n\n"
               "2. Each rephrased query should convey the same core meaning as the original query. Ensure semantic consistency.\n\n"

               # "Examples:\n"
               # "Query: 'What are the health benefits of regular check-ups?'\n"
               # "   Subqueries:\n"
               # "   - What are the benefits of regular health screenings?\n"
               # "   - How do check-ups contribute to early disease detection?\n"
               # "   - What is the recommended frequency of health check-ups for adults?\n"
               # 
                "Output Format: List of rewrited queries."

               "Now, rewrite the following query:\n"
               "Query: {query}\n\n"
               "Rewrite queries:",

    'follow': "You are an expert in query generation. Your task is to generate follow-up questions that address knowledge gaps and consider the reasoning history necessary to ultimately answer the healthcare-related query. "
              "Follow these rules:\n"
              "1. Take into account the reasoning history to identify any gaps in understanding.\n\n"
              
              "Output Format: a question.\n"
              
              # "Example:\n"
              # "Query: What are the long-term effects of hypertension on kidney function? \n"
              # "Reason History: Hypertension can damage blood vessels and affect kidney health, but specific effects are unclear.\n"
              # "Follow-up Question:\n"
              # "What specific complications arise from kidney damage due to long-term hypertension?\n\n"

              "Now, generate follow-up questions for the following query:\n"
              "Query: {query}\n\n"
              "Reason History: {reason_history}\n\n"
              "Follow-up Question:",

    "decide": "You are an expert in query analysis. Your task is to decide whether the following healthcare subquery requires external knowledge retrieval or can be answered directly by an LLM. "
              "Follow these rules:\n"
              "1. If the subquery can be answered with general knowledge or simple reasoning, respond with 'no'.\n"
              "2. If the subquery requires specific medical knowledge, external data, or complex reasoning, respond with 'yes'.\n"
              "3. Provide a confidence score (0 to 1) for your decision, where 1 means highly confident and 0 means not confident at all.\n"
              "4. Use the reason history to support your answer with evidence.\n"
              "5. If the reason history does not contain enough information, state that the answer is incomplete and suggest additional sources.\n\n"
              
              "Example:\n"
              "Subquery: What are the common side effects of aspirin?\n"
              "Reason History: Aspirin is widely used for pain relief and has known side effects.\n"
              "Decision: no, Confidence Score: 0.9\n\n"
              
              
              "Now, analyze the following healthcare subquery:\n"
              "Subquery: {subquery}\n\n"
              "Reason History: {reason_history}\n"
              "Decision:",
    
# 这个我也不知道要不要和decide合起来。if merge, we will get uncertainty; otherwise we will use different prompt
    'direct_answer': "You are an expert in answering healthcare queries. Your task is to generate a final answer based on the user query and the provided reason history.\n\n"
                   "Follow these rules:\n"
                   "1. Ensure the answer directly addresses the user query.\n"
                   "2. Use the reason history to support your answer with evidence.\n"
                     
                   "Output Format: Please be short and coherent.\n\n"
                     
                   # "Example:\n"
                   # "Query: What are the benefits of regular exercise for heart health?\n"
                   # "Reason History:  Regular exercise can improve cardiovascular fitness, lower blood pressure, and reduce cholesterol levels.\n"
                   # "Answer: Regular exercise significantly benefits heart health by improving cardiovascular fitness, lowering blood pressure, and reducing cholesterol levels. \n\n"
                   #   
                   "Now, answer the following query based on the reason history:\n"
                   "Query: {subquery}\n"
                   "Reason History: {reason_history}\n"
                   "Answer:",

    'meta_path': "You are an expert in meta-path analysis. Your task is to select three most relevant meta-paths for a given healthcare subquery based on the provided meta-path descriptions and reason history.\n\n"
                     "Follow these rules:\n"
                     "1. Analyze the subquery, reason history, and the provided meta-path descriptions carefully.\n"
                     # "2. Use the reason history to support your selection with evidence.\n"
                     "2. Output Format: List of selected ID numbers. Do not return any explanation.\n\n"
                 
                     # "Example:\n"
                     # "Subquery: What are the effects of diabetes on cardiovascular health?\n"
                     # "Reason History: Previous studies indicate a strong correlation between diabetes and heart disease.\n"
                     # "Meta-path Descriptions:\n"
                     # "Meta-path 1: Diabetes → Cardiovascular Disease\n"
                     # "Meta-path 2: Diabetes → Risk Factors → Cardiovascular\n"
                     # "...(too long, all meta-paths)\n\n"
                     # "Selected Meta-path IDs: [1, 2]\n\n"
                 
                     "Now, select three most relevant meta-paths for the following healthcare subquery:\n"
                     "Subquery: {subquery}\n"
                     "Reason History: {reason_history}\n"
                     "Meta-path Descriptions: {meta_path}\n"
                     "Selected Meta-path IDs:",

    'complete_checking': "You are an expert in evaluating reasoning completeness. Your task is to determine whether the provided reason history fully addresses the healthcare query.\n\n"
            "Follow these rules:\n"
            "1. Assess the query and the follow-up subquery-answer carefully.\n"
            "2. If the reason history fully answers the query, respond with the final answer.\n"
            "3. If it is incomplete or uncertain, respond with 'incomplete.'\n\n"
                         
            # "Example:\n"
            # "Query: ... Will this patient be readmitted to the hospital within 15 days?\n"
            # "Reason History: ... (too long, historical QA pair)\n"
            # "Answer: incomplete.\n\n"
            #         
            # "Query: ... How long will he live in the hospital?\n"
            # "Reason History: ... (too long, historical QA pair)\n"
            # "Answer: 5 days\n\n"
            #              
            "Now, evaluate the following:\n"
            "Query: {query}\n"
            "Reason History: {reason_history}\n"
            "Answer:",


    'combined_prompt': "You are an expert in answering healthcare queries. Your task is to generate a final answer based on the user query and the provided retrieved results.\n\n"
                       "Follow these rules:\n"
                       "1. Ensure the answer directly addresses the user query.\n"
                       "2. Use retrieval methods to support your answer with evidence.\n"
                       "3. Output Format: Please be short and coherent.\n\n"
                       
                       # "Example:\n"
                       # "Query: What are the benefits of a balanced diet for overall health?\n"
                       # "Knowledge Graph Results:\n"
                       # "Nodes: Balanced Diet, Nutrients….\n"
                       # "Edges: (Balanced Diet → Provides → Nutrients)...\n"
                       # # "Subgraph: ... (too long, a lot of triples)\n"
                       # # "Naive RAG Results: A balanced diet can lower the risk of chronic diseases and enhance mental well-being.\n"
                       # "Answer: A balanced diet is vital for overall health as it provides essential nutrients and supports immune function while helping maintain a healthy weight.\n\n"
                       
                       "Now, answer the following query based on the retrieval:\n"
                       "Query: {subquery}\n"
                       "Knowledge Graph Results:\n {kg_results}\n"
                       # "Naive RAG Results: {naive_results}\n"
                       "Answer:",
}





################### 下面是baseline对应的rag措施 ######
kare_prompt ={
    'combined_prompt': "You are an expert in answering healthcare queries. "
                       "Your task is to summarize the provided retrieval entity knowledge graph results and retrieval community results to extract content that is most relevant to the user query.\n"
                       "Follow these rules:\n"
                       "1. Ensure the answer directly addresses the user query.\n"
                       "2. Use retrieval methods to support your answer with evidence.\n\n"
                       
                       "Example:\n"
                       "Query: What are the common symptoms of asthma?\n"
                       "Knowledge Graph Results:\n"
                       "Nodes: Asthma, Symptoms...\n"
                       "Edges: (Asthma → Causes → Allergens)...\n"
                       "Community Results: Symptoms include wheezing, shortness of breath, chest tightness, and coughing.\n"
                       "Answer: Common symptoms of asthma include wheezing, shortness of breath, chest tightness, and coughing.\n\n"
                       
                       "Now, summarize the following knowledge graph results based on the query:\n"
                       "Query: {subquery}\n"
                       "Knowledge Graph Results: {kg_results}\n"
                       "Community Results: {com_results}\n"
                       "Answer:",

    'community': "You are an expert in processing community description extraction from clustered medical node information. "
                "Your task is to generate a concise and informative description of the community based on the provided clustered node information.\n" 
                "Follow these rules:\n" 
                "1. Focus on key characteristics and themes that define the community.\n" 
                "2. Ensure that the description is clear and relevant, highlighting important aspects.\n" 
                "3. Eliminate irrelevant or redundant information from the cluster paragraph.\n\n" 
                
                "Example:\n" 
                "Cluster Paragraph: This community includes individuals with diabetes, focusing on managing blood sugar levels, dietary choices, and exercise. Members share experiences related to insulin use, medication adherence, and lifestyle changes. Many are interested in the latest research on diabetes management and seek support in navigating challenges.\n" 
                "Community Description: This community comprises individuals with diabetes who focus on managing their condition through dietary choices, exercise, and medication adherence. Members share experiences and seek support while staying informed about the latest research in diabetes management.\n\n" 
                
                "Now, refine the following cluster paragraph to create a cohesive community description:\n" 
                "Cluster Paragraph: {cluster_paragraph}\n" 
                "Community Description:",
}

medrag_prompt = {
    'combined_prompt':   "You are an expert in answering healthcare queries. Your task is to generate a final answer based on the user query and the provided retrieved results.\n\n"
                   "Follow these rules:\n"
                   "1. Ensure the answer directly addresses the user query.\n"
                   "2. Use retrieval methods to support your answer with evidence.\n"
                   "3. Output Format: Please be short and coherent.\n\n"
                   
                   "Example:\n"
                   "Query: What are the differences in symptoms between Type 1 and Type 2 diabetes?\n"
                   "Knowledge Graph Results:\n"
                   "Nodes: Type 1 Diabetes, Type 2 Diabetes, Symptoms….\n"
                   "Edges: (Type 1 Diabetes → Symptoms → Increased Thirst)...\n"
                   # "Subgraph: ... (too long, a lot of triples)\n"
                   "Disease Differences: Type 1 typically presents with rapid onset of symptoms, while Type 2 often has gradual onset and may be asymptomatic initially.\n"
                   "Answer: Type 1 diabetes usually has a rapid onset of symptoms such as increased thirst and frequent urination, whereas Type 2 diabetes often has a gradual onset and may not show symptoms initially.\n\n"
                   
                   "Now, answer the following query based on the retrieval:\n"
                   "Query: {subquery}\n"
                   "Knowledge Graph Results: {kg_results}\n"
                   "Disease Differences: {disease_results}\n"
                   "Answer:",

    'disease_gen': "Please generate a comprehensive academic description of a disease. The description should include the following:\n"
                "1. **Symptoms**: Outline common symptomatology, potential manifestations, and the experiences patients may encounter.\n"
                "2. **Geographic Distribution**: Detail the regions most affected by the disease, including environmental factors that contribute to its spread, as well as demographic and socioeconomic elements influencing its prevalence.\n"
                "3. **Locations of Outbreaks**: Discuss various locations where the disease is commonly found or has significant outbreaks. Include information about geographical regions, environmental factors, and any relevant cultural or societal aspects that may influence the spread of the disease.\n\n"
                
                "Example:\n"
                "Disease: Diabetes Mellitus\n"
                "1. **Symptoms**: Common symptoms include increased thirst, frequent urination, extreme fatigue, and blurred vision. Patients may also experience slow healing of wounds and frequent infections.\n"
                "2. **Geographic Distribution**: Diabetes is a global health issue, with higher prevalence in urban areas and regions with a sedentary lifestyle. Environmental factors such as diet and physical activity greatly influence its spread.\n"
                "3. **Locations of Outbreaks**: Significant clusters of diabetes cases are often found in developed countries, particularly in urban centers with high obesity rates. Cultural factors, including dietary habits and healthcare access, affect disease management and prevalence.\n\n"
                
                "Now, generate a comprehensive academic description of the disease: {disease}.\n"
                "Answer:",

    'disease_diff': "Given the following list of diseases, generate a series of tuples that describe their relationships and differences.\n" 
                    "Each tuple should contain three elements: \n" 
                    "1. A brief description of the similarities between the diseases.\n" 
                    "2. A brief description of the differences between the diseases.\n" 
                    "3. Any relevant factors that may influence their connections, such as causative agents, symptoms, or geographic distribution.\n" 
                    "Ensure the language is clear, precise, and suitable for an academic context.\n"
                    "Now, give the list of triples,\n"
                    "Diseases List: {disease}\n"
                    "Triples: "
}

lightrag_prompt = {
    'combined_prompt': "You are an expert in answering healthcare queries. Your task is to generate a final answer based on the user query and the provided retrieved results.\n\n"
                       "Follow these rules:\n"
                       "1. Ensure the answer directly addresses the user query.\n"
                       "2. Use retrieval methods to support your answer with evidence.\n"
                       "3. Output Format: Please be short and coherent.\n\n"
                       
                       "Example:\n"
                       "Query: What are the risk factors for developing cardiovascular disease?\n"
                       "Knowledge Graph Results:\n"
                       "Nodes: Cardiovascular Disease, Risk Factors….\n"
                       "Edges: (Cardiovascular Disease → Associated With → Hypertension)...\n"
                       # "Subgraph: ... (too long, a lot of triples)\n"
                       "Concept Knowledge Graph Results:(too long, a lot of triples).\n"
                       "Answer: Key risk factors for developing cardiovascular disease include high blood pressure, high cholesterol, smoking, obesity, and diabetes.\n\n"
                      
                       "Now, answer the following query based on the retrieval:\n"
                       "Query: {subquery}\n"
                       "Knowledge Graph Results: {kg_results}\n"
                       "Concept Knowledge Graph Results: {coarse_results}\n"
                       "Answer:",


    'concept_extract':"You are a medical concept extraction specialist with expertise in healthcare terminology. "
                    "Your task is to identify and extract medical concepts, conditions, procedures, or therapeutic areas from the user's query. "
                    "Focus on general medical ideas, diseases, treatment categories, or physiological processes, rather than specific patient names or institutional details.\n"
                    "Follow these rules:\n"
                    "1. Prioritize concepts from standard medical taxonomies (e.g., ICD-10, SNOMED CT), such as 'diabetes mellitus', 'chemotherapy', 'inflammatory response'.\n"
                    "2. Ignore specific patient identifiers (e.g., 'Patient X') or non-medical terms unless they modify a medical concept (e.g., 'pediatric asthma' → 'asthma' as the core concept).\n"
                    "3. Normalize synonyms to their standard medical terms (e.g., 'high blood pressure' → 'hypertension').\n"
                    "4. Use commas as separators.\n"
                    
                    "Example:\n"
                    "Query: The patient diagnosed with hypertension and diabetes mellitus is undergoing chemotherapy for cancer treatment.\n"
                    "Extracted Medical Concepts: hypertension, diabetes mellitus, chemotherapy, cancer\n\n"
                    
                    "Now, extract medical concepts from the following query:\n"
                    "Query: {query}\n"
                    "Extracted Medical Concepts:",

    'entity_extract': "You are a healthcare named entity recognition (NER) expert specialized in clinical data. "
                        "Your task is to identify and extract specific medical entities from the user's query.\n"
                        "Follow these rules:\n"
                        "1. Classify entities into standardized medical categories (e.g., DISEASE, DRUG, PROCEDURE, ANATOMY).\n"
                        "2. Include trade names or generic names of medications, but specify their category (e.g., 'Lipitor' [DRUG] → 'atorvastatin' [GENERIC DRUG]).\n"
                        "3. Exclude general terms without clear medical context (e.g., 'patient' unless part of a specific entity like 'ICU patient').\n"
                        "4. Use commas as separators.\n"
                        
                        "Example:\n"
                        "Query: The patient was prescribed Lipitor for high cholesterol and will undergo a coronary artery bypass graft.\n"
                        "Extracted Medical Entities (with Categories): Lipitor [DRUG], atorvastatin [GENERIC DRUG], high cholesterol [DISEASE], coronary artery bypass graft [PROCEDURE]\n\n"
                        
                        "Now, extract medical entities from the following query:\n"
                        "Query: {query}\n"
                        "Extracted Medical Entities (with Categories):",
}


cot_prompt = {
    "decide": "You are an expert in query analysis. Your task is to decide whether the following healthcare subquery requires external knowledge retrieval or can be answered directly by an LLM.\n\n"
                 "Follow these rules:\n"
                 "1. If the subquery can be answered with general knowledge or simple reasoning, respond with 'no'.\n"
                 "2. If the subquery requires specific medical knowledge, external data, or complex reasoning, respond with 'yes'.\n"
                 "3. Provide a confidence score (0 to 1) for your decision, where 1 means highly confident and 0 means not confident at all.\n\n"
              
                 "Examples:\n"
                 "Query: What are the common symptoms of influenza?\n"
                 "Decision: no, Confidence Score: 0.9\n"
                 "Query: What are the latest treatment guidelines for managing hypertension in elderly patients?\n"
                 "Decision: yes, Confidence Score: 0.95\n\n"
              
                 "Now, analyze the following healthcare subquery:\n"
                 "Subquery: {subquery}\n"
                 "Decision:",

    'combined_prompt': "You are an expert in answering healthcare queries. Your task is to generate a final answer based on the user query and the provided retrieved results.\n\n"
                       "Follow these rules:\n"
                       "1. Ensure the answer directly addresses the user query.\n"
                       "2. Use retrieval methods to support your answer with evidence.\n"
                       "3. Output Format: Please be short and coherent.\n\n"
                       
                       "Example:\n"
                       "Query: What are the benefits of a balanced diet for overall health?\n"
                       "Knowledge Graph Results:\n"
                       "Nodes: Balanced Diet, Nutrients….\n"
                       "Edges: (Balanced Diet → Provides → Nutrients)...\n"
                       "Answer: A balanced diet is vital for overall health as it provides essential nutrients and supports immune function while helping maintain a healthy weight.\n\n"
                       
                       "Now, answer the following query based on the retrieval:\n"
                       "Query: {subquery}\n"
                       "Knowledge Graph Results:\n {kg_results}\n"
                       "Answer:",
}
#"Naive RAG Results: {naive_results}\n"
#"Subgraph: ... (too long, a lot of triples)\n"
#"Naive RAG Results: A balanced diet can lower the risk of chronic diseases and enhance mental well-being.\n"

purellm_template = {
    'combined_prompt': "You are an expert in answering healthcare queries. Your task is to generate a final answer based on the user query, which includes the patient's historical context. "
                       "Follow these rules:\n"
                       "1. Ensure the answer directly addresses the user query.\n"
                       "2. Use retrieval methods to support your answer with evidence.\n"
                       
                       "Example:\n"
                       "Query: Will a patient with a history of heart failure and recent pneumonia be readmitted within the next 15 days?\n"
                       "Answer: Given the patient's history of heart failure and recent pneumonia, factors such as recent exacerbations, medication adherence, and the presence of comorbid conditions could increase the risk of readmission within the next 15 days. Studies suggest that patients with these risk factors are often at a higher likelihood for early readmission, necessitating close monitoring and follow-up care.\n\n"
                       
                       "Now, answer the following query:\n"
                       "Query: {subquery}\n"
                       "Answer:",
}
