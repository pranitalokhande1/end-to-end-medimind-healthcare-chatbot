# import streamlit as st
# from langchain.vectorstores import FAISS
# from langchain.embeddings import HuggingFaceEmbeddings
# from langchain.llms import Ollama
# from langchain.chains import RetrievalQA
# from langchain.prompts import PromptTemplate
# from langchain.chains.question_answering import load_qa_chain

# # Page setup
# st.set_page_config(page_title="🧠 MediMind: Clinical Decision Support", layout="wide")

# st.title("🧠 MediMind: Clinical Decision Support")
# st.markdown("Provide structured clinical input below to generate differential diagnoses and outpatient treatment recommendations.")

# # Input fields
# context = st.text_area("🔍 Clinical Input\nContext (e.g., Emergency Dept, Primary Care, etc.)")
# demographics = st.text_area("Patient Demographics (e.g., 52-year-old male)")
# symptoms = st.text_area("Symptoms")
# history = st.text_area("Medical History")
# tests = st.text_area("Test Results")
# query = st.text_area("Physician Query")

# # Combine user input
# full_query = f"""
# Context: {context}
# Patient: {demographics}
# Symptoms: {symptoms}
# History: {history}
# Test Results: {tests}
# Query: {query}
# """

# # Load components
# embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
# vectorstore = FAISS.load_local("faiss_index", embeddings=embedding_model, allow_dangerous_deserialization=True)

# retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
# llm = Ollama(model="llama3")

# # Custom prompt (input variable must be "context")
# prompt_template = PromptTemplate(
#     input_variables=["context"],
#     template="""
# You are a clinical decision support assistant. Use outpatient guidelines (ADA, NICE, WHO) to provide a comprehensive diagnosis and treatment plan.

# Clinical Case:
# {context}

# Instructions:
# - ONLY recommend outpatient-safe diagnosis and treatment. If any red flags are detected, mention them clearly.
# - Follow evidence-based outpatient guidelines.
# - Fill in EVERY section below with complete, detailed information.
# - DO NOT skip any section.

# Format your response like this:

# **1. Symptom Analysis**
# - Differential Diagnoses (include % likelihood + 1-line reasoning per item)
# - Red Flags (list if any, else write "None")

# **2. Clinical Decision Support**
# - Immediate Outpatient Actions:
#     • Vitals to check
#     • Labs to order (e.g., HbA1c, CBC, TSH, etc.)
#     • Physical assessments
# - Medications:
#     • Name
#     • Dose
#     • Frequency
#     • Purpose
# - Lifestyle Modifications:
#     • Diet changes (e.g., low glycemic index, reduced carbs)
#     • Exercise (type, intensity, duration per week)
#     • Smoking cessation if applicable
# - Follow-Up Plan:
#     • When to repeat labs
#     • When to schedule next visit
#     • Any specialist referrals needed
# - Documentation Notes:
#     • Summary of findings
#     • Diagnosis and plan in 1-2 sentences
# """
# )

# qa_chain = RetrievalQA.from_chain_type(
#     llm=llm,
#     retriever=retriever,
#     chain_type="stuff",
#     chain_type_kwargs={
#         "prompt": prompt_template,
#         "document_variable_name": "context"
#     },
#     return_source_documents=False
# )

# # Trigger response
# if st.button("💡 Generate Clinical Plan"):
#     if full_query.strip():
#         with st.spinner("Analyzing case and generating response..."):
#             response = qa_chain.run({"query": full_query})
#         st.markdown("### 📝 Clinical Decision Support")
#         st.markdown(response)
#     else:
#         st.warning("Please fill in all the input fields.")



import streamlit as st
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Page setup
st.set_page_config(page_title="🧠 MediMind: Clinical Decision Support", layout="wide")

st.title("🧠 MediMind: Clinical Decision Support")
st.markdown("Provide structured clinical input below to generate differential diagnoses and outpatient treatment recommendations.")

# Input fields
context = st.text_area("🔍 Clinical Input\nContext (e.g., Emergency Dept, Primary Care, etc.)")
demographics = st.text_area("Patient Demographics (e.g., 52-year-old male)")
symptoms = st.text_area("Symptoms")
history = st.text_area("Medical History")
tests = st.text_area("Test Results")
query = st.text_area("Physician Query")

# Combine user input into a single context string
full_context = f"""
Context: {context}
Patient: {demographics}
Symptoms: {symptoms}
History: {history}
Test Results: {tests}
Query: {query}
"""

# Load embedding model and vectorstore
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
vectorstore = FAISS.load_local("faiss_index", embeddings=embedding_model, allow_dangerous_deserialization=True)

retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
llm = Ollama(model="llama3")

# Define prompt template with input variable "context"
prompt_template = PromptTemplate(
    input_variables=["context"],
    template="""
You are a clinical decision support assistant specialized in evidence-based outpatient care using ADA, NICE, and WHO guidelines.

Given the following clinical case, provide a detailed and structured response including:

1. Symptom Analysis:
   - Differential Diagnoses (include % likelihood and brief reasoning)
   - Red Flags (if any, otherwise "None")

2. Clinical Decision Support:
   - Immediate Outpatient Actions:
       • Vitals to check
       • Labs to order
       • Physical assessments
   - Medications (name, dose, frequency, purpose)
   - Lifestyle Modifications (diet, exercise, smoking cessation)
   - Follow-Up Plan (labs, visits, referrals)
   - Documentation Notes (summary and diagnosis/plan)

Clinical Case:
{context}

Format your answer clearly with sections as above.
"""
)

# Build the RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={
        "prompt": prompt_template,
        "document_variable_name": "context",
    },
    return_source_documents=False
)

# Streamlit button trigger
if st.button("💡 Generate Clinical Plan"):
    if full_context.strip():
        with st.spinner("Analyzing case and generating response..."):
            response = qa_chain.run({"query": full_context})
        st.markdown("### 📝 Clinical Decision Support")
        st.markdown(response)
    else:
        st.warning("Please fill in all the input fields.")
