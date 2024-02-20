# !pip install -q torch transformers accelerate bitsandbytes transformers sentence-transformers faiss-cpu pypdf streamlit langchain==0.1.6 langchain-community==0.0.19 langchain-core==0.1.23
# !pip install -q torch transformers 
# !pip install -q transformers sentence-transformers 
# !pip install -q faiss-cpu pypdf streamlit 
# !pip install -q langchain==0.1.6 langchain-community==0.0.19 langchain-core==0.1.23

import locale       # In Google Colab, use UTF-8 locale to install LangChain
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.schema.runnable import RunnablePassthrough
import streamlit as st

locale.getpreferredencoding = lambda: "UTF-8"

# Constants
PDF_FILE_PATH = 'test_pdf.pdf'  # Update with your file path
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
OLLAMA_MODEL_NAME = 'mistral'

loader = PyPDFLoader(PDF_FILE_PATH)
pages = loader.load_and_split()

print(f'Pages from the loader: {pages[0]} \n\n')

text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
doc_splits = text_splitter.split_documents(pages)

print(f'Pages from the loader: {doc_splits} \n\n')

db = FAISS.from_documents(doc_splits, HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5"))
retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 4})

## Summarization
bart_model_name = "facebook/bart-large-cnn"
summarizer = pipeline("summarization", model=bart_model_name)

llm = HuggingFacePipeline(pipeline=summarizer)

prompt = ChatPromptTemplate.from_messages(
    [("system", "Summarize the story about Daisy:\n\n{context}")]
)

llm_chain = LLMChain(llm=llm, prompt=prompt)

"""**Combine the LLM + Retriever to create the RAG**"""
# retriever = db.as_retriever()
rag_chain = {"context": retriever, "question": RunnablePassthrough()} | llm_chain

# question = "Tell a story about a girl in the country side"

# print(llm_chain.invoke({"context": "", "question": question}))
# print('\n\n\n')
# print(rag_chain.invoke(question))

if __name__ == "__main__":
    st.title("Summarization with RAG Development")
    question = st.text_input("Summarize tex:", value=" ")
    if question != " ":
        st.title('Summarization - LLM Chain with no context')
        st.write(llm_chain.invoke({"context": "", "question": question})['text'])

    if question != " ":
        st.title('Summarization with RAG')
        st.write(rag_chain.invoke(question)['text'])
