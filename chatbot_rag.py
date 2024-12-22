import os
import itertools
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader  #document loader: https://python.langchain.com/docs/modules/data_connection/document_loaders
from langchain.text_splitter import RecursiveCharacterTextSplitter  #document transformer: text splitter for chunking
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma #vector store
from langchain import HuggingFaceHub  #model hub
from langchain.chains import RetrievalQA
import chainlit as cl

#loading the API key
load_dotenv(override=True)

# Specify the folder path
folder_path = "./database/"

# Initializing the text splitter
splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=20)

# Initializing the list to populate splitted text with
docs = []

# List all files in the folder and filter for PDF files
pdf_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.pdf')]
print(pdf_files)

# Process each PDF file
for pdf_file in pdf_files:
    print(f"Loading: {pdf_file}")
    try:
        # Open and read the PDF file
        loader = PyPDFLoader(pdf_file)
        pages = loader.load()
        # splitting and adding the pdf to a list
        docs_temp = splitter.split_documents(pages)
        docs.append(docs_temp)
        
    except Exception as e:
        print(f"Error loading {pdf_file}: {e}")

# Unlisting nested lists
docs = list(itertools.chain(*docs))

embeddings = HuggingFaceEmbeddings()
doc_search = Chroma.from_documents(docs, embeddings)

repo_id = "meta-llama/Meta-Llama-3-8B-Instruct"
llm = HuggingFaceHub(repo_id=repo_id, 
                     model_kwargs={"temperature":0.8, 
                            "max_new_tokens":250,
                            "max_length": 64}) 

@cl.on_chat_start
def main():
    retrieval_chain = RetrievalQA.from_chain_type(llm, chain_type='stuff', retriever=doc_search.as_retriever())
    cl.user_session.set("retrieval_chain", retrieval_chain)
    
@cl.on_message
# async def main(message:str):
async def main(message: cl.Message):
    retrieval_chain = cl.user_session.get("retrieval_chain")
    res = await retrieval_chain.acall(message.content, callbacks=
                                      [cl.AsyncLangchainCallbackHandler()])
    
    #print(res)
    await cl.Message(content=res["result"]).send()