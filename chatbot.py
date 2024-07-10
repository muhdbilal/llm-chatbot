import os
import chainlit as cl
from langchain import HuggingFaceHub, PromptTemplate, LLMChain
from getpass import getpass
from huggingface_hub import login
from langchain.prompts import ChatPromptTemplate

HUGGINGFACEHUB_API_TOKEN = getpass(prompt = "Token: ")
os.environ['HUGGINGFACEHUB_API_TOKEN'] = HUGGINGFACEHUB_API_TOKEN

model_id = "meta-llama/Meta-Llama-3-8B-Instruct" # "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "openai-community/gpt2-medium", "google/flan-t5-base" 
conv_model = HuggingFaceHub(huggingfacehub_api_token=
                            os.environ['HUGGINGFACEHUB_API_TOKEN'],
                            repo_id=model_id,
                            model_kwargs={"temperature":0.8, 
                            "max_new_tokens":500,
                            "max_length": 64})


# template = """Question: {question}

# Answer: Let's take this step by step
# """
template = """Let's go this step by step

{question}

"""


@cl.on_chat_start
def main():
    prompt = PromptTemplate(template=template, input_variables=['question'])
    conv_chain = LLMChain(llm=conv_model,
                          prompt=prompt,
                          verbose=True)
    
    cl.user_session.set("llm_chain", conv_chain)
    
@cl.on_message
async def main(message:str):
    llm_chain = cl.user_session.get("llm_chain")
    res = await llm_chain.acall(message.content, callbacks=[cl.AsyncLangchainCallbackHandler()])
    
    #perform post processing on the received response here
    #res is a dict and the response text is stored under the key "text"
    await cl.Message(content=res["text"]).send()