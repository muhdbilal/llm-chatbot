import os
from dotenv import load_dotenv
import chainlit as cl
from langchain import HuggingFaceHub, PromptTemplate, LLMChain
from getpass import getpass
from huggingface_hub import login
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage

#loading the API key
load_dotenv(override=True)

model_id = "meta-llama/Meta-Llama-3-8B-Instruct" # "meta-llama/Meta-Llama-3-8B-Instruct" , "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "openai-community/gpt2-medium", "google/flan-t5-base" 
conv_model = HuggingFaceHub(huggingfacehub_api_token=
                            os.environ['HUGGINGFACEHUB_API_TOKEN'],
                            repo_id=model_id,
                            model_kwargs={"temperature":0.8, 
                            "max_new_tokens":250,
                            "max_length": 64})

# template = """

# Answer in a very concise manner

# {question}

# """

# prompt = PromptTemplate(template=template, input_variables=['question'])

prompt = ChatPromptTemplate.from_messages(
    [
    ("system", 
    """
    Answer very shortly and to the point.
    The answer should not exceed more than a few lines.
    Answer only 1 question.
    """
    ),
    ("user", "{question}\n"),
    ]
)

@cl.on_chat_start
def main():
    conv_chain = LLMChain(llm=conv_model,
                          prompt=prompt,
                          verbose=False)
    
    cl.user_session.set("llm_chain", conv_chain)
    
@cl.on_message
async def main(message:str):
    llm_chain = cl.user_session.get("llm_chain")
    res = await llm_chain.acall(message.content, callbacks=[cl.AsyncLangchainCallbackHandler()])
    
    #perform post processing on the received response here
    #res is a dict and the response text is stored under the key "text"
    await cl.Message(content=res["text"]).send()