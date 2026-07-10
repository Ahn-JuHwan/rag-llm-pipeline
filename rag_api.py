from fastapi import FastAPI

from llama_index.core import StorageContext, load_index_from_storage

import os
# OPENAI_API_KEY는 환경변수로 설정해야 합니다

from llama_index.llms.huggingface import HuggingFaceLLM

from langchain_api import chain, graph_chain
from langserve import add_routes

#gpt 불러오기
storage_context = StorageContext.from_defaults(persist_dir="/content/drive/MyDrive/models/vector_index_lm_text_chatGPT")
vector_index = load_index_from_storage(storage_context)
query_engine = vector_index.as_query_engine()

# 로컬 모델
#storage_context_bge = StorageContext.from_defaults(persist_dir="/content/drive/MyDrive/models/vector_index_lm_text_bge")
# embed 모델 선언도 필요
#vector_index_bge = load_index_from_storage(storage_context_bge, embed_model="local:/content/drive/MyDrive/models/bge-small-en-v1.5")
#query_engine_local = vector_index.as_query_engine()


#llm = HuggingFaceLLM(
 #           tokenizer_name="/content/drive/MyDrive/models/camel-5b-hf",
  #          model_name="/content/drive/MyDrive/models/camel-5b-hf",
   #         context_window=2048,
    #        max_new_tokens=512,
     #       generate_kwargs={"temperature":0.25, "do_sample":True},
      #      device_map="auto",
       #     tokenizer_kwargs={"max_length":2048},
#)

app = FastAPI()



@app.get("/")
async def root():
    return{"message": "Hello World!"}

@app.get("/chat_gpt")
async def rag_chatgpt_get(user_message):
    return{"message": query_engine.query(user_message).response}

#API로 호출
@app.post("/chat_gpt")
async def rag_chatgpt(param: dict={}):
    user_message = param.get('user_message', ' ')
    return{"message": query_engine.query(user_message)}

   
@app.post("/local")
async def rag_local(param: dict={}):
        user_message = param.get('user_message', ' ')
        return{"message": query_engine_local.query(user_message)}

#langserve를이용해 serving 하는 코드
add_routes(app, chain, path="/openai_chain")
add_routes(app, graph_chain, path="/openai_graph")
