#api call method 추가
import json
import requests

# 어떤 문서들로 참고헀는지 보기
import pandas as pd

import streamlit as st

from langchain_core.messages import ToolMessage
from langchain_core.messages import HumanMessage
from langserve import RemoteRunnable
from streamlit_chat import message

class rag_web():
    def __init__(self):
        self.BASE_URL = "http://127.0.0.1:9000"
        self.URL_DICT = {
                "Chat-GPT": f"{self.BASE_URL}/chat_gpt",
                "Local": f"{self.BASE_URL}/local",
                "LangChain": f"{self.BASE_URL}/openai_chain/",
                "LangGraph": f"{self.BASE_URL}/openai_graph/",
                }
        self.gen_sidebar()        

        
    def gen_sidebar(self):
        with st.sidebar:
            st.radio(
            "Select the model to use!",
            ["LangChain", "LangGraph", "Chat-GPT", "Local"],
            key="model",
        
            )
    # llamaindex와 langserve 분리
    def get_answer_langserve(self, message):
        url = self.URL_DICT[st.session_state['model']]
        #RemoteRunnable을 통해 url만 있으면 결과 쉽게 가져옴
        lang = RemoteRunnable(url)
        result = lang.invoke(message)
        # 이미지를 디스플레이 해야  하는지와  소스 노드를
        # 같이 리턴하기 위해 초기화 

        return_image = False
        source_nodes = []
        
        # 랭그래프가 아닌 랭체인일 경우 결과는 단순 스트링
        if isinstance(result, str):
            # 결과와 필리스트 그리고 이미지 표시 x flag 리턴
            return result, source_nodes, return_image
        
        response = ""
        # 랭그래프 리스폰스는 마지막 AI의 컨텐츠를 가져오면 된다
        # 결과 리스펀스 중에 tool이 있는지 확인
        for msg in result:
            
            if hasattr(msg, "content") and msg.content:
                response = msg.content

            # 메세지에 tool name이 key로 있다면 tool 이 사용된 경우
            if isinstance(msg, ToolMessage):
                source_nodes.append({
                     "tool_name": msg.name,
                     "content": msg.content
                })
                    
                #tool name이 retrieve인 경우 소스 노드에 결과 추가
                if "generated graph" in str(msg.content).lower():
                                return_image = True
                

        return response, source_nodes, return_image

    # method
    def get_answer_llamaindex(self, message):
        # user message input으로 받음
      
        # host방식으로 넘기기 위해 dictionary 변환
        param = {'user_message': message}
        url =  self.URL_DICT[st.session_state['model']]
        # url로 post 형식으로 보내면 답옴
        resp = requests.post(url, json=param)

        #답변을 메세지 안에 넣어서 보내게 했으니 메세지 만 추출
        output = json.loads(resp.content)['message']

        return output["response"], output["source_nodes"], False

    # 모델의 종류에 따라서 어떤 함수로 답을 가져올지 
    def get_answer(self, message):
        # 모델명에 Lang이 포함되면 Langchain모델
        if st.session_state['model'].find("Lang") > -1:
            return self.get_answer_langserve(message)
        else:
            # 아닐 경우 llamaindex 호출
            return self.get_answer_llamaindex(message)

    
    # window 안쪽에서  get_answer call
    def window(self):
        st.title("My RAG Model")
        
        st.subheader("fastcampus")
    
        st.caption("created by Summber")
        
        #message = st.session_state.user_message
        # st.write(output)
        
        st.text_input("type a message..", key="user_message")
        if user_message := st.session_state.user_message:
            response, source_nodes, return_image = self.get_answer(user_message)
            st.success(response)

           # output  = self.get_answer(user_message)
            # message = st.session_state.user_message
            # st.write(output)
            #st.json(output)
            #st.success(output['response'])
            st.write(type(source_nodes))
            st.write(source_nodes)
            source_nodes = pd.DataFrame(source_nodes)
           
            if not source_nodes.empty:
                    st.subheader("Source Nodes")
                    st.dataframe(source_nodes)

            #st.write(source_nodes)
            # LamaIndex의 Retrieve에만 해당되는거라
            # node 컬럼이 있을 경우 실행
            if "node" in source_nodes.columns:
                source_nodes[["text", "metadata"]] = source_nodes["node"].apply(
                        lambda x: pd.Series((x["text"], x["metadata"]["file_name"]))
                        )
                # 원본 노드 제거
                source_nodes = source_nodes.drop('node', axis=1)
             
                # 결과 출력
                st.dataframe(source_nodes)

if __name__ == '__main__':
    st.set_page_config(
        page_title="FastCampus RAG-Demo",
        page_icon=":pizza:"
    )
    # call
    web = rag_web()
    web.window()

