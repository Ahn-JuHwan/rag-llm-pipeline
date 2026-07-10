import json

import pandas as pd
import requests
import streamlit as st
from langchain_core.messages import HumanMessage
from langserve import RemoteRunnable
from streamlit_chat import message

from langchain_core.messages import ToolMessage


class rag_web:
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
                # 라디오 버튼 바뀌면 로그 리셋
                on_change=self.reset_log,
            )

    def get_answer_langserve(self, message):
        url = self.URL_DICT[st.session_state["model"]]
        lang = RemoteRunnable(url)
        result = lang.invoke(message)
        return_image = False
        source_nodes = []

        if isinstance(result, str):
            return result, source_nodes, return_image

        response = result[-1].content

        for msg in result:
            if isinstance(msg, ToolMessage):
                if msg.name == "retriever-llm":
                    source_nodes.append({
                        "tool_name": msg.name,
                        "content": msg.content
                    })

                elif msg.name == "graph_generator":
                    return_image = True

        return response, source_nodes, return_image

    def get_answer_llamaindex(self, message):
        param = {"user_message": message}
        # resp = requests.post(self.URL, json=param)
        # url = self.URL_LOCAL if st.session_state['model'] == 'Local' else self.URL_CHATGPT
        url = self.URL_DICT[st.session_state["model"]]
        resp = requests.post(url, json=param)
        output = json.loads(resp.content)["message"]
        return output["response"], output["source_nodes"], False

    def get_answer(self, message):
        if st.session_state["model"].find("Lang") > -1:
            return self.get_answer_langserve(message)
        else:
            return self.get_answer_llamaindex(message)

    def update_log(self, user_message, bot_message):
        if "chat_log" not in st.session_state:
            st.session_state.chat_log = {"user_message": [], "bot_message": []}
        
        # 이번턴에 유저 메시지와 봇 메시지를 받아 세션 state에 추가
        st.session_state.chat_log["user_message"].append(user_message)
        st.session_state.chat_log["bot_message"].append(bot_message)
    # 라디오 버튼으로 오갈때 대화 초기화 함수
    def reset_log(self):
        st.session_state.chat_log = {"user_message": [], "bot_message": []}
        st.session_state.bot_output = None, None, None

    # 화면 디스플레이에 추가
    def display_chat_log(self):
        bot_messages = st.session_state.chat_log["bot_message"][::-1]
        user_messages = st.session_state.chat_log["user_message"][::-1]

        # message history 가져오고 유저 메시지, 봇 메시지를 디스플레이
        for idx, (bot, user) in enumerate(zip(bot_messages, user_messages)):
            # 디스플레이는 streamlit_chat 사용
            message(bot, key=f"{idx}_bot")
            message(user, key=str(idx), is_user=True)
    
    # 이전턴의 유저 인풋이 차 있어서 그 인풋에 대한 답변이 또 자동으로 생성 됨
    def user_input(self):
        user_message = st.session_state.user_message
        st.session_state['user_message'] = ""
        st.session_state['bot_output'] = self.get_answer(user_message)
        self.update_log(user_message, st.session_state['bot_output'][0])
    
    def window_chat(self):
        st.title("My RAG Model")
        st.subheader("fastcampus")
        st.caption("created by summer")

        st.text_input("type a message..", key="user_message", on_change=self.user_input)

        if st.button("reset"):
            self.reset_log()
            # 초록창 대신 채팅 디스플레이
            self.display_chat_log()

        if 'bot_output' in st.session_state:
            response, source_nodes, return_image = st.session_state['bot_output']
            self.display_chat_log()

            if return_image:
                st.image("/content/drive/MyDrive/results/chain_out.png")

            source_nodes = pd.DataFrame(source_nodes)
            if "node" in source_nodes.columns:
                source_nodes[["text", "metadata"]] = source_nodes["node"].apply(
                    lambda x: pd.Series((x["text"], x["metadata"]["file_name"]))
                )
                source_nodes = source_nodes.drop("node", axis=1)
            st.dataframe(source_nodes)

    def window(self):
        st.title("My RAG Model")
        st.subheader("fastcampus")
        st.caption("created by summer")

        st.text_input("type a message..", key="user_message")

        # message = st.session_state.user_message
        # st.write(f"user message: {message}")

        if user_message := st.session_state["user_message"]:
            response, source_nodes, return_image = self.get_answer(user_message)
            st.success(response)

            if return_image:
                st.image("/content/drive/MyDrive/results/chain_out.png")

            source_nodes = pd.DataFrame(source_nodes)
            if "node" in source_nodes.columns:
                source_nodes[["text", "metadata"]] = source_nodes["node"].apply(
                    lambda x: pd.Series((x["text"], x["metadata"]["file_name"]))
                )
                source_nodes = source_nodes.drop("node", axis=1)
            
            if not source_nodes.empty:
                st.dataframe(source_nodes)


if __name__ == "__main__":
    st.set_page_config(page_title="FastCampus RAG-Demo", page_icon=":pizza:")
    web = rag_web()
    web.window_chat()
    #web.window()
