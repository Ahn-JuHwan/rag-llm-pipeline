import re

import matplotlib.pyplot as plt
import seaborn as sns

from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain_core.tools import create_retriever_tool
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_openai.embeddings import OpenAIEmbeddings
from langgraph.graph import END, MessageGraph

#그래프를 그려주는 함
@tool
def graph_generator(text):
    """Use this function when you need to generate a graph.
                    input format should be label1|number1,label2|number2""" 
	# 여러번 그리지 않게 명세화
    plt.clf()
    label, data = list(zip(*re.findall("([^\|,\"]+) *\| *([\d]+)", text)))
    data = list(map(lambda x: int(x), data))
    print(label)
    print(data)
    sns.barplot(x=label, y=data)
	# 저
    plt.savefig("/content/drive/MyDrive/results/chain_out.png")
    
    return "I have generated graph for you!"

# 모델에 대해 대답을 받아옴
def invoke_model(state):
    print("===== ORACLE =====")
    print(state)
    
    result = model_with_tools.invoke(state)

    print("===== AI RESULT =====")
    print(result)
    return result

# openai의 결과에 tool call이 있으 tool call에 있는 
# function과 argument가지고 call 해
def use_tool(state):       
    print("====== TOOL CALL ======")
    print(state[-1].tool_calls)
    tool_calls = state[-1].tool_calls

    response = []
    for tool_call in tool_calls:                        
        function = tool_dict[tool_call["name"]]
    
        args = tool_call["args"]

        result = function.invoke(args)
        
        response.append(                                                                        ToolMessage(
                        tool_call_id=tool_call["id"],
                        content=str(result)
                  )
                )
             
    return response


# 마지막 state를 보고 어느 node로 향할지 정함
# 오픈 AI의 결과를 보고 tool call이 있으면use tool로  라우트 해줌
# 없으면 프로세종료 종료
def router(state):
        tool_calls = state[-1].additional_kwargs.get("tool_calls", [])
        if tool_calls:
            print("route to use_tool")
            return "use_tool"

        else:
            print("route to end")   
            return "end"

# ********** LangChain **********
embed_model = OpenAIEmbeddings(model='text-embedding-3-small')
vector_index = FAISS.load_local("/content/drive/MyDrive/models/faiss_chatgpt.json", embeddings=embed_model, allow_dangerous_deserialization=True)
retriever = vector_index.as_retriever(search_type="mmr")

llm = ChatOpenAI(model="gpt-3.5-turbo")

from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template("""
You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know.
Use three sentences maximum and keep the answer concise.

Question: {question}

Context: {context}

Answer:
""")

chain = RunnableParallel({"context": retriever, "question": RunnablePassthrough()})\
        | prompt \
        | llm \
        | StrOutputParser()




# ********** LangGraph **********

# 일반 랭체인에 로드했던 일반 리트리버에서 리트리벌 툴 만듬
retriever_tool = create_retriever_tool(
            retriever,
            "retriever-llm",
            description = "You should use this tool for queries related to machine learning. A function for searching documents related to LLM or ML. " 
            )

# 리트리벌 툴과 그래프 제너레이터 묶어서 openai 모델과 바인딩
tools = [retriever_tool, graph_generator]
openai_tools = [convert_to_openai_tool(tool) for tool in tools]
model_with_tools = llm.bind(tools = openai_tools)

# use tool 함수가 참고할 수 있는 tool dict 선언
tool_dict = {'retriever-llm': retriever_tool,
        'graph_generator': graph_generator}

# 그래프 정의
graph = MessageGraph()

# 어떤 액션 취할지
graph.add_node("oracle", invoke_model)
graph.add_node("use_tool", use_tool)

# oracle의 결과에 따라 라우터가  tool 사용 혹은 end로 엣지 연결
graph.add_conditional_edges("oracle", router, {
       "use_tool": "use_tool",
       "end": END,
       })
# tool 사용 후 항상 오라클로 돌아오기 때문에
# conditional endge가 아닌 그냥 endge
graph.add_edge("use_tool", "oracle")

# 진입점
graph.set_entry_point("oracle")
runnable = graph.compile()

from langchain_core.runnables import RunnableLambda
graph_chain = RunnableLambda(
            lambda x: runnable.invoke(
                        [HumanMessage(content=x)]
              )
            )


