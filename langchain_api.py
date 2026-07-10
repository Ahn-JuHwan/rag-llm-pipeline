import re

import matplotlib.pyplot as plt
import seaborn as sns

from langchain_openai  import ChatOpenAI
from langchain.tools import tool
from langchain_core.tools import create_retriever_tool
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_openai.embeddings import OpenAIEmbeddings
from langgraph.graph import END, MessageGraph
from langchain_core.prompts import ChatPromptTemplate


@tool
def graph_generator(text):
    """Use this function when you need to generate a graph.
                    input format should be label1|number1,label2|number2"""
    plt.clf()
    label, data = list(zip(*re.findall(r"([^\|,\"]+) *\| *([\d]+)", text)))
    data = list(map(lambda x: int(x), data))
    print(label)
    print(data)
    sns.barplot(x=label, y=data)
    plt.savefig("/content/drive/MyDrive/results/chain_out.png")
    
    return "I have generated graph for you!"

def invoke_model(state):
    print(state)
    return model_with_tools.invoke(state)


def use_tool(state):
    print("use_tool")
    tool_calls = state[-1].tool_calls

    response = []

    for tool_call in tool_calls:
        function_name = tool_call["name"]
        args = tool_call["args"]

        function = tool_dict[function_name]
        result = function.invoke(args)

        response.append(
            ToolMessage(
                tool_call_id=tool_call["id"],
                content=str(result),
                name=function_name
            )
        )

    return response


def router(state):
    tool_calls = state[-1].tool_calls
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
retriever_tool = create_retriever_tool(
    retriever,
    "retriever-llm",
    description = "You should use this tool for queries related to machine learning. A function for searching documents related to LLM or ML. "
)

tools = [retriever_tool, graph_generator]
openai_tools = [convert_to_openai_tool(tool) for tool in tools]
model_with_tools = llm.bind(tools = openai_tools)


tool_dict = {'retriever-llm': retriever_tool,
          'graph_generator': graph_generator}

graph = MessageGraph()
graph.add_node("oracle", invoke_model)
graph.add_node("use_tool", use_tool)

graph.add_conditional_edges("oracle", router, {
   "use_tool": "use_tool",
   "end": END,
})
graph.add_edge("use_tool", "oracle")
graph.set_entry_point("oracle")
runnable = graph.compile()

graph_chain = HumanMessage|runnable
