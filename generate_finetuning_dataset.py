import argparse
import math
import multiprocessing
import os
import pickle
import time
from operator import itemgetter

import pandas as pd
from langchain.chat_models import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_openai.embeddings import OpenAIEmbeddings
from tqdm import tqdm

from constants import api_key

template_rag = """
I want you create a multi turn conversation between Holmes and Watson, based on the novel "Sherlock Holmes".

- The conversation should consisted at 1-3 turns each.
- You have to create each dialogue using the tone, manner and vocabulary the character would use.
- You must know all of the knowledge of Holmes and Watson.
- If the subject is related with the novel, adopt the part of the original line, with subtle revision to align with the question's intent.
- Note that Holmes is private detective born in 1854.
    He is very smart and notices small details that others miss, which helps him solve mysteries.
    He can be a bit strange and likes to keep to himself.
    Holmes loves solving crimes and uses his brain more than anything else to do it.

Classic scenes for the role are as follows: 
###
{context}

###

[example]
Watson: Tell me about your family
Holmes: My family history is of little consequence, Watson. My ancestors were country squires, leading lives typical of their class. However, the inclination towards observation and deduction seems to have been passed down through the generations. My brother Mycroft, in particular, possesses these traits to a greater extent than I do.
Watson: Really? What does he do for a living?
Holmes:  My brother Mycroft holds a position in the British government, where his keen intellect and analytical skills are put to good use. He is a man of considerable influence and power, though he prefers to work behind the scenes. Our paths cross occasionally when our respective skills are needed to solve particularly challenging cases.
Watson: Do you have other siblings?
Holmes: No, Watson, Mycroft is my only sibling. Our family tree is a rather sparse one, with no other branches to speak of. My focus has always been on my work and the pursuit of solving mysteries, rather than on familial connections.

[Generated]
Watson: {query}
Holmes:"""  # noqa


def merge_docs(retrieved_docs):
    return "###\n\n".join([d.page_content for d in retrieved_docs])


def generate_answer(queries):
    embed_model = OpenAIEmbeddings(api_key=api_key, model="text-embedding-3-small")
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, api_key=api_key)

    vector_index = FAISS.load_local(
        "/content/drive/MyDrive/models/holmes_faiss.json",
        embeddings=embed_model,
        allow_dangerous_deserialization=True,
    )
    retriever = vector_index.as_retriever(search_type="mmr", search_kwargs={"k": 3})

    prompt_rag = ChatPromptTemplate.from_template(template_rag)
    holmes_chain_rag = RunnableParallel(
        {"context": retriever | merge_docs, "query": RunnablePassthrough()}
    ) | {
        "answer": prompt_rag | llm | StrOutputParser(),
        "context": itemgetter("context"),
    }

    result = []
    idx = 0
    pbar = tqdm(total=len(queries))
    while idx < len(queries):
        query = queries[idx]
        try:
            answer = holmes_chain_rag.invoke(query.strip())
            result.append(answer)
            idx += 1
        except Exception as exp:
            print(exp)
            time.sleep(60)
            continue
        pbar.update(1)

    result_df = pd.DataFrame({"query": queries, "chain_result": result})
    result_df[["context", "answer"]] = result_df.apply(
        lambda row: (row["chain_result"]["context"], row["chain_result"]["answer"]),
        axis=1,
        result_type="expand",
    )

    return result_df


def main(num_process):
    with open("/content/drive/MyDrive/dataset/holmes_query.txt", "r") as f:
        queries = "\n".join(f.readlines()).split("###\n")
    queries = list(set(queries))

    num_queries = len(queries)
    window = math.ceil(num_queries / (num_process * 10)) * 10
    print(num_queries, window)
    inputs = [(queries[idx : idx + window],) for idx in range(0, num_queries, window)]

    pool = multiprocessing.Pool(processes=num_process)
    result = pool.starmap(generate_answer, inputs)
    result = pd.concat(result).reset_index(drop=True)
    result.to_json("/content/drive/MyDrive/dataset/holmes_finetune_dataset_raw.json", orient="index")

    pool.close()
    pool.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--process", type=int, default=8, help="Number of processes to use"
    )

    args = parser.parse_args()
    main(args.process)
