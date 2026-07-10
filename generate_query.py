import argparse
import math
import multiprocessing
import os
import re
import time

from langchain.chat_models import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from tqdm import tqdm

from constants import api_key

template_gen_query = """
Generate 10 hypothetical questions that could be asked to a Sherlock Holmes Chatbot.

[example]
1. User: What is Mycroft's job?
2. User: Where do you live?
3. User: What is a mind palace?

[generated]
"""


def generate_query(num_generate):
    llm_temp = ChatOpenAI(temperature=0.8, model="gpt-3.5-turbo", api_key=api_key)
    prompt = ChatPromptTemplate.from_template(template_gen_query)
    gen_question_chain = prompt | llm_temp | StrOutputParser()

    generated_queries = []
    idx = 0
    pbar = tqdm(total=num_generate)
    while idx < num_generate:
        try:
            result = gen_question_chain.invoke({})
            generated_queries += re.findall("User: ([^\n]+)", result)
            idx += 1
            pbar.update(1)
        except Exception as exp:
            print(exp)
            time.sleep(60)

    return generated_queries


def main(num_process, num_generate):
    pool = multiprocessing.Pool(processes=num_process)
    input_list = [(num_generate,) for i in range(num_process)]
    result = pool.starmap(generate_query, input_list)
    result = sum(result, [])

    with open("/content/drive/MyDrive/dataset/holmes_query.txt", "w") as f:
        f.write("###\n".join(result))

    pool.close()
    pool.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract and process scripts from documents."
    )
    parser.add_argument(
        "--process", type=int, default=8, help="Number of processes to use"
    )
    parser.add_argument(
        "--num_generate", type=int, default=8, help="Number of data to generate"
    )

    args = parser.parse_args()
    main(args.process, args.num_generate)
