import argparse
import math
import multiprocessing
import os
import pickle
import time

import openai
from kor.extraction import create_extraction_chain
from langchain_openai.chat_models import ChatOpenAI
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from tqdm import tqdm

from constants import api_key

def parse_kor_result(data):
    try:
        script = data["data"]["script"]
        results = [
            f"{scr['role']}: {scr['dialogue']}\n" for scr in script if "role" in scr
        ]
        holmes_inc = any(scr["role"] == "Holmes" for scr in script if "role" in scr)
    except:
        return "", False
    return "".join(results), holmes_inc


def extract_scripts(documents, schema):
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, api_key=api_key)

    kor_chain = create_extraction_chain(llm, schema)

    doc_script = []
    pbar = tqdm(total=len(documents))

    idx = 0
    while idx < len(documents):
        # try except 사용 이유 -> 
        # OpenAi 콜 많이 쓰면 리미트 초과해서 에러
        # 발생할 수 있어서서
        try:
            doc = documents[idx]
            script = kor_chain.invoke(doc.page_content)
            script_parsed, holmes_inc = parse_kor_result(script)
            if holmes_inc:
                doc_script.append(script_parsed)
            idx += 1
            pbar.update(1)
        except Exception as e:
            print(e)
            time.sleep(60)

    return doc_script


def main(num_process):
    loader = DirectoryLoader('/content/drive/MyDrive/dataset/holmes', glob="*", show_progress=True)
    docs = loader.load()

    text_splitter = CharacterTextSplitter(
        separator="\n\n",
        chunk_size=2048,
        chunk_overlap=256,
        length_function=len,
        is_separator_regex=False,
    )
    documents = text_splitter.split_documents(docs)[:10]
    num_docs = len(documents)

    with open("/content/drive/MyDrive/dataset/kor_schema_holmes.json", "rb") as file:
        schema = pickle.load(file)
    #process 갯수는 커멘드라인에서 인자로 받을거
    #->각 프로세스마다 처리해야할 다큐먼트 갯수 나눔
    window = math.ceil(num_docs / (num_process * 10)) * 10
    # 각 프로세스에 들어갈 인풋 정의
    inputs = [
        (documents[idx : idx + window], schema) for idx in range(0, num_docs, window)
    ]

    # 워크 프로세스들 생성
    pool = multiprocessing.Pool(processes=num_process)
    
    #각각의 프로세스에 함수와 결과를 넣어줘 한번에 받아옴
    result = pool.starmap(extract_scripts, inputs)
    result = sum(result, [])
    
    #결과 저장
    with open("/content/drive/MyDrive/dataset/test.txt", "w") as f:
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

    args = parser.parse_args()
    main(args.process)
