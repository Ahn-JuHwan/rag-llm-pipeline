import pandas as pd
from llama_index.llms.openai import OpenAI
from llama_index.core.evaluation import CorrectnessEvaluator

from google.colab import userdata

import os

api_key = os.environ["OPENAI_API_KEY"]


llm = OpenAI("gpt-4")
evaluator = CorrectnessEvaluator(llm=llm)

data_path = "/content/drive/MyDrive/results/inference_result"
eval_data_path = "/content/drive/MyDrive/results/eval_result"

for fname in["ml_qa_ft", "ml_qa_ft_new_models"]:
    eval_df = pd.read_csv(f"{data_path}/{fname}.csv")

    eval_df['result'] = eval_df.apply(lambda row: evaluator.evaluate(query=row.question, response=row.answer_llm, reference=row.answer), axis=1)

    eval_df['score'] = eval_df['result'].apply(lambda x: x.score)

    eval_df['feedback'] = eval_df['result'].apply(lambda x: x.feedback)

    eval_df.to_csv(f"{eval_data_path}/{fname}_eval.csv")
