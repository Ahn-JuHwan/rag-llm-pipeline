import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          DataCollatorForLanguageModeling, PreTrainedTokenizer,
                          Trainer, TrainingArguments, pipeline)

from peft import LoraConfig, get_peft_model
import torch
# 각 데이터 셋에 element에 대해서 트레인 인풋을 토큰화
def tokenize(element, tokenizer=None):
    outputs = tokenizer(
        element["train_input"],
        truncation=True,
        max_length=256,
        padding="max_length",
    )
    return {
            "input_ids": outputs["input_ids"],
            "attention_mask": outputs["attention_mask"],
            "labels": outputs["input_ids"].copy(),
     }

# 저장했던 데이터 셋 불러옴
def load_data():
    dataset_train = Dataset.from_json("/content/drive/MyDrive/dataset/holmes_finetune_dataset_train.json")
    dataset_test = Dataset.from_json("/content/drive/MyDrive/dataset/holmes_finetune_dataset_test.json")
    dataset = DatasetDict({"train": dataset_train, "test": dataset_test})
    return dataset

# 모델 로드
def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    )
    lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["c_attn", "c_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            )
    
    model = get_peft_model(model, lora_config)
    model.enable_input_require_grads()

    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    
    model.print_trainable_parameters()
    tokenizer.pad_token = tokenizer.eos_token

    return tokenizer, model


# 토큰 나이저 함수를 데이터 셋에 맵핑
# 배치 프로세스 적용
# 기존에 있던 컬럼들을 모두 삭제
# 토큰나이저는 키워드 알규먼트로 넘김

def preprocess(dataset, tokenizer):
    tokenized_datasets = dataset.map(
        tokenize,
        batched=True,
        remove_columns=dataset["train"].column_names,
        fn_kwargs={"tokenizer": tokenizer},
    )

    # 학습을 사용할 데이터 콜레이터 만듬
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    return tokenized_datasets, data_collator


if __name__ == "__main__":
    dataset = load_data()
    tokenizer, model = load_model("/content/drive/MyDrive/models/camel-5b-hf")
    
    # 데이터 전처리
    tokenized_datasets, data_collator = preprocess(dataset, tokenizer)

    # 저장 경로 지정
    save_dir = "/content/drive/MyDrive/models/camel-5b-finetuned_0331"

    # 트레이닝 변수 제공
    training_params = TrainingArguments(
        output_dir=save_dir,
        num_train_epochs=3,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=32,
        fp16=False,
        report_to="none",
        # 딥스피드를 통한 분산 학습이 가능
        # deepspeed="/content/drive/MyDrive/models/ds_config.json",
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_params,
        data_collator=data_collator,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
    )

    trainer.train()

    trainer.save_model(output_dir=save_dir)
