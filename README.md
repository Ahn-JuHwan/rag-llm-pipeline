# rag-llm-pipeline

패스트캠퍼스 "RAG를 활용한 LLM 서비스 구축" 과정을 들으면서 작성한 코드입니다.
셜록 홈즈 챗봇을 목표로 데이터셋 생성부터 LoRA 파인튜닝, 평가, 서빙까지 직접 돌려봤습니다.

## 파이프라인

1. **데이터셋 생성**
   - `extract_script.py` : 셜록 홈즈 원문에서 GPT-3.5와 Kor로 대화 스크립트 추출
   - `generate_query.py` : 챗봇에 들어올 만한 가상 질문 생성
   - `generate_finetuning_dataset.py` : FAISS 리트리버(MMR)와 GPT-3.5를 조합한 RAG 체인으로 홈즈 말투의 답변을 생성해 파인튜닝용 데이터셋 구성
2. **파인튜닝**
   - `finetune_ds.py` : camel-5b-hf 모델을 LoRA(r=8, alpha=16)로 파인튜닝.
     Colab GPU 메모리에 맞추기 위해 배치 크기 1, gradient accumulation 32, gradient checkpointing 사용
3. **평가**
   - `evaluate.py` : GPT-4를 평가자로 두고(LlamaIndex CorrectnessEvaluator) 파인튜닝 전후 답변을 채점해 비교
4. **서빙**
   - `rag_api.py` : FastAPI + LangServe로 RAG API 서빙. LlamaIndex 벡터 인덱스 사용(OpenAI 임베딩, bge-small-en-v1.5 로컬 임베딩)
   - `rag_st.py`, `my_rag_st.py` : Streamlit 챗 UI. 사이드바에서 LangChain / LangGraph / Chat-GPT / Local 모델 선택
5. **에이전트**
   - `langchain_api.py`, `my_langchain_api.py` : LangGraph로 리트리버 툴과 그래프 생성 툴을 호출하는 에이전트 구성

## 사용 모델

- camel-5b-hf — 파인튜닝 대상 (Hugging Face)
- bge-small-en-v1.5 — 로컬 임베딩
- gpt-3.5-turbo, gpt-4 — 데이터 생성과 평가

## 실행 환경

- Google Colab(GPU) + Google Drive에서 작업했습니다. 코드의 `/content/drive/...` 경로는 그 흔적입니다.
- `OPENAI_API_KEY`는 환경변수로 설정해야 합니다.
- Colab에서 FastAPI 서버를 외부로 열 때는 localtunnel을 사용했습니다(`package.json`).
