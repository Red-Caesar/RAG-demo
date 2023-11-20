import os
import uuid
import pickle
from tqdm import tqdm
from typing import List, Dict

from langchain.llms import OpenAI
from langchain.llms.octoai_endpoint import OctoAIEndpoint
from langchain.embeddings.octoai_embeddings import OctoAIEmbeddings
from llama_index.schema import Document
from llama_index.embeddings import LangchainEmbedding
from llama_index import (
    LLMPredictor,
    ServiceContext,
    download_loader,
    GPTVectorStoreIndex,
    VectorStoreIndex,
    PromptHelper,
)

from langchain.chat_models import ChatOllama


from parse_data import get_contexts, get_qa

# print("TOKEN", os.environ["OCTOAI_API_KEY"])
octoai_api_token = os.environ.get("OCTOAI_API_KEY"),
endpoint_url = "https://ollm-deploy-for-downstream-actions-ucxl30slhvyv.octoai.run/v1/chat/completions"
# endpoint_url =  https://llama-2-7b-chat-demo-kk0powt97tmb.octoai.run/v1/chat/completions
model_name = "llama-2-7b-chat"
qa = get_qa("datasets/triviaqa_train.json")[:10]

def get_language_model(endpoint_url: str, model_name: str):
    llm = OctoAIEndpoint(
        endpoint_url=endpoint_url,
        model_kwargs={
            "model": model_name,
            "messages": [
                {
                    "role": "system",
                    "content": "Below is an instruction that describes a task. Write an answer in one word.",
                }
            ],
            "stream": True,
            "max_tokens": 256,
        },
    )
    return llm

def evaluate():
    pass

def get_RAG_acc(octoai_api_token: str, endpoint_url: str, qa: List[Dict[str, str]]) -> float:
    
    # llm_llama2_7b = get_language_model(endpoint_url, "llama-2-7b-chat")
    # llm_llama2_7b = ChatOllama(model="llama2:7b-chat")

    # llm_predictor = LLMPredictor(llm=llm_llama2_7b)

    embeddings = LangchainEmbedding(
        OctoAIEmbeddings(
            endpoint_url="https://instructor-large-f1kzsig6xes9.octoai.run/predict"
        )
    )

    service_context = ServiceContext.from_defaults(
        llm_predictor=llm_predictor, chunk_size_limit=512, embed_model=embeddings,
    )

    texts = get_contexts("datasets/triviaqa_train.json")[:10]
    doc_ids = [str(uuid.uuid4()) for _ in texts]
    documents = [ Document(text=s, metadata={"doc_id": doc_ids[i]}) for i, s in enumerate(texts)]
    print("Start create index store")
    index = GPTVectorStoreIndex.from_documents(
        documents, service_context=service_context
    )
    print("Finish create index store")
    with open('GPTVectorStoreIndex.pkl', 'wb') as file: 
        pickle.dump(index, file) 

    query_engine = index.as_query_engine(llm_predictor=llm_predictor)
    
    acc = 0
    for element in tqdm(qa, position=0, leave=True):
        prompt, answer = element["question"], element["answer"]

        response = query_engine.query(prompt)
        acc += (response == answer)
    
    return acc / len(qa)

llm_llama2_7b_acc = get_RAG_acc(octoai_api_token, endpoint_url, qa)
print(llm_llama2_7b_acc)