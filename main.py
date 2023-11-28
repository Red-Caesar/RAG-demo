import os
import uuid
from tqdm import tqdm

from langchain.llms.octoai_endpoint import OctoAIEndpoint
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from llama_index.schema import Document
from llama_index import (
    LLMPredictor,
    ServiceContext,
    GPTVectorStoreIndex,
)

from parse_data import get_contexts, get_qa
from dotenv import load_dotenv

dotenv_path = os.path.join(os.getcwd(), '.env')
load_dotenv(dotenv_path)

token = os.environ["OCTOAI_TOKEN"]
endpoint = os.environ["ENDPOINT"]
model_name = "llama-2-13b-chat-fp16"

qa = get_qa("datasets/triviaqa_train.json")[:10]
contexts = get_contexts("datasets/triviaqa_train.json")[:10]

def get_language_model(model_name: str):
    llm = OctoAIEndpoint(
        endpoint_url=endpoint,
        octoai_api_token=token,
        model_kwargs={
            "model": model_name,
            "messages": [
                {
                    "role": "system",
                    "content": "Below is an instruction that describes a task. Write an answer in one word.",
                }
            ],
            "stream": False,
            "max_tokens": 256,
        },
    )
    return llm

def get_RAG_acc() -> float:
    
    llm_llama2_13b = get_language_model(model_name)
    llm_predictor = LLMPredictor(llm=llm_llama2_13b)

    embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    service_context = ServiceContext.from_defaults(
        llm_predictor=llm_predictor, chunk_size_limit=512, embed_model=embedding_function,
    )

    doc_ids = [str(uuid.uuid4()) for _ in contexts]
    documents = [ Document(text=s, metadata={"doc_id": doc_ids[i]}) for i, s in enumerate(contexts)]

    print("Start create index store")
    index = GPTVectorStoreIndex.from_documents(
        documents, service_context=service_context
    )
    print("Finish create index store")

    query_engine = index.as_query_engine(llm_predictor=llm_predictor)
    
    acc = 0
    for element in tqdm(qa, position=0, leave=True):
        prompt, answer = element["question"], element["answer"]

        response = query_engine.query(prompt)
        acc += (answer.lower() in str(response).lower())
    
    return acc / len(qa)

llm_llama2_13b_acc = get_RAG_acc()
print(llm_llama2_13b_acc)