import json
import os
import uuid

import chromadb
from chromadb.config import DEFAULT_DATABASE, DEFAULT_TENANT, Settings
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

model = ChatOpenAI(model="gpt-3.5-turbo")
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

urls = [
    "https://www2.deloitte.com/us/en/insights/economy/global-economic-outlook/weekly-update/weekly-update-2023-10.html?icid=archive_click"
]

loader = UnstructuredURLLoader(urls=urls)
data = loader.load()

documents = [
    Document(page_content=content.page_content, metadata={"source": "url"}, id=i)
    for i, content in enumerate(data, start=1)
]

uuids = [str(uuid.uuid4()) for _ in documents]

client = chromadb.HttpClient(
    host="localhost",
    port=8000,
    ssl=False,
    headers=None,
    settings=Settings(),
    tenant=DEFAULT_TENANT,
    database=DEFAULT_DATABASE,
)


def populate_vector_store(client, embeddings, documents, uuids):
    vector_store = Chroma(
        client=client,
        collection_name="economic_updates_collection",
        embedding_function=embeddings,
    )
    vector_store.add_documents(documents=documents, ids=uuids)
    return vector_store


vector_store_from_client = populate_vector_store(client, embeddings, documents, uuids)


def query_vector_store(question, vector_store, model):
    results = vector_store.similarity_search(question, k=5)
    context = "\n\n".join([doc.page_content for doc in results])

    max_tokens = 16000
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=max_tokens, chunk_overlap=0)
    context_chunks = text_splitter.split_text(context)

    truncated_context = context_chunks[0]

    system_prompt = (
        "You are an assistant designed to provide insights based on Deloitte's weekly economic updates."
        "These updates offer a brief overview of the global political and economic situation, summarizing key impacts and trends."
        "\n\n"
        "{context}"
    ).format(context=truncated_context)

    response = model.invoke([HumanMessage(content=system_prompt)])
    return response.content


def generate_responses():
    with open('reasoning/questions.json', 'r') as questions_file:
        questions_data = json.load(questions_file)
        questions_list = questions_data['questions']

    with open('reasoning/response.txt', 'w') as response_file:
        for question in questions_list:
            response = query_vector_store(question, vector_store_from_client, model)
            response_file.write(f"Question: {question}\n")
            response_file.write(f"Answer: {response}\n\n")
            print(f"Question: {question}")
            print(f"Answer: {response}\n")


generate_responses()