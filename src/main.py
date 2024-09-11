import json
import os
import uuid
import chromadb
from chromadb.config import DEFAULT_TENANT, DEFAULT_DATABASE, Settings
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.documents import Document
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


def initialize_vector_store(client, embeddings, documents, uuids):
    vector_store = Chroma(
        client=client,
        collection_name="economic_updates_collection",
        embedding_function=embeddings,
    )
    vector_store.add_documents(documents=documents, ids=uuids)
    return vector_store


vector_store_from_client = initialize_vector_store(client, embeddings, documents, uuids)


def query_vector_store(query, vector_store, model):
    if isinstance(query, list):
        query = " ".join(query)  # Convert list to string if necessary
    query_embedding = embeddings.embed_query(query)
    results = vector_store.similarity_search(query_embedding, k=5)
    context = "\n\n".join([doc.page_content for doc in results])

    system_prompt = (
        "You are an assistant designed to provide insights based on Deloitte's weekly economic updates."
        "These updates offer a brief overview of the global political and economic situation, summarizing key impacts and trends."
        "\n\n"
        "{context}"
    ).format(context=context)

    response = model.invoke([HumanMessage(content=system_prompt)])
    return response.content


def generate_responses():
    with open('reasoning/questions.json', 'r') as questions_file:
        questions_data = json.load(questions_file)
        questions_list = questions_data['questions']
        print(questions_list)

        for question in questions_list:
            response = query_vector_store(question, vector_store_from_client, model)
            print(f"Question: {question}\nResponse: {response}\n")


generate_responses()
