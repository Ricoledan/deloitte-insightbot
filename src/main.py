import json
import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_community.document_loaders import UnstructuredURLLoader

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

urls = [
    "https://www2.deloitte.com/us/en/insights/economy/global-economic-outlook/weekly-update/weekly-update-2023-10.html?icid=archive_click"
]

model = ChatOpenAI(model="gpt-3.5-turbo")

loader = UnstructuredURLLoader(urls=urls)

data = loader.load()

vectorstore = Chroma.from_documents(
    data,
    embedding=OpenAIEmbeddings(),
)

retriever = vectorstore.as_retriever()

system_prompt = (
    "You are an assistant designed to provide insights based on Deloitte's weekly"
    "economic updates.These updates offer a brief overview of the global political and economic situation, summarizing key"
    "impacts and trends."
    "\n\n"
    "{context}"
)


def generate_responses():
    with open('query/questions.json', 'r') as file:
        q = json.load(file)
        questions = q['questions']

    with open('query/responses.txt', 'w') as file:
        for question in questions:
            response = model.invoke([HumanMessage(content=question)])
            response_text = response.content
            file.write(f"Question: {question}\nResponse: {response_text}\n\n")


generate_responses()
