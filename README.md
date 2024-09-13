# Deloitte-Insightbot

## Overview

The `deloitte-insightbot` is a question and answer system designed to provide insights based on Deloitte's weekly
economic updates. These updates offer a brief overview of the global political and economic situation, summarizing key
impacts and trends.

## Features

- **Data Ingestion**: Fetches content from Deloitte's weekly economic
  update [URL](https://www2.deloitte.com/us/en/insights/economy/global-economic-outlook/weekly-update/weekly-update-2023-10.html?icid=archive_click).
- **Embeddings Storage**: Stores embeddings of the content in a VectorDB.
- **Retrieval-Augmented Generation**: Retrieves relevant passages to generate answers for user queries using an LLM.

## Components

- **Data Ingestion**: A module to scrape and parse content from the specified URL.
    - `UnstructuredURLLoader` class to fetch and parse the content from the URL.
- **Embeddings Model**: Utilizes an embedding model to convert content into vector representations.
    - `OpenAIEmbeddings` model with the model name `text-embedding-3-large`.
- **VectorDB**: Stores the embeddings for efficient retrieval.
    - `Chroma` class from langchain_chroma is used to interact with ChromaDB.
- **LLM**: Generates answers based on the retrieved passages.
    - ChatOpenAI class with the model name `gpt-3.5-turbo`.

## Usage

1. **Ingest Data**: Run the data ingestion script to fetch and parse the content.
2. **Store Embeddings**: Use the embeddings model to convert the content into vectors and store them in the VectorDB.
3. **Query System**: Input a user query to retrieve relevant passages and generate an answer using the LLM.

## Commands

Install the required packages

```bash
pip install -r requirements.txt
```

Start the ChromaDB container

```bash
docker compose up -d
```

Ping the ChromaDB container to check if it is running

```bash
curl localhost:8000/api/v1/heartbeat
```

Run the application

```bash
python src/main.py
```