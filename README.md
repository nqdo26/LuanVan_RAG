# RAG Server Python (Ongoing)

A Retrieval Augmented Generation (RAG) server implementation using Python for enhanced question-answering capabilities and document processing.

## Overview

This project implements a RAG system that combines document retrieval with language model generation to provide accurate, context-aware responses. It serves as the AI processing backend for the Study Assistant Platform.

## Features

-   🚀 Document processing and indexing
-   📚 Vector database connection (Pinecone)
-   🔍 Semantic search capabilities (Pinecone)
-   🤖 LLM integration for response generation (Groq)
-   📝 Document embedding and chunking
-   🔄 API integration with main backend
-   🎯 Question answering from documents
-   📊 Text similarity search

## Tech Stack

-   Python 3.8+
-   FastAPI
-   LangChain
-   Uvicorn

## Installation

1. Clone repository

```bash
git clone https://github.com/lethinhhung/thesis_project_rag_server.git
cd thesis_project_rag_server
```

2. Create and activate virtual environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment (Windows)
.\venv\Scripts\activate
```

3. Install dependencies

```bash
pip install -r requirements.txt
```

4. Create environment file
   Create a `.env` file in the root directory and configure:

```env
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=your_pinecone_index_name
GROQ_API_KEY=your_grok_api_key
```

5. Run the server

```bash
uvicorn server:app
```

## Project Structure

```
RAG/
├── server.py   # Main server
```

## API Documentation

### Base URL

```
http://localhost:8000
```

### Endpoints

#### 1. Health Check

```http
GET /
```

**Response:**

```json
{
    "message": "Hello World!"
}
```

#### 2. Keep Alive Health Check

```http
HEAD /v1/keep-alive
```

**Response:**

```json
{
    "status": "healthy"
}
```

#### 3. Document Ingestion

```http
POST /v1/ingest
```

**Description:** Processes and indexes documents into the vector database for retrieval.

**Request Body:**

```json
{
    "documentId": "string",
    "userId": "string",
    "document": "string",
    "title": "string",
    "courseId": "string (optional)",
    "courseTitle": "string (optional)"
}
```

**Response:**

```json
{
    "status": "done",
    "chunks_processed": 15
}
```

#### 4. Question Answering

```http
POST /v1/question
```

**Description:** Answers questions using RAG (Retrieval Augmented Generation) based on indexed documents.

**Request Body:**

```json
{
    "userId": "string",
    "query": "string"
}
```

**Response:**

```json
{
    "id": "string",
    "object": "chat.completion",
    "created": 1234567890,
    "model": "openai/gpt-oss-120b",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "string",
                "documents": [
                    {
                        "id": "string",
                        "text": "string",
                        "documentId": "string",
                        "score": 0.95
                    }
                ]
            },
            "finish_reason": "stop"
        }
    ]
}
```

#### 5. Delete Document

```http
POST /v1/delete-document
```

**Description:** Removes all vectors associated with a specific document from the vector database.

**Request Body:**

```json
{
    "documentId": "string",
    "userId": "string"
}
```

**Response:**

```json
{
    "deleted_ids": ["doc-1-0", "doc-1-1", "doc-1-2"]
}
```

#### 6. Chat Completions

```http
POST /v1/chat/completions
```

**Description:** Provides chat completions with optional knowledge base integration.

**Request Body:**

```json
{
    "messages": [
        {
            "role": "user|assistant|system",
            "content": "string"
        }
    ],
    "model": "string (optional, default: openai/gpt-oss-120b)",
    "userId": "string",
    "isUseKnowledge": "boolean (optional, default: false)",
    "courseId": "string (optional)",
    "courseTitle": "string (optional)"
}
```

**Response:**

-   When `isUseKnowledge: false`: Standard chat completion response
-   When `isUseKnowledge: true`: Chat completion with document references in the `documents` field

#### 7. Streaming Chat Completions (Under development)

```http
POST /v1/chat/streaming-completions
```

**Description:** Similar to chat completions but with streaming response support.

**Request/Response:** Same format as `/v1/chat/completions`

### Error Responses

**404 Not Found:**

```json
{
    "detail": "Không tìm thấy vectors nào với documentId này."
}
```

**500 Internal Server Error:**

```json
{
    "detail": "Error message details"
}
```

### Features

-   📝 **Document Processing**: Automatic text cleaning and chunking
-   🔍 **Semantic Search**: Vector-based similarity search using Pinecone
-   🤖 **AI Integration**: Powered by Groq's LLM for response generation
-   📚 **Knowledge Base**: RAG implementation for context-aware responses
-   🎯 **Course Filtering**: Ability to filter search results by course
-   📊 **Document References**: Responses include source document information
