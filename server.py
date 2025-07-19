from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
import requests
from pinecone import Pinecone, ServerlessSpec
from groq import Groq
from langchain.text_splitter import RecursiveCharacterTextSplitter
import re
import unicodedata
import os
from dotenv import load_dotenv

from models import IngestPayload, QuestionPayload, DeletePayload, ChatCompletionPayload

# Load environment variables
load_dotenv()
app = FastAPI()

# Pinecone init
pc = Pinecone(
    api_key=os.getenv("PINECONE_API_KEY")
)

index_name = os.getenv("PINECONE_INDEX_NAME")

# Groq init
client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)

# Define the text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", ".", " ", ""]
)

# Check if the index exists, if not create it
if not pc.has_index(index_name):
    pc.create_index_for_model(
        name=index_name,
        cloud="aws",
        region="us-east-1",
        embed={
            "model":"llama-text-embed-v2",
            "field_map":{"text": "text"}
        }
    )

# Connect to the index
index = pc.Index(index_name)


# Define endpoints
@app.get("/")
def hello_world():
    return {"message": "Hello World!"}

@app.head("/v1/keep-alive")
def health_check():
        return {"status": "healthy"}

@app.post("/v1/ingest")
def ingest(payload: IngestPayload):
        
        
        chunks = text_splitter.split_text(payload.info)

        records = [
            {
                "id": f"{payload.destinationId}-{i}",
                "text": chunk,
                'destinationId': payload.destinationId,
   
            } for i, chunk in enumerate(chunks)
        ]

        # Batch size of 90 (below Pinecone's limit of 96)
        batch_size = 90
    
        # Split records into batches and upsert
        for i in range(0, len(records), batch_size):
            batch = records[i:i + batch_size]
            index.upsert_records(payload.cityId, batch)

        return {"status": "done", "chunks_processed": len(records)}

@app.post("/v1/question")
def question(payload: QuestionPayload):
    # Define the query

    # Search the dense index
    results = index.search(
        namespace=payload.cityId,
        query={
            "top_k": 10,
            "inputs": {
                'text': payload.query
            }
        }
    )

    # Print the results
    for hit in results['result']['hits']:
            print(f"id: {hit['_id']:<5} | destinationId: {hit['fields']['destinationId']} | text: {hit['fields']['text']:<50}")
            

    chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": (
                "### 📘 Yêu cầu:\n"
                f"Trả lời câu hỏi sau bằng cách dựa trên các đoạn văn bên dưới. "
                "Nếu thông tin không đủ, hãy trả lời dựa trên kiến thức của bạn và ghi rõ điều đó.\n\n"
                f"**Câu hỏi:** {payload.query}\n\n"
                "### 📚 Đoạn văn tham khảo:\n"
                # + "\n---\n".join([hit['fields']['text'] for hit in results['result']['hits']]) +
                # "\n\n"
                + "\n---\n".join([
                     f"**Đoạn văn {i+1}:**\n"
                     f"{hit['fields']['text']}\n"
                     for i, hit in enumerate(results['result']['hits'])
                     ]) +
                "### ✏️ Ghi chú khi trả lời:\n"
                "- Trình bày câu trả lời bằng [Markdown] để hệ thống `react-markdown` có thể hiển thị tốt.\n"
                "- Thêm emoji phù hợp để làm nổi bật nội dung chính 🧠📌💡.\n"
                "- Nếu câu trả lời không thể rút ra từ đoạn văn, hãy bắt đầu bằng câu: `⚠️ Không tìm thấy thông tin, câu trả lời được tạo từ kiến thức nền.`"
            )
        }
    ],
    model="llama-3.3-70b-versatile",
    )
    response_dict = chat_completion.model_dump()
    response_dict["choices"][0]["message"]["documents"] = [
        {
            "id": hit["_id"],
            "text": hit["fields"]["text"],
            "destinationId": hit["fields"]["destinationId"],
            "score": hit["_score"]
        } for hit in results['result']['hits']
    ]
    return response_dict

@app.post("/v1/delete-document")
def delete_document(payload: DeletePayload):

    ids_to_delete = list(index.list(prefix=payload.destiationId, namespace=payload.cityId))

    if not ids_to_delete:
        raise HTTPException(status_code=404, detail="Không tìm thấy vectors nào với documentId này.")

    # Bước 2: Xoá vector theo ID
    index.delete(
        namespace=payload.cityId,
        ids=ids_to_delete
    )

    return {"deleted_ids": ids_to_delete}

@app.post("/v1/chat/completions")
def create_chat_completion(payload: ChatCompletionPayload):

    if not payload.isUseKnowledge:
        try:
            # Convert Pydantic Message models to dictionaries if payload.messages contains them
            messages_for_api = [message.model_dump() for message in payload.messages]

            last_message = messages_for_api[-1] if messages_for_api else None
            # Remove the last message since we'll handle it separately
            messages_for_api = messages_for_api[:-1]

            chat_completion = client.chat.completions.create(
                messages=messages_for_api + [
                     {
                        "role": "user",
                        "content": (
                            "### 📘 Yêu cầu:\n"
                            f"Trả lời câu hỏi sau: {last_message['content']}\n\n"
                            "### ✏️ Ghi chú khi trả lời:\n"
                            "- Trình bày câu trả lời bằng [Markdown] để hệ thống `react-markdown` có thể hiển thị tốt.\n"
                            "- Thêm emoji phù hợp để làm nổi bật nội dung chính 🧠📌💡.\n" 
                            "- Nếu nội dung có thể so sánh hoặc phân loại, hãy sử dụng **bảng Markdown** để trình bày.\n"
                        )
                     }
                ],
                model=payload.model or "deepseek-r1-distill-llama-70b",  # Use model from payload or default
                # You can pass other parameters from payload to the API call if needed
                # e.g., temperature=payload.temperature
                temperature=0.5,
                max_completion_tokens=1024,
                top_p=1,
            )

            return chat_completion
        
        except Exception as e:
            print(f"Error during chat completion: {e}") # For server-side logging
            raise HTTPException(status_code=500, detail=str(e))
        
    else:
         
        messages_for_api = [message.model_dump() for message in payload.messages]

        # Clean the question for the query
        def clean_text(text: str) -> str:
            # 1. Chuyển về chữ thường
            # text = text.lower()

            # 2. Chuẩn hóa Unicode (dùng NFC để ghép dấu)
            text = unicodedata.normalize("NFC", text)

            # 3. Loại bỏ ký tự đặc biệt (giữ lại tiếng Việt và chữ số)
            text = re.sub(r"[^\w\sàáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩ"
                        r"òóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]", "", text)

            # 4. Loại bỏ khoảng trắng dư thừa
            text = re.sub(r"\s+", " ", text).strip()

            return text


        # Combine all previous user question into a single string for the query
        combined_question = [message.model_dump() for message in payload.messages]
        combined_question = [message for message in combined_question if message['role'] == 'user']
        combined_question = [message['content'] for message in combined_question]
        combined_question = " ".join(combined_question)
        combined_question = clean_text(combined_question)
        print(combined_question)




        # Search the dense index
        query = {
            "top_k": 15,
            "inputs": {
                # 'text': clean_text(payload.messages[len(payload.messages) - 1].content)
                'text': combined_question
            }
        }


        results = index.search(
            namespace=payload.cityId,
            query=query
        )
      
        # results = index.search(
        #     namespace=payload.userId,
        #     query={
        #         "top_k": 15,
        #         "inputs": {
        #             'text': clean_text(payload.messages[len(payload.messages) - 1].content)
        #         },
        #         "filter": {
        #             "courseId": payload.courseId
        #         } if payload.courseId else None
        #     },
        # )

        # Print the results
        # for hit in results['result']['hits']:
        #         print(f"id: {hit['_id']:<5} | documentId: {hit['fields']['documentId']} | title: {hit['fields']['title']} | score: {round(hit['_score'], 2):<5} | text: {hit['fields']['text']:<50}")
                

        chat_completion = client.chat.completions.create(
            messages=messages_for_api + [
                {
                    "role": "user",
                    "content": (
                        "### 📘 Yêu cầu:\n"
                        f"Trả lời câu hỏi sau bằng cách dựa trên các đoạn văn bên dưới. "
                        "Nếu thông tin không đủ, hãy trả lời dựa trên kiến thức của bạn và ghi rõ điều đó.\n\n"
                        f"**Câu hỏi:** {payload.messages[len(payload.messages) - 1].content}\n\n"
                        "### 📚 Đoạn văn tham khảo:\n"
                        + "\n---\n".join([
              
                            f"{hit['fields']['text']}\n"
                            for i, hit in enumerate(results['result']['hits'])
                        ]) +
                        "### ✏️ Ghi chú khi trả lời:\n"
                        "- Trình bày câu trả lời bằng [Markdown] để hệ thống `react-markdown` có thể hiển thị tốt.\n"
                        "- Đảm bảo mỗi thông tin được trích dẫn đều có tham chiếu đến **Document title** tương ứng (ví dụ: `[Python đại cương]` chỉ cần tựa của tài liệu gốc, không cần ghi đoạn văn nào, không nhắc lại 'Document title' và không nhắc lại tựa tài liệu nếu bị lặp).\n"
                        "- Thêm emoji phù hợp để làm nổi bật nội dung chính 🧠📌💡.\n"
                        "- Nếu nội dung có thể so sánh hoặc phân loại, hãy sử dụng **bảng Markdown** để trình bày.\n"
                        "- Nếu câu trả lời không thể rút ra từ đoạn văn, hãy bắt đầu bằng câu: `⚠️ Không tìm thấy thông tin trong đoạn văn, câu trả lời được tạo từ kiến thức nền.`\n"  
                    )
                }
            ],
            model=payload.model or "deepseek-r1-distill-llama-70b",
        )

        response_dict = chat_completion.model_dump()

        response_dict["choices"][len(response_dict["choices"])-1]["message"]["destinations"] = [
            {
                "id": hit["_id"],
                "text": hit["fields"]["text"],
                "destinationId": hit["fields"]["destinationId"],
                "score": hit["_score"]
            } for hit in results['result']['hits']
        ]
        return response_dict
    


    