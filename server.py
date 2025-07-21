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


        # Add slug and name to each record for downstream use
        records = [
            {
                "id": f"{payload.destinationId}-{i}",
                "text": chunk,
                'destinationId': payload.destinationId,
                'slug': getattr(payload, 'slug', None),
                'name': getattr(payload, 'name', None),
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
                "- Nếu câu trả lời không thể rút ra từ đoạn văn, hãy bắt đầu bằng câu: `Dưới đây là một số gợi ý của tôi, để có thể nhận gợi ý chính xác hơn từ hệ thống, vui lòng chọn điểm đến trước nhé`"
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

    notice = "👆 Nhớ chọn điểm đến phía trên trước khi hỏi để Gobot gợi ý chính xác từ hệ thống nheee!"
    if not payload.isUseKnowledge or not payload.cityId:
        try:
            messages_for_api = [message.model_dump() for message in payload.messages]
            last_message = messages_for_api[-1] if messages_for_api else None
            messages_for_api = messages_for_api[:-1]
            chat_completion = client.chat.completions.create(
                messages=messages_for_api + [
                    {
                        "role": "user",
                        "content": (
                            "🌏 Xin chào!\n"
                            "Bạn hỏi: " + (last_message['content'] if last_message else "") + "\n\n"
                            "🤖 Dưới đây là câu trả lời dựa trên kiến thức nền của hệ thống.\n"
                            "Nếu bạn muốn nhận gợi ý từ hệ thống chính xác hơn, hãy chọn điểm đến trước nhé! 👆\n\n"
                            "---\n"
                            "### Trả lời:"
                        )
                    }
                ],
                model=payload.model or "deepseek-r1-distill-llama-70b",
                temperature=0.5,
                max_completion_tokens=1024,
                top_p=1,
            )
            response_dict = chat_completion.model_dump()
            # Ghép notice vào đầu content trả về cho FE (không lặp)
            if response_dict.get("choices") and response_dict["choices"][0].get("message"):
                content = response_dict["choices"][0]["message"].get("content")
                # Nếu content là string, luôn prepend notice
                if isinstance(content, str):
                    response_dict["choices"][0]["message"]["content"] = notice + "\n\n" + content.strip()
                else:
                    # Nếu content không phải string hoặc không có, chỉ trả về notice
                    response_dict["choices"][0]["message"]["content"] = notice
            return response_dict
        except Exception as e:
            print(f"Error during chat completion: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    # Nếu có cityId thì giữ nguyên logic cũ
    messages_for_api = [message.model_dump() for message in payload.messages]

    # Clean the question for the query
    # Đưa hàm clean_text ra ngoài
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
            'text': combined_question
        }
    }

    results = index.search(
        namespace=payload.cityId,
        query=query
    )

    chat_completion = client.chat.completions.create(
        messages=messages_for_api + [
            {
                "role": "user",
                    "content": (
                        "👋 **Bạn đang là trợ lý du lịch thông minh Gobot!**\n\n"
                        "Nhiệm vụ của bạn là giúp người dùng tìm kiếm và lựa chọn địa điểm du lịch phù hợp nhất.\n\n"
                        f"### ❓ Câu hỏi từ người dùng:\n"
                        f"{payload.messages[-1].content}\n\n"
                        "---\n"
                        "📚 **Thông tin tham khảo từ hệ thống:**\n"
                        "Dưới đây là các đoạn mô tả về địa điểm du lịch có liên quan. Hãy sử dụng chúng để đưa ra gợi ý chính xác:\n\n"
                        + "\n---\n".join([
                            f"{hit['fields']['text']}\n"
                            for i, hit in enumerate(results['result']['hits'])
                        ]) +
                        "\n---\n"
                        "✏️ **Hướng dẫn trình bày câu trả lời:**\n"
                        "- Viết bằng cú pháp [Markdown] để hệ thống có thể hiển thị đẹp.\n"
                        "- Đưa ra **gợi ý cụ thể, ngắn gọn và rõ ràng**.\n"
                        "- Nếu có thể, hãy liệt kê các lựa chọn bằng **danh sách hoặc bảng Markdown**.\n"
                        "- Thêm emoji để tạo cảm giác thân thiện và trực quan 🏝️🏔️🌇.\n"
                        "- Nếu thông tin không có trong dữ liệu, hãy dựa vào kiến thức nền, và mở đầu bằng: `⚠️ Gợi ý dựa trên kiến thức nền:`\n"
                    )
            }
        ],
        model=payload.model or "deepseek-r1-distill-llama-70b",
    )

    response_dict = chat_completion.model_dump()


    # Limit the number of unique destination links shown (e.g., top 3)
    MAX_LINKS = 3
    seen = set()
    link_lines = []
    for idx, hit in enumerate(results['result']['hits']):
        slug = hit['fields'].get('slug')
        dest_id = hit['fields'].get('destinationId')
        name = hit['fields'].get('name') or f"Địa điểm {dest_id}"
        unique_key = slug or dest_id
        if unique_key and unique_key not in seen:
            seen.add(unique_key)
            if slug:
                url = f"http://localhost:3000/destination/{slug}"
            else:
                url = f"http://localhost:3000/destination/{dest_id}"
            link_lines.append(f"{len(link_lines)+1}. {name}\n   Xem chi tiết tại: {url}")
        if len(link_lines) >= MAX_LINKS:
            break

    # Append the markdown list to the end of the answer content
    if response_dict.get("choices") and response_dict["choices"][-1].get("message"):
        content = response_dict["choices"][-1]["message"].get("content", "")
        if link_lines:
            section = "\n\n---\n**🔗 Đường dẫn tới các địa điểm được gợi ý:**\n" + "\n".join(link_lines)
            content = content.rstrip() + section
            response_dict["choices"][-1]["message"]["content"] = content

    response_dict["choices"][len(response_dict["choices"])-1]["message"]["destinations"] = [
        {
            "id": hit["_id"],
            "text": hit["fields"]["text"],
            "destinationId": hit["fields"].get("destinationId"),
            "slug": hit["fields"].get("slug"),
            "name": hit["fields"].get("name"),
            "score": hit["_score"]
        } for hit in results['result']['hits']
    ]
    return response_dict
# Đưa hàm clean_text ra ngoài
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
    


    