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
    model=payload.model or "deepseek-r1-distill-llama-70b",
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
    """
    Tạo chat completion cho Gobot
    - Nếu không dùng knowledge hoặc không có cityId -> trả lời dựa trên kiến thức nền
    - Nếu có knowledge -> tìm kiếm Pinecone theo intent, AI trả lời, link địa điểm được ghép xuống cuối
    """
    notice = "👆 Nhớ chọn điểm đến phía trên trước khi hỏi để Gobot gợi ý chính xác từ hệ thống nheee!"

    # ==================================================
    # 1️⃣ Trường hợp không dùng knowledge
    # ==================================================
    if not payload.isUseKnowledge or not payload.cityId:
        try:
            messages_for_api = [msg.model_dump() for msg in payload.messages]
            last_message = messages_for_api[-1] if messages_for_api else None

            system_prompt = (
                "🌏 Xin chào!\n"
                f"Bạn hỏi: {last_message['content'] if last_message else ''}\n\n"
                "🤖 Đây là câu trả lời dựa trên kiến thức nền của hệ thống.\n"
                "Nếu muốn nhận gợi ý chính xác hơn, hãy chọn điểm đến trước nhé! 👆\n\n"
                "---\n### Trả lời:"
            )

            chat_completion = client.chat.completions.create(
                messages=messages_for_api[:-1] + [{"role": "user", "content": system_prompt}],
                model=payload.model or "deepseek-r1-distill-llama-70b",
                temperature=0.3,
                top_p=0.9,
                max_completion_tokens=1024,
            )

            response_dict = chat_completion.model_dump()
            # Ghép notice vào đầu câu trả lời
            if response_dict.get("choices") and response_dict["choices"][0].get("message"):
                content = response_dict["choices"][0]["message"].get("content", "")
                response_dict["choices"][0]["message"]["content"] = notice + "\n\n" + (content or "").strip()
            return response_dict

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error during chat completion: {str(e)}")

    # ==================================================
    # 2️⃣ Trường hợp dùng knowledge
    # ==================================================
    messages_for_api = [msg.model_dump() for msg in payload.messages]

    # Làm sạch câu hỏi để search Pinecone
    combined_question = " ".join([msg.content for msg in payload.messages if msg.role == "user"])
    combined_question = clean_text(combined_question)

    results = index.search(
        namespace=payload.cityId,
        query={"top_k": 20, "inputs": {"text": combined_question}},
    )
    hits = results.get("result", {}).get("hits", [])

    # ==================================================
    # 3️⃣ Xác định intent
    # ==================================================
    user_question = payload.messages[-1].content.lower()
    intent_keywords = {
        "quán ăn": ["quán ăn", "ăn uống", "món ăn", "ẩm thực địa phương"],
        "nhà hàng": ["nhà hàng", "restaurant", "ăn uống"],
        "quán cà phê": ["cà phê", "coffee", "quán cafe", "quán cà phê"],
        "bảo tàng": ["bảo tàng", "museum", "trưng bày"],
        "khu vui chơi": ["khu vui chơi", "công viên giải trí", "vui chơi", "giải trí"],
    }

    detected_intent = None
    for intent, keywords in intent_keywords.items():
        if any(kw in user_question for kw in keywords):
            detected_intent = intent
            break

    # ==================================================
    # 4️⃣ Lọc hits theo intent
    # ==================================================
    if detected_intent:
        filtered_hits = []
        for hit in hits:
            type_val = (hit["fields"].get("type") or hit["fields"].get("category") or "").lower()
            name_val = (hit["fields"].get("name") or "").lower()

            if detected_intent == "quán ăn":
                if "food" in type_val or "ăn" in type_val or "ẩm thực" in type_val or "quán ăn" in name_val:
                    filtered_hits.append(hit)

            elif detected_intent == "quán cà phê":
                if any(k in type_val for k in ["cafe", "coffee", "cà phê"]) \
                   or any(k in name_val for k in ["cà phê", "cafe", "coffee"]):
                    filtered_hits.append(hit)

            elif detected_intent == "bảo tàng":
                if "museum" in type_val or "bảo tàng" in type_val or "museum" in name_val or "bảo tàng" in name_val:
                    filtered_hits.append(hit)

            elif detected_intent == "khu vui chơi":
                if any(k in type_val for k in ["park", "amusement", "vui chơi", "giải trí"]) \
                   or any(k in name_val for k in ["khu vui chơi", "công viên", "giải trí"]):
                    filtered_hits.append(hit)

        if filtered_hits:
            hits = filtered_hits

    if not hits:
        return {"choices": [{"message": {"content": notice + "\n\n⚠️ Không tìm thấy dữ liệu phù hợp."}}]}

    # ==================================================
    # 5️⃣ Chuẩn bị danh sách link gọn gàng
    # ==================================================
    MAX_LINKS = 5
    seen = set()
    link_lines = []

    for hit in hits:
        slug = hit["fields"].get("slug")
        dest_id = hit["fields"].get("destinationId")
        name = hit["fields"].get("name") or f"Địa điểm {dest_id}"
        unique_key = slug or dest_id
        if unique_key and unique_key not in seen:
            seen.add(unique_key)
            url = f"http://localhost:3000/destination/{slug or dest_id}"
            link_lines.append(f"- [{name}]({url})")
        if len(link_lines) >= MAX_LINKS:
            break

    # ==================================================
    # 6️⃣ Chuẩn bị prompt cho AI
    # ==================================================
    reference_texts = "\n---\n".join([hit["fields"]["text"] for hit in hits])
    prompt = (
        f"👋 **Xin chào, mình là Gobot - trợ lý du lịch thông minh của bạn!**\n\n"
        f"Mình sẽ giúp bạn tìm các địa điểm thuộc loại **{detected_intent or 'du lịch/ăn uống'}** hấp dẫn nhất dựa trên dữ liệu hệ thống.\n\n"
        f"### ❓ Câu hỏi từ người dùng:\n{payload.messages[-1].content}\n\n"
        "---\n"
        "📚 **Thông tin tham khảo từ hệ thống:**\n"
        f"{reference_texts}\n"
        "---\n"
        "✏️ **Hướng dẫn trả lời:**\n"
        "- Chỉ gợi ý các địa điểm đúng loại yêu cầu, dựa trên dữ liệu hệ thống.\n"
        "- Viết câu trả lời thân thiện, tự nhiên, như một người bạn địa phương đang tư vấn.\n"
        "- Kiểm tra kỹ thời gian mở cửa của địa điểm trước khi trả lời, xem địa điểm có mở cửa vào thời điểm người dùng hỏi không.\n"
        "- Không tự chế tên địa điểm nếu không có trong dữ liệu.\n"
        "- Không chèn link trong nội dung, link sẽ hiển thị riêng ở cuối.\n"
        "- Nếu không đủ dữ liệu, mở đầu bằng: `⚠️ Gợi ý dựa trên kiến thức nền:`\n"
        "- Kết thúc bằng lời chúc du lịch vui vẻ.\n\n"
    )

    # ==================================================
    # 7️⃣ Gọi model
    # ==================================================
    chat_completion = client.chat.completions.create(
        messages=messages_for_api + [{"role": "user", "content": prompt}],
        model=payload.model or "deepseek-r1-distill-llama-70b",
        temperature=0.5,
        top_p=0.8,
        max_completion_tokens=600,
    )
    response_dict = chat_completion.model_dump()

    # ==================================================
    # 8️⃣ Ghép link vào cuối câu trả lời (chỉ 1 lần)
    # ==================================================
    if response_dict.get("choices") and response_dict["choices"][-1].get("message"):
        content = response_dict["choices"][-1]["message"].get("content", "")
        import re
        # Xóa section link cũ nếu AI tự thêm
        content = re.sub(
            r"\n+---\n\*\*🔗 Đường dẫn tới các địa điểm được gợi ý:\*\*[\s\S]*$",
            "",
            content,
            flags=re.MULTILINE,
        )

        # Ưu tiên link khớp với tên xuất hiện trong câu trả lời
        filtered_links = []
        lower_content = content.lower()
        seen_links = set()
        for hit in hits:
            name = (hit["fields"].get("name") or f"Địa điểm {hit['fields'].get('destinationId')}").lower()
            slug = str(hit["fields"].get("slug") or "").lower()
            unique_key = slug or name
            if (name in lower_content or slug in lower_content) and unique_key not in seen_links:
                url = f"http://localhost:3000/destination/{slug or hit['fields'].get('destinationId')}"
                filtered_links.append(
                    f"- [{hit['fields'].get('name') or f'Địa điểm {hit['fields'].get('destinationId')}'}]({url})"
                )
                seen_links.add(unique_key)

        final_links = filtered_links or link_lines
        if final_links:
            section = "\n\n---\n**🔗 Đường dẫn tới các địa điểm được gợi ý:**\n" + "\n".join(final_links)
            response_dict["choices"][-1]["message"]["content"] = content.rstrip() + section

    # ==================================================
    # 9️⃣ Gắn danh sách địa điểm vào JSON trả về
    # ==================================================
    response_dict["choices"][-1]["message"]["destinations"] = [
        {
            "id": hit["_id"],
            "text": hit["fields"]["text"],
            "destinationId": hit["fields"].get("destinationId"),
            "slug": hit["fields"].get("slug"),
            "name": hit["fields"].get("name"),
            "score": hit["_score"],
        } for hit in hits
    ]

    return response_dict



# clean_text
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
    


    