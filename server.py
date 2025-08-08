from fastapi import FastAPI, Request, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import requests
from pinecone import Pinecone, ServerlessSpec
from groq import Groq
from langchain.text_splitter import RecursiveCharacterTextSplitter
import re
import unicodedata
import os
from dotenv import load_dotenv

from models import IngestPayload, QuestionPayload, DeletePayload, ChatCompletionPayload, UpdatePayload

# Load environment variables
load_dotenv()
app = FastAPI()

# Thêm exception handler cho validation errors
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    print(f"[VALIDATION ERROR] {exc.errors()}")
    print(f"[REQUEST BODY] {await request.body()}")
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors(), "body": str(await request.body())}
    )

# Pinecone init
pc = Pinecone(
    api_key=os.getenv("PINECONE_API_KEY")
)

index_name = os.getenv("PINECONE_INDEX_NAME")

# Groq init
client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)

# Define the text splitter với semantic separators cho du lịch
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,  # Tăng size để giữ nguyên thông tin
    chunk_overlap=100,  # Tăng overlap để giữ context
    separators=[
        "\n### ",  # Phân chia theo section headers
        "\n## ",   # Headers nhỏ hơn
        "\n- ",    # List items
        ".\n",     # Kết thúc câu + newline
        ". ",      # Kết thúc câu
        "\n",      # Newline
        " "        # Space (fallback)
    ]
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
        
        # Parse JSON data từ backend
        import json
        try:
            destination_data = json.loads(payload.info)
        except:
            # Fallback nếu vẫn là string cũ
            destination_data = {"description": payload.info}
        
        # Semantic chunking theo loại destination
        chunks = create_semantic_chunks(payload.name, destination_data, payload.destinationId, payload.slug)

        # Tạo records với metadata đầy đủ
        records = [
            {
                "id": f"{payload.destinationId}-{chunk['type']}",
                "text": chunk['content'],
                'destinationId': payload.destinationId,
                'slug': payload.slug,
                'name': payload.name,
                'chunk_type': chunk['type'],
                'chunk_index': i,
                'total_chunks': len(chunks)
            } for i, chunk in enumerate(chunks)
        ]

        # Batch size of 90 (below Pinecone's limit of 96)
        batch_size = 90
    
        # Split records into batches and upsert
        for i in range(0, len(records), batch_size):
            batch = records[i:i + batch_size]
            index.upsert_records(payload.cityId, batch)

        return {"status": "done", "chunks_processed": len(records)}

@app.post("/v1/update")
def update_destination(payload: UpdatePayload):
    try:

        
        # Bước 1: Tìm và xóa các chunks cũ từ TẤT CẢ namespaces
        # Vì có thể cityId đã thay đổi, ta cần tìm trong tất cả namespaces
        deleted_count = 0
        
        # Lấy danh sách tất cả namespaces
        try:
            stats = index.describe_index_stats()
            all_namespaces = list(stats.get('namespaces', {}).keys())
            
            # Tìm và xóa chunks cũ trong tất cả namespaces
            for namespace in all_namespaces:
                try:
                    ids_to_delete = list(index.list(prefix=payload.destinationId, namespace=namespace))
                    if ids_to_delete:
                        index.delete(namespace=namespace, ids=ids_to_delete)
                        deleted_count += len(ids_to_delete)
                except Exception as ns_error:
                    continue
                    
        except Exception as stats_error:
            # Fallback: chỉ xóa từ namespace hiện tại
            try:
                ids_to_delete = list(index.list(prefix=payload.destinationId, namespace=payload.cityId))
                if ids_to_delete:
                    index.delete(namespace=payload.cityId, ids=ids_to_delete)
                    deleted_count = len(ids_to_delete)
            except Exception as fallback_error:
                pass
        
        # Bước 2: Parse JSON data từ backend (giống như ingest)
        import json
        try:
            destination_data = json.loads(payload.info)
            print(f"[UPDATE] 📍 Đang update địa điểm: {payload.name}")
        except Exception as parse_error:
            destination_data = {"description": payload.info}
        
        # Bước 3: Tạo semantic chunks mới với 4 chunks
        chunks = create_semantic_chunks(payload.name, destination_data, payload.destinationId, payload.slug)

        # Bước 4: Tạo records mới với metadata đầy đủ
        records = [
            {
                "id": f"{payload.destinationId}-{chunk['type']}",
                "text": chunk['content'],
                'destinationId': payload.destinationId,
                'slug': payload.slug,
                'name': payload.name,
                'chunk_type': chunk['type'],
                'chunk_index': i,
                'total_chunks': len(chunks)
            } for i, chunk in enumerate(chunks)
        ]

        # Bước 5: Upsert records mới vào namespace mới (cityId từ payload)
        batch_size = 90
        
        for i in range(0, len(records), batch_size):
            batch = records[i:i + batch_size]
            index.upsert_records(payload.cityId, batch)

        print(f"[UPDATE] ✅ UPDATE THÀNH CÔNG: {payload.name}")
        
        return {
            "status": "updated", 
            "chunks_deleted": deleted_count,
            "chunks_created": len(records),
            "new_namespace": payload.cityId,
            "destination_name": payload.name
        }
        
    except Exception as e:
        print(f"[UPDATE ERROR] ❌ UPDATE THẤT BẠI cho {payload.name}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Update failed: {str(e)}")

@app.post("/v1/question")
def question(payload: QuestionPayload):
    # Define the query

    # Search the dense index với tăng top_k để bao phủ tốt hơn
    results = index.search(
        namespace=payload.cityId,
        query={
            "top_k": 12,  # Tăng lên vì mỗi destination có 4 chunks
            "inputs": {
                'text': payload.query
            }
        }
    )

    # Print the results với thông tin chunk type
    for hit in results['result']['hits']:
            chunk_type = hit['fields'].get('chunk_type', 'unknown')
            print(f"id: {hit['_id']:<5} | type: {chunk_type:<10} | destinationId: {hit['fields']['destinationId']} | text: {hit['fields']['text'][:50]}")
            

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

# Hàm clean_text để xử lý văn bản
def clean_text(text: str) -> str:
    """
    Làm sạch và chuẩn hóa văn bản tiếng Việt
    """
    # 1. Chuẩn hóa Unicode (dùng NFC để ghép dấu)
    text = unicodedata.normalize("NFC", text)

    # 2. Loại bỏ ký tự đặc biệt (giữ lại tiếng Việt và chữ số)
    text = re.sub(r"[^\w\sàáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩ"
                r"òóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]", "", text)

    # 3. Loại bỏ khoảng trắng dư thừa
    text = re.sub(r"\s+", " ", text).strip()

    return text

def create_semantic_chunks(name: str, data: dict, destination_id: str, slug: str = None) -> list:
    """
    Tạo 4 chunks semantic cho mỗi destination theo strategy mới
    """
    chunks = []
    
    # 1. Tổng quan - Thông tin tổng quan (bắt đầu bằng tên địa điểm)
    overview_content = f"**{name}**\n\n"
    
    # Mô tả chính
    if data.get('description'):
        overview_content += f"{data['description']}\n\n"
    
    # Điểm nổi bật
    if data.get('highlight'):
        overview_content += f"**Điểm nổi bật:** {data['highlight']}\n\n"
    
    chunks.append({
        'type': 'tong-quan',
        'content': overview_content.strip()
    })
    
    # 2. Trải nghiệm - Trải nghiệm và hoạt động (bắt đầu bằng tên địa điểm)
    experience_content = f"**{name}**\n\n"
    
    # Dịch vụ
    if data.get('services'):
        experience_content += f"**Dịch vụ:** {data['services']}\n\n"
    
    # Hoạt động
    if data.get('activities'):
        experience_content += f"**Hoạt động:** {data['activities']}\n\n"
    
    # Thông tin hữu ích
    if data.get('usefulInfo'):
        experience_content += f"**Thông tin hữu ích:** {data['usefulInfo']}\n\n"
    
    chunks.append({
        'type': 'trai-nghiem',
        'content': experience_content.strip()
    })
    
    # 3. Thực tế - Thông tin thực tế (bắt đầu bằng tên địa điểm)
    practical_content = f"**{name}**\n\n"
    
    # Giờ mở cửa
    if data.get('openHour'):
        practical_content += f"**Giờ mở cửa:** {data['openHour']}\n\n"
    
    # Phí tham quan
    if data.get('fee'):
        practical_content += f"**Phí tham quan:** {data['fee']}\n\n"
    
    # Thông tin liên hệ
    if data.get('contactInfo'):
        practical_content += f"**Liên hệ:** {data['contactInfo']}\n\n"
    
    chunks.append({
        'type': 'thuc-te',
        'content': practical_content.strip()
    })
    
    # 4. Danh mục - Tags và từ khóa tìm kiếm (bắt đầu bằng tên địa điểm)
    tags_content = f"**{name}**\n\n"
    
    # Tags chính - làm nổi bật để AI dễ nhận biết
    if data.get('tags'):
        tags_content += f"🏷️ **DANH MỤC/TAGS:** {data['tags']}\n\n"
    
    # Loại hình du lịch - chuyển từ chunk tổng quan
    if data.get('cultureType'):
        tags_content += f"🎯 **LOẠI HÌNH DU LỊCH:** {data['cultureType']}\n\n"
    
    # Thêm thông tin phân loại để AI dễ phân tích
    if data.get('type'):
        tags_content += f"📂 **PHÂN LOẠI:** {data['type']}\n\n"

    chunks.append({
        'type': 'danh-muc',
        'content': tags_content.strip()
    })
    
    return chunks

def filter_destinations_by_content(content: str, hits: list) -> list:
    """
    Lọc destinations dựa trên việc tên địa điểm có xuất hiện trong nội dung câu trả lời hay không
    Sử dụng multiple matching strategies để tăng độ chính xác
    """
    if not content or not hits:
        return hits
    
    # Chuẩn hóa content để so sánh
    content_normalized = clean_text(content.lower())
    
    # Tạo set để tránh trùng lặp destinations
    mentioned_destination_ids = set()
    filtered_hits = []
    
    for hit in hits:
        dest_name = hit["fields"].get("name", "").strip()
        dest_id = hit["fields"].get("destinationId")
        
        if not dest_name or not dest_id:
            continue
            
        # Chuẩn hóa tên địa điểm để so sánh
        dest_name_normalized = clean_text(dest_name.lower())
        
        # Strategy 1: Exact match
        is_mentioned = dest_name_normalized in content_normalized
        
        # Strategy 2: Partial word match (tránh false positive)
        if not is_mentioned:
            # Tách từ và kiểm tra từng từ quan trọng
            dest_words = [word for word in dest_name_normalized.split() if len(word) > 2]
            if dest_words:
                # Phải có ít nhất 70% từ xuất hiện trong content
                matched_words = sum(1 for word in dest_words if word in content_normalized)
                word_match_ratio = matched_words / len(dest_words)
                is_mentioned = word_match_ratio >= 0.7
        
        # Strategy 3: Common abbreviations và alternative names
        if not is_mentioned:
            # Kiểm tra các pattern phổ biến
            name_patterns = [
                dest_name_normalized.replace(" ", ""),  # Loại bỏ khoảng trắng
                dest_name_normalized.replace("quán", "").strip(),  # Loại bỏ "quán"
                dest_name_normalized.replace("nhà hàng", "").strip(),  # Loại bỏ "nhà hàng"
                dest_name_normalized.replace("khách sạn", "").strip(),  # Loại bỏ "khách sạn"
                dest_name_normalized.replace("cà phê", "coffee").strip(),  # Thay thế cà phê
            ]
            
            for pattern in name_patterns:
                if pattern and len(pattern) > 2 and pattern in content_normalized:
                    is_mentioned = True
                    break
        
        if is_mentioned:
            # Chỉ thêm nếu chưa có destination này
            if dest_id not in mentioned_destination_ids:
                mentioned_destination_ids.add(dest_id)
                filtered_hits.append(hit)
                print(f"[FILTER] ✅ Giữ lại: {dest_name} (xuất hiện trong câu trả lời)")
            else:
                print(f"[FILTER] ⚠️ Bỏ qua duplicate: {dest_name}")
        else:
            print(f"[FILTER] ❌ Loại bỏ: {dest_name} (không xuất hiện trong câu trả lời)")
    
    print(f"[FILTER] 📊 Kết quả: {len(filtered_hits)}/{len(hits)} destinations được giữ lại")
    return filtered_hits

@app.post("/v1/chat/completions")
def create_chat_completion(payload: ChatCompletionPayload):
    """
    Chat completion cho Gobot - trợ lý du lịch Việt Nam
    """
    # Thông báo chung khi chưa chọn thành phố
    notice = (
        "👋 Xin chào! Hiện tại bạn **chưa chọn thành phố** hoặc điểm đến cụ thể.\n"
        "Để nhận gợi ý **chính xác từ hệ thống**, hãy chọn một thành phố trước nhé! 🏙️\n"
    )

    # -----------------------
    # 1️⃣ Không dùng Knowledge
    # -----------------------
    if not payload.isUseKnowledge or not payload.cityId:
        messages_for_api = [msg.model_dump() for msg in payload.messages]
        user_question = messages_for_api[-1]["content"] if messages_for_api else ""

        system_prompt = (
            "Bạn là **Gobot**, trợ lý du lịch thông minh và thân thiện của website **GoOhNo**, nền tảng hỗ trợ lên kế hoạch du lịch Việt Nam.\n"
            "Bạn đóng vai trò như một **hướng dẫn viên bản địa**, trò chuyện tự nhiên và gần gũi để giúp người dùng khám phá Việt Nam dễ dàng.\n"
            "Quy tắc quan trọng:\n"
            "1. Chỉ tư vấn về các địa điểm, hoạt động và trải nghiệm du lịch tại Việt Nam.\n"
            "2. Trả lời bằng **tiếng Việt**, giọng điệu thân thiện, gần gũi, dễ hiểu.\n"
            "3. Trình bày bằng **Markdown** với tiêu đề, danh sách và emoji minh họa (📍🏖️☕🍜🏯).\n"
            "4. Nếu không chắc chắn, hãy nói: *Tôi không chắc về điều này.*\n"
            "5. Cuối câu trả lời, thêm **lời khuyên hữu ích cho du khách** và nhắc nhẹ rằng họ có thể tìm hiểu thêm trên GoOhNo.\n"
        )


        user_prompt = (
            f"\"Câu hỏi của người dùng: {user_question}\"\n"
            "\"Hãy trả lời dựa trên kiến thức nền về du lịch Việt Nam.\"\n"
            "\"Đưa ra các gợi ý chi tiết, dễ đọc, kèm emoji minh họa.\"\n"
            "\"Chia nhỏ nội dung thành mục hoặc danh sách Markdown.\"\n"
            "\"Kết thúc bằng lời khuyên hữu ích và thân thiện.\"\n"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            *messages_for_api,
            {"role": "user", "content": user_prompt},
        ]

        fallback_models = [
            payload.model or "llama-3.3-70b-versatile",
            "llama-3.1-70b-versatile",
            "mixtral-8x7b-32768"
        ]

        response_dict = None
        last_error = None

        for attempt, model in enumerate(fallback_models):
            try:
                print(f"[NO-KNOWLEDGE ATTEMPT {attempt+1}] Model: {model}")
                chat_completion = client.chat.completions.create(
                    messages=messages,
                    model=model,
                    temperature=0.4,
                    top_p=0.9,
                    max_completion_tokens=1024,
                )
                response_dict = chat_completion.model_dump()
                print(f"[SUCCESS] Model {model} worked!")
                break
            except Exception as e:
                last_error = e
                if attempt < len(fallback_models) - 1:
                    print(f"[FALLBACK] Error with {model}: {str(e)}")
                    continue

        if response_dict is None:
            raise HTTPException(status_code=500, detail=f"All fallback models failed. Last error: {str(last_error)}")

        # Làm sạch nội dung trả lời
        if response_dict.get("choices") and response_dict["choices"][0].get("message"):
            content = response_dict["choices"][0]["message"]["content"]
            import re
            content = re.sub(r'<thinking>.*?</thinking>', '', content, flags=re.DOTALL)
            content = re.sub(r'(Alright|First|I need|The user).*?(?=\n\n|\n[A-ZÀ-Ỹ])', '', content, flags=re.DOTALL)
            content = re.sub(r'\n\s*\n\s*\n', '\n\n', content).strip()
            response_dict["choices"][0]["message"]["content"] = f"{notice}\n{content}"

        return response_dict

    # -----------------------
    # 2️⃣ Dùng Knowledge
    # -----------------------
    messages_for_api = [msg.model_dump() for msg in payload.messages]
    combined_question = clean_text(
        " ".join([msg.content for msg in payload.messages if msg.role == "user"])
    )

    results = index.search(
        namespace=payload.cityId,
        query={"top_k": 12, "inputs": {"text": combined_question}},  # Tăng top_k cho 4 chunks mỗi destination
    )
    hits = results.get("result", {}).get("hits", [])

    # Lọc quán cà phê
    user_question = payload.messages[-1].content.lower()
    cafe_keywords = ["cà phê", "coffee", "quán cafe", "quán cà phê", "cafe 24h"]
    if any(kw in user_question for kw in cafe_keywords):
        filtered_hits = [
            hit for hit in hits
            if "cafe" in (hit["fields"].get("type","") + hit["fields"].get("category","")).lower()
            or "cà phê" in (hit["fields"].get("type","") + hit["fields"].get("category","")).lower()
            or "coffee" in (hit["fields"].get("name","")).lower()
            or "cà phê" in (hit["fields"].get("name","")).lower()
        ]
        if filtered_hits:
            hits = filtered_hits

    if not hits:
        return {
            "choices": [
                {"message": {"content": f"{notice}\n⚠️ Không tìm thấy dữ liệu phù hợp."}}
            ]
        }

    # -----------------------
    # 3️⃣ Chuẩn bị prompt
    # -----------------------
    # Trích xuất tên địa điểm từ hits
    destination_names = []
    reference_texts_with_names = []
    
    for hit in hits:
        dest_name = hit["fields"].get("name", "")
        if dest_name and dest_name not in destination_names:
            destination_names.append(dest_name)
        
        # Đảm bảo text luôn có tên địa điểm
        text = hit["fields"]["text"]
        if dest_name and dest_name not in text:
            text = f"**{dest_name}**: {text}"
        reference_texts_with_names.append(text)
    
    reference_texts = "\n---\n".join(reference_texts_with_names)

    system_prompt = (
        "Bạn là **Gobot**, trợ lý du lịch Việt Nam thân thiện và hiểu biết 🇻🇳.\n"
        "Quy tắc:\n"
        "1. Chỉ tư vấn các địa điểm và trải nghiệm tại Việt Nam.\n"
        "2. Trả lời bằng tiếng Việt, giọng điệu tự nhiên, dễ gần, như đang trò chuyện.\n"
        "3. Trình bày bằng **Markdown** với tiêu đề, danh sách và emoji (📍☕🏖️🍜🏯).\n"
        "4. **QUAN TRỌNG**: Luôn sử dụng TÊN CHÍNH XÁC của địa điểm từ dữ liệu được cung cấp.\n"
        "5. **PHÂN TÍCH TAGS**: Đọc kỹ trường 'Danh mục' và 'Loại hình' của mỗi địa điểm để đánh giá mức độ phù hợp với yêu cầu.\n"
        "6. Nếu không chắc chắn, hãy nói: *Tôi không chắc về điều này.*\n"
        "7. Kết thúc câu trả lời bằng **lời khuyên hữu ích cho khách du lịch**.\n"
    )

    user_prompt = (
        f"\"Câu hỏi của người dùng: {payload.messages[-1].content}\"\n"
        "\"Dựa trên các thông tin tham khảo từ hệ thống, hãy trả lời đầy đủ và thân thiện.\"\n"
        "\n**HƯỚNG DẪN PHÂN TÍCH QUAN TRỌNG:**\n"
        "1. **Đọc kỹ tags và danh mục**: Xem xét trường 'Danh mục' và 'Loại hình' của mỗi địa điểm\n"
        "2. **Đánh giá độ phù hợp**: Chỉ đề xuất địa điểm có tags/danh mục phù hợp với yêu cầu\n"
        "3. **Ưu tiên theo mức độ phù hợp**: Sắp xếp địa điểm theo độ phù hợp từ cao đến thấp\n"
        "4. **Giải thích lý do**: Nêu rõ tại sao địa điểm phù hợp dựa trên tags/danh mục\n"
        "\n**VÍ DỤ PHÂN TÍCH:**\n"
        "- Nếu user hỏi về 'quán cà phê': Chỉ chọn địa điểm có tag 'cà phê', 'coffee', 'đồ uống'\n"
        "- Nếu user hỏi về 'ăn uống': Chọn địa điểm có tag 'ẩm thực', 'nhà hàng', 'món ăn'\n"
        "- Nếu user hỏi về 'du lịch văn hóa': Chọn địa điểm có tag 'văn hóa', 'lịch sử', 'truyền thống'\n"
        "\n"
        "\"""QUAN TRỌNG: Hãy kiểm tra thời gian mở cửa của địa điểm trước khi trả lời, nếu địa điểm đã đóng cửa thì loại địa điểm đó ra khỏi câu trả lời.\"\n"
        f"\"Các địa điểm có sẵn: {', '.join(destination_names)}\"\n"
        "\"Thông tin chi tiết:\" \n"
        f"{reference_texts}\n\n"
        "\"QUAN TRỌNG: \"\n"
        "\"- Hãy sử dụng CHÍNH XÁC tên địa điểm từ danh sách trên.\"\n"
        "\"- CHỈ đưa vào câu trả lời những địa điểm có tags/danh mục PHÙ HỢP với yêu cầu của user.\"\n"
        "\"- Trình bày dưới dạng danh sách Markdown với emoji, có tên địa điểm rõ ràng.\"\n"
        "\"- Giải thích ngắn gọn tại sao địa điểm phù hợp (dựa trên tags/danh mục).\"\n"
        "\"- Kết thúc bằng một lời khuyên hữu ích và thân thiện.\"\n"
        "\n**QUY TRÌNH PHÂN TÍCH TAGS:**\n"
        "\"1. Đọc từng địa điểm và tìm phần có emoji 🏷️ DANH MỤC/TAGS và 🎯 LOẠI HÌNH DU LỊCH\"\n"
        "\"2. So sánh tags với từ khóa trong câu hỏi của user (ví dụ: 'cà phê' khớp với tag 'coffee')\"\n"
        "\"3. Chỉ đưa vào câu trả lời những địa điểm có tags phù hợp >= 70%\"\n"
        "\"4. Sắp xếp theo mức độ phù hợp: Rất phù hợp > Phù hợp > Có thể phù hợp\"\n"
        "\"5. Trong câu trả lời, ghi rõ lý do chọn dựa trên tags (ví dụ: 'phù hợp với nhu cầu tìm cà phê')\"\n"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        *messages_for_api,
        {"role": "user", "content": user_prompt},
    ]

    fallback_models = [
        payload.model or "deepseek-r1-distill-llama-70b",
        "llama-3.3-70b-versatile",
        "llama-3.1-70b-versatile",
        "mixtral-8x7b-32768"
    ]

    response_dict = None
    last_error = None

    for attempt, model in enumerate(fallback_models):
        try:
            print(f"[ATTEMPT {attempt+1}] Model: {model}")
            chat_completion = client.chat.completions.create(
                messages=messages,
                model=model,
                temperature=0.2,
                top_p=0.85,
                max_completion_tokens=800,
            )
            response_dict = chat_completion.model_dump()
            print(f"[SUCCESS] Model {model} worked!")
            break
        except Exception as e:
            last_error = e
            print(f"[ERROR] {model}: {str(e)}")
            if attempt < len(fallback_models) - 1:
                continue

    if response_dict is None:
        raise HTTPException(status_code=500, detail=f"All models failed. Last error: {str(last_error)}")

    # Làm sạch output và lọc destinations dựa trên nội dung câu trả lời
    if response_dict.get("choices") and response_dict["choices"][-1].get("message"):
        content = response_dict["choices"][-1]["message"]["content"]
        import re
        content = re.sub(r'\n\s*\n\s*\n', '\n\n', content).strip()
        response_dict["choices"][-1]["message"]["content"] = content
        
        # Lọc destinations chỉ giữ lại những địa điểm được nhắc đến trong câu trả lời
        filtered_hits = filter_destinations_by_content(content, hits)
    else:
        # Fallback nếu không có content
        filtered_hits = hits

    response_dict["choices"][-1]["message"]["destinations"] = [
        {
            "id": hit.get("_id") or hit.get("id", ""),
            "text": hit["fields"]["text"],
            "destinationId": hit["fields"].get("destinationId"),
            "slug": hit["fields"].get("slug"),
            "name": hit["fields"].get("name"),
            "chunk_type": hit["fields"].get("chunk_type", "unknown"),
            "score": hit.get("_score", 0),
        } for hit in filtered_hits
    ]

    return response_dict
