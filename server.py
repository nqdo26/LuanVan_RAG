from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
import requests
from pinecone import Pinecone, ServerlessSpec
from groq import Groq
import re
import unicodedata
import os
from dotenv import load_dotenv
from pymongo import MongoClient
from bson import ObjectId

from models import (
    DestinationSearchPayload,
    DestinationSearchResponse,
    DestinationResult,
    ChatCompletionPayload
)

# Load environment variables
load_dotenv()
app = FastAPI()

# MongoDB init
mongo_client = MongoClient(os.getenv("MONGODB_URI", "mongodb://localhost:27017"))
db = mongo_client[os.getenv("MONGODB_DB", "travel_db")]
cities_collection = db.cities
destinations_collection = db.destinations

# Pinecone init
pc = Pinecone(
    api_key=os.getenv("PINECONE_API_KEY")
)

index_name = os.getenv("PINECONE_INDEX_NAME")

# Groq init
client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
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
    return {
        "message": "Travel Destination Search RAG System", 
        "version": "1.0.0",
        "endpoints": {
            "search": "/v1/search-destinations",
            "ingest": "/v1/ingest-destinations", 
            "chat": "/v1/chat/completions",
            "health": "/v1/keep-alive"
        }
    }

@app.head("/v1/keep-alive")
def health_check():
        return {"status": "healthy"}

@app.get("/v1/cities")
async def get_cities():
    """
    Lấy danh sách tất cả cities để client có thể chọn cityId
    """
    try:
        cities = list(cities_collection.find({}, {
            "_id": 1,
            "name": 1,
            "slug": 1,
            "description": 1
        }))
        
        # Convert ObjectId to string
        for city in cities:
            city["_id"] = str(city["_id"])
        
        return {
            "cities": cities,
            "total": len(cities)
        }
        
    except Exception as e:
        print(f"Error in get_cities: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/ingest-destinations")
async def ingest_destinations():
    """
    Ingest tất cả destinations từ MongoDB vào Pinecone vector database
    """
    try:
        # Lấy tất cả destinations từ MongoDB
        destinations = list(destinations_collection.find({}))
        
        records = []
        
        for dest in destinations:
            # Tạo text content cho embedding
            content_parts = [
                dest['title'],
                dest['details']['description'],
                " ".join(dest['details']['highlight']),
                " ".join(dest['details']['services']),
                " ".join(dest['details']['activities']),
                dest['location']['address']
            ]
            
            # Lấy tên tags
            tag_names = []
            if 'tags' in dest and dest['tags']:
                for tag_id in dest['tags']:
                    tag_doc = db.tags.find_one({"_id": tag_id})
                    if tag_doc:
                        tag_names.append(tag_doc.get('name', ''))
            
            if tag_names:
                content_parts.append(" ".join(tag_names))
            
            # Tạo content text
            content_text = " ".join([part for part in content_parts if part])
            
            # Clean text
            def clean_text(text: str) -> str:
                text = re.sub(r'\s+', ' ', text).strip()
                return text
            
            cleaned_content = clean_text(content_text)
            
            # Tạo record cho Pinecone
            record = {
                "id": f"dest-{dest['_id']}",
                "text": cleaned_content,
                "destinationId": str(dest['_id']),
                "title": dest['title'],
                "cityId": str(dest['location']['city']),
                "tags": tag_names
            }
            
            records.append(record)
        
        # Batch size của Pinecone
        batch_size = 90
        
        # Upsert vào Pinecone theo batch
        for i in range(0, len(records), batch_size):
            batch = records[i:i + batch_size]
            index.upsert_records("destinations", batch)
        
        return {
            "status": "success", 
            "destinations_processed": len(records),
            "message": f"Đã ingest {len(records)} destinations vào Pinecone"
        }
        
    except Exception as e:
        print(f"Error in ingest_destinations: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/search-destinations")
async def search_destinations(payload: DestinationSearchPayload):
    """
    Tìm kiếm địa điểm du lịch dựa trên cityId và purpose
    """
    try:
        # 1. Tìm city theo ID
        city = cities_collection.find_one({"_id": ObjectId(payload.cityId)})
        if not city:
            raise HTTPException(status_code=404, detail=f"Không tìm thấy thành phố với ID: {payload.cityId}")
        
        # 2. Gọi LLM để phân tích purpose thành tags
        tag_analysis_prompt = f"""
        ### 📍 Yêu cầu phân tích:
        Phân tích mục đích du lịch sau đây và trả về danh sách các tags phù hợp cho việc tìm kiếm địa điểm du lịch.
        
        **Mục đích:** {payload.purpose}
        **Thành phố:** {city['name']}
        
        ### 🏷️ Tags cần trả về:
        - Trả về tối đa 5 tags phù hợp nhất
        - Mỗi tag phải ngắn gọn, rõ ràng
        - Tags phải liên quan đến loại địa điểm, hoạt động, hoặc đặc điểm du lịch
        - Ví dụ: "chụp ảnh" → ["photography", "scenic", "landmark", "viewpoint", "instagram"]
        
        ### 📋 Format trả về:
        Chỉ trả về danh sách tags, mỗi tag trên một dòng, không có đánh số:
        tag1
        tag2
        tag3
        """
        
        tag_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": tag_analysis_prompt}],
            model="deepseek-r1-distill-llama-70b",
            temperature=0.3,
            max_tokens=200
        )
        
        # Parse tags từ response
        generated_tags = []
        tag_response = tag_completion.choices[0].message.content.strip()
        for line in tag_response.split('\n'):
            tag = line.strip().lower()
            if tag and not tag.startswith('#'):
                generated_tags.append(tag)
        
        # 3. Tạo embedding cho câu truy vấn
        query_text = f"{payload.purpose} in {city['name']}"
        
        # Search trong Pinecone
        search_results = index.search(
            namespace="destinations",
            query={
                "top_k": 50,  # Lấy nhiều hơn để filter sau
                "inputs": {
                    'text': query_text
                }
            }
        )
        
        # 4. Lọc và xử lý kết quả
        filtered_destinations = []
        destination_ids = []
        
        for hit in search_results['result']['hits']:
            destination_id = hit['fields']['destinationId']
            destination_ids.append(destination_id)
        
        # Lấy thông tin chi tiết từ MongoDB
        destinations = list(destinations_collection.find({
            "_id": {"$in": [ObjectId(did) for did in destination_ids]},
            "location.city": city['_id']
        }))
        
        # Tạo map để truy cập nhanh
        destination_map = {str(dest['_id']): dest for dest in destinations}
        score_map = {hit['fields']['destinationId']: hit['_score'] for hit in search_results['result']['hits']}
        
        # Lấy thông tin tags cho mỗi destination
        for dest in destinations:
            dest_id = str(dest['_id'])
            if dest_id in score_map:
                # Tính điểm dựa trên similarity score và tag matching
                base_score = score_map[dest_id]
                
                # Tính điểm bonus cho tag matching
                tag_bonus = 0
                if 'tags' in dest and dest['tags']:
                    # Lấy tên tags từ ObjectId
                    tag_names = []
                    for tag_id in dest['tags']:
                        tag_doc = db.tags.find_one({"_id": tag_id})
                        if tag_doc:
                            tag_names.append(tag_doc.get('name', '').lower())
                    
                    # Tính số tag trùng khớp
                    matching_tags = set(tag_names) & set(generated_tags)
                    tag_bonus = len(matching_tags) * 0.1  # Bonus 0.1 cho mỗi tag trùng
                
                final_score = base_score + tag_bonus
                
                # Tạo response object
                destination_result = DestinationResult(
                    title=dest['title'],
                    slug=dest['slug'],
                    tags=tag_names if 'tag_names' in locals() else [],
                    location={
                        "address": dest['location']['address'],
                        "city": city['name']
                    },
                    details={
                        "description": dest['details']['description'],
                        "highlight": dest['details']['highlight'],
                        "services": dest['details']['services'],
                        "activities": dest['details']['activities'],
                        "fee": dest['details']['fee']
                    },
                    album=dest['album'],
                    score=round(final_score, 3)
                )
                
                filtered_destinations.append(destination_result)
        
        # Sắp xếp theo điểm số giảm dần
        filtered_destinations.sort(key=lambda x: x.score, reverse=True)
        
        # Giới hạn số lượng kết quả
        limited_destinations = filtered_destinations[:payload.limit]
        
        return DestinationSearchResponse(
            city={
                "id": str(city['_id']),
                "name": city['name'],
                "slug": city['slug'],
                "description": city.get('description', '')
            },
            purpose=payload.purpose,
            generatedTags=generated_tags,
            destinations=limited_destinations,
            totalFound=len(limited_destinations)
        )
        
    except Exception as e:
        print(f"Error in search_destinations: {e}")
        raise HTTPException(status_code=500, detail=str(e))

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
        if payload.courseId:
            query["filter"] = {"courseId": payload.courseId}

        results = index.search(
            namespace=payload.userId,
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
                            f"**Đoạn văn {i+1} (Document title: {hit['fields']['title']}):**\n"
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

        response_dict["choices"][len(response_dict["choices"])-1]["message"]["documents"] = [
            {
                "id": hit["_id"],
                "text": hit["fields"]["text"],
                "documentId": hit["fields"]["documentId"],
                "score": hit["_score"]
            } for hit in results['result']['hits']
        ]
        return response_dict
    
