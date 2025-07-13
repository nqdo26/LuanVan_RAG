# Travel Destination Search RAG System

Hệ thống tìm kiếm địa điểm du lịch thông minh sử dụng RAG (Retrieval-Augmented Generation) với Pinecone vector database và Groq LLM.

## 🚀 Tính năng chính

### 1. Tìm kiếm địa điểm thông minh
- **Input**: `citySlug` + `purpose` (ví dụ: "chụp ảnh")
- **Process**: 
  - Tìm city theo slug trong MongoDB
  - Phân tích purpose thành tags bằng LLM
  - Tìm kiếm semantic trong Pinecone
  - Lọc kết quả theo city và ưu tiên tag matching
- **Output**: Danh sách địa điểm phù hợp nhất

### 2. Ingest dữ liệu
- Tự động chuyển đổi destinations từ MongoDB sang Pinecone
- Tạo embeddings cho title, description, tags, services, activities
- Hỗ trợ batch processing

### 3. Chat với context
- Chat conversation với knowledge base
- Context-aware responses

## 🛠️ Cài đặt

### 1. Cài đặt dependencies
```bash
pip install -r requirements.txt
```

### 2. Cấu hình environment variables
Tạo file `.env`:
```env
# Pinecone
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=travel-destinations

# Groq
GROQ_API_KEY=your_groq_api_key

# MongoDB
MONGODB_URI=mongodb://localhost:27017
MONGODB_DB=travel_db
```

### 3. Khởi chạy server
```bash
uvicorn server:app --reload --host 0.0.0.0 --port 8000
```

## 📡 API Endpoints

### 1. Tìm kiếm địa điểm
```http
POST /v1/search-destinations
Content-Type: application/json

{
  "citySlug": "ho-chi-minh",
  "purpose": "chụp ảnh",
  "limit": 10
}
```

**Response:**
```json
{
  "city": {
    "name": "Hồ Chí Minh",
    "slug": "ho-chi-minh",
    "description": "..."
  },
  "purpose": "chụp ảnh",
  "generatedTags": ["photography", "scenic", "landmark", "viewpoint"],
  "destinations": [
    {
      "title": "Landmark 81",
      "slug": "landmark-81",
      "tags": ["landmark", "viewpoint", "photography"],
      "location": {
        "address": "Vinhomes Central Park",
        "city": "Hồ Chí Minh"
      },
      "details": {
        "description": "...",
        "highlight": ["..."],
        "services": ["..."],
        "activities": ["..."],
        "fee": ["..."]
      },
      "album": {
        "highlight": ["..."],
        "space": ["..."],
        "fnb": ["..."],
        "extra": ["..."]
      },
      "score": 0.95
    }
  ],
  "totalFound": 10
}
```

### 2. Ingest destinations
```http
POST /v1/ingest-destinations
```

### 3. Chat completion
```http
POST /v1/chat/completions
Content-Type: application/json

{
  "messages": [
    {"role": "user", "content": "Tôi muốn tìm địa điểm chụp ảnh ở Hà Nội"}
  ],
  "userId": "test_user",
  "isUseKnowledge": true
}
```

### 4. Health check
```http
HEAD /v1/keep-alive
```

## 🔧 Cấu trúc dữ liệu

### MongoDB Collections

#### Cities
```javascript
{
  _id: ObjectId,
  name: String,
  slug: String,
  description: String,
  type: [ObjectId], // cityType references
  views: Number,
  images: [String],
  weather: [{
    title: String,
    minTemp: Number,
    maxTemp: Number,
    note: String
  }],
  info: [{
    title: String,
    description: String
  }]
}
```

#### Destinations
```javascript
{
  _id: ObjectId,
  title: String,
  slug: String,
  type: String,
  tags: [ObjectId], // tag references
  location: {
    address: String,
    city: ObjectId // city reference
  },
  album: {
    highlight: [String],
    space: [String],
    fnb: [String],
    extra: [String]
  },
  details: {
    description: String,
    highlight: [String],
    services: [String],
    activities: [String],
    fee: [String]
  }
}
```

### Pinecone Vector Structure
```javascript
{
  id: "dest-{destinationId}",
  text: "combined content for embedding",
  destinationId: "string",
  title: "destination title",
  cityId: "string",
  tags: ["tag1", "tag2"]
}
```

## 🎯 Workflow

1. **Ingest Phase**: Chuyển đổi destinations từ MongoDB sang Pinecone
2. **Search Phase**: 
   - User gửi citySlug + purpose
   - Tìm city trong MongoDB
   - LLM phân tích purpose thành tags
   - Search semantic trong Pinecone
   - Filter theo city và tag matching
   - Trả về kết quả được sắp xếp theo relevance

## 🔍 Tối ưu hóa

- **Tag Matching**: Bonus score cho destinations có tags trùng khớp
- **City Filtering**: Chỉ trả về destinations trong city được chọn
- **Semantic Search**: Tìm kiếm dựa trên nội dung, không chỉ keywords
- **Batch Processing**: Xử lý hiệu quả với large datasets

## 🚀 Deployment

### Docker
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Environment Variables
Đảm bảo cấu hình đầy đủ các API keys và database connections trước khi deploy.
