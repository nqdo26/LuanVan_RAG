#!/usr/bin/env python3
"""
Test script cho RAG Server endpoints
Kiểm tra toàn bộ chức năng CRUD với Pinecone synchronization
"""

import requests
import json
import time
from datetime import datetime

# Configuration
RAG_SERVER_URL = "http://localhost:8000"
TEST_CITY_ID = "test_city_123"
TEST_DESTINATION_ID = "test_dest_456"

def print_test_header(test_name):
    print(f"\n{'='*60}")
    print(f"🧪 TEST: {test_name}")
    print(f"{'='*60}")

def print_result(success, message, data=None):
    status = "✅ PASS" if success else "❌ FAIL"
    print(f"{status}: {message}")
    if data:
        print(f"📊 Data: {json.dumps(data, indent=2, ensure_ascii=False)}")
    print("-" * 40)

def test_health_check():
    """Test basic health endpoints"""
    print_test_header("Health Check")
    
    try:
        # Test GET /
        response = requests.get(f"{RAG_SERVER_URL}/")
        if response.status_code == 200:
            print_result(True, "GET / endpoint working", response.json())
        else:
            print_result(False, f"GET / failed with status {response.status_code}")
        
        # Test HEAD /v1/keep-alive
        response = requests.head(f"{RAG_SERVER_URL}/v1/keep-alive")
        if response.status_code == 200:
            print_result(True, "HEAD /v1/keep-alive endpoint working")
        else:
            print_result(False, f"HEAD /v1/keep-alive failed with status {response.status_code}")
            
    except Exception as e:
        print_result(False, f"Health check failed: {str(e)}")

def test_ingest_destination():
    """Test destination ingestion with semantic chunking"""
    print_test_header("Destination Ingestion")
    
    # Sample destination data với đầy đủ thông tin
    structured_data = {
        "description": "Chùa Cầu Hội An là một trong những biểu tượng nổi tiếng nhất của phố cổ Hội An. Được xây dựng vào thế kỷ 16 bởi cộng đồng người Nhật, chùa có kiến trúc độc đáo kết hợp giữa Nhật Bản và Việt Nam.",
        "highlight": "Kiến trúc cổ kính, cầu có mái che, tượng khỉ và chó canh gác",
        "services": "Tham quan, chụp ảnh, mua vé vào cửa",
        "cultureType": "Di tích lịch sử, kiến trúc cổ",
        "activities": "Đi bộ qua cầu, chụp ảnh, tìm hiểu lịch sử",
        "fee": "Vé vào cửa phố cổ Hội An: 120,000 VNĐ",
        "usefulInfo": "Nên đi vào buổi sáng sớm hoặc chiều tối để tránh đông đúc, mặc trang phục lịch sự",
        "tags": "di tích lịch sử, phố cổ Hội An, chùa cầu, kiến trúc Nhật Bản",
        "openHour": "thứ hai: 07:00-18:00, thứ ba: 07:00-18:00, thứ tư: 07:00-18:00, thứ năm: 07:00-18:00, thứ sáu: 07:00-18:00, thứ bảy: 07:00-18:00, chủ nhật: 07:00-18:00",
        "contactInfo": "Phone: +84 235 3861 540, Website: hoianworldheritage.org.vn, Facebook: , Instagram: "
    }
    
    payload = {
        "destinationId": TEST_DESTINATION_ID,
        "cityId": TEST_CITY_ID,
        "info": json.dumps(structured_data, ensure_ascii=False),
        "slug": "chua-cau-hoi-an",
        "name": "Chùa Cầu Hội An"
    }
    
    try:
        response = requests.post(f"{RAG_SERVER_URL}/v1/ingest", json=payload)
        if response.status_code == 200:
            result = response.json()
            print_result(True, f"Ingestion successful, processed {result.get('chunks_processed', 0)} chunks", result)
        else:
            print_result(False, f"Ingestion failed with status {response.status_code}", response.text)
            
    except Exception as e:
        print_result(False, f"Ingestion failed: {str(e)}")

def test_search_question():
    """Test semantic search với câu hỏi tiếng Việt"""
    print_test_header("Question Answering")
    
    questions = [
        "Chùa Cầu Hội An có gì đặc biệt?",
        "Giờ mở cửa của Chùa Cầu như thế nào?",
        "Phí tham quan Chùa Cầu bao nhiêu?",
        "Nên đi tham quan Chùa Cầu vào thời gian nào?"
    ]
    
    for question in questions:
        payload = {
            "cityId": TEST_CITY_ID,
            "query": question,
            "model": "deepseek-r1-distill-llama-70b"
        }
        
        try:
            response = requests.post(f"{RAG_SERVER_URL}/v1/question", json=payload)
            if response.status_code == 200:
                result = response.json()
                answer = result['choices'][0]['message']['content']
                documents = result['choices'][0]['message'].get('documents', [])
                print_result(True, f"Question: {question}")
                print(f"🤖 Answer: {answer[:200]}...")
                print(f"📚 Found {len(documents)} relevant documents")
            else:
                print_result(False, f"Question failed: {question}", response.text)
                
        except Exception as e:
            print_result(False, f"Question failed: {str(e)}")

def test_update_destination():
    """Test cập nhật destination"""
    print_test_header("Destination Update")
    
    # Updated data with new information
    updated_data = {
        "description": "Chùa Cầu Hội An (còn gọi là cầu Nhật Bản) là một trong những biểu tượng nổi tiếng nhất của phố cổ Hội An. Được xây dựng vào thế kỷ 16 bởi cộng đồng người Nhật, chùa có kiến trúc độc đáo kết hợp giữa Nhật Bản và Việt Nam. Đây là di tích được UNESCO công nhận.",
        "highlight": "Kiến trúc cổ kính, cầu có mái che, tượng khỉ và chó canh gác, di sản UNESCO",
        "services": "Tham quan, chụp ảnh, mua vé vào cửa, hướng dẫn viên",
        "cultureType": "Di tích lịch sử, kiến trúc cổ, di sản thế giới",
        "activities": "Đi bộ qua cầu, chụp ảnh, tìm hiểu lịch sử, ngắm cảnh sông Thu Bồn",
        "fee": "Vé vào cửa phố cổ Hội An: 120,000 VNĐ (có thể thay đổi)",
        "usefulInfo": "Nên đi vào buổi sáng sớm hoặc chiều tối để tránh đông đúc, mặc trang phục lịch sự, không được chạy xe máy qua cầu",
        "tags": "di tích lịch sử, phố cổ Hội An, chùa cầu, kiến trúc Nhật Bản, UNESCO",
        "openHour": "Mở cửa 24/7 cho khách đi bộ, tham quan chi tiết: 07:00-18:00",
        "contactInfo": "Phone: +84 235 3861 540, Website: hoianworldheritage.org.vn, Facebook: hoianheritage, Instagram: hoi_an_ancient_town"
    }
    
    payload = {
        "destinationId": TEST_DESTINATION_ID,
        "cityId": TEST_CITY_ID,
        "info": json.dumps(updated_data, ensure_ascii=False),
        "slug": "chua-cau-hoi-an",
        "name": "Chùa Cầu Hội An (Cầu Nhật Bản)"
    }
    
    try:
        response = requests.post(f"{RAG_SERVER_URL}/v1/update", json=payload)
        if response.status_code == 200:
            result = response.json()
            print_result(True, f"Update successful", result)
        else:
            print_result(False, f"Update failed with status {response.status_code}", response.text)
            
    except Exception as e:
        print_result(False, f"Update failed: {str(e)}")

def test_chat_completion():
    """Test chat completion với knowledge base"""
    print_test_header("Chat Completion")
    
    messages = [
        {"role": "user", "content": "Tôi muốn tìm hiểu về Chùa Cầu Hội An. Bạn có thể cho tôi biết thông tin chi tiết không?"}
    ]
    
    payload = {
        "messages": messages,
        "model": "deepseek-r1-distill-llama-70b",
        "isUseKnowledge": True,
        "cityId": TEST_CITY_ID,
        "userId": "test_user_123"
    }
    
    try:
        response = requests.post(f"{RAG_SERVER_URL}/v1/chat/completions", json=payload)
        if response.status_code == 200:
            result = response.json()
            answer = result['choices'][0]['message']['content']
            destinations = result['choices'][0]['message'].get('destinations', [])
            print_result(True, "Chat completion successful")
            print(f"🤖 Assistant: {answer[:300]}...")
            print(f"📍 Found {len(destinations)} related destinations")
        else:
            print_result(False, f"Chat completion failed with status {response.status_code}", response.text)
            
    except Exception as e:
        print_result(False, f"Chat completion failed: {str(e)}")

def test_delete_destination():
    """Test xóa destination"""
    print_test_header("Destination Deletion")
    
    payload = {
        "destiationId": TEST_DESTINATION_ID,  # Note: API có typo "destiation"
        "cityId": TEST_CITY_ID
    }
    
    try:
        response = requests.post(f"{RAG_SERVER_URL}/v1/delete-document", json=payload)
        if response.status_code == 200:
            result = response.json()
            print_result(True, f"Deletion successful", result)
        else:
            print_result(False, f"Deletion failed with status {response.status_code}", response.text)
            
    except Exception as e:
        print_result(False, f"Deletion failed: {str(e)}")

def run_all_tests():
    """Chạy tất cả các test theo thứ tự"""
    print(f"🚀 Starting RAG Server Tests at {datetime.now()}")
    print(f"🌐 Server URL: {RAG_SERVER_URL}")
    
    # 1. Health check
    test_health_check()
    time.sleep(1)
    
    # 2. Test ingestion
    test_ingest_destination()
    time.sleep(2)  # Wait for indexing
    
    # 3. Test search
    test_search_question()
    time.sleep(1)
    
    # 4. Test update
    test_update_destination()
    time.sleep(2)  # Wait for re-indexing
    
    # 5. Test chat completion
    test_chat_completion()
    time.sleep(1)
    
    # 6. Test deletion (cleanup)
    test_delete_destination()
    
    print(f"\n🎉 All tests completed at {datetime.now()}")
    print("=" * 60)

if __name__ == "__main__":
    run_all_tests()
