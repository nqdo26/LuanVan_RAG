#!/usr/bin/env python3
"""
Quick test script để kiểm tra RAG server nhanh chóng
"""

import requests
import json

RAG_URL = "http://localhost:8000"

def test_health():
    """Kiểm tra server có chạy không"""
    print("🔍 Checking server health...")
    try:
        response = requests.get(f"{RAG_URL}/", timeout=5)
        if response.status_code == 200:
            print("✅ Server is running!")
            print(f"Response: {response.json()}")
        else:
            print(f"❌ Server responded with status {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"❌ Server not accessible: {e}")

def test_simple_ingest():
    """Test đơn giản để ingest 1 destination"""
    print("\n📤 Testing simple destination ingest...")
    
    data = {
        "description": "Đây là một địa điểm du lịch tuyệt vời",
        "highlight": "Cảnh đẹp, không khí trong lành",
        "services": "Nhà hàng, cafe, wifi miễn phí",
        "tags": "du lịch, thiên nhiên, nghỉ dưỡng"
    }
    
    payload = {
        "destinationId": "test_simple_123",
        "cityId": "test_city_simple",
        "info": json.dumps(data, ensure_ascii=False),
        "slug": "test-simple-destination",
        "name": "Điểm Du Lịch Test"
    }
    
    try:
        response = requests.post(f"{RAG_URL}/v1/ingest", json=payload, timeout=10)
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Ingest successful! Processed {result.get('chunks_processed', 0)} chunks")
        else:
            print(f"❌ Ingest failed: {response.status_code}")
            print(f"Error: {response.text}")
    except Exception as e:
        print(f"❌ Ingest error: {e}")

def test_simple_question():
    """Test câu hỏi đơn giản"""
    print("\n❓ Testing simple question...")
    
    payload = {
        "cityId": "test_city_simple",
        "query": "Địa điểm này có gì hay?"
    }
    
    try:
        response = requests.post(f"{RAG_URL}/v1/question", json=payload, timeout=10)
        if response.status_code == 200:
            result = response.json()
            answer = result['choices'][0]['message']['content']
            print(f"✅ Question answered!")
            print(f"🤖 Answer: {answer[:100]}...")
        else:
            print(f"❌ Question failed: {response.status_code}")
            print(f"Error: {response.text}")
    except Exception as e:
        print(f"❌ Question error: {e}")

if __name__ == "__main__":
    print("🧪 Quick RAG Server Test")
    print("=" * 40)
    
    test_health()
    test_simple_ingest()
    test_simple_question()
    
    print("\n✨ Quick test completed!")
