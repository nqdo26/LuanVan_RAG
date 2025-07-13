import requests
import json

# Cấu hình
BASE_URL = "http://localhost:8000"

def test_search_destinations():
    """Test API tìm kiếm địa điểm"""
    
    # Test case 1: Tìm địa điểm chụp ảnh ở Hà Nội
    payload = {
        "citySlug": "ha-noi",
        "purpose": "chụp ảnh",
        "limit": 5
    }
    
    print("🔍 Testing search destinations...")
    print(f"Request: {json.dumps(payload, indent=2, ensure_ascii=False)}")
    
    try:
        response = requests.post(f"{BASE_URL}/v1/search-destinations", json=payload)
        
        if response.status_code == 200:
            result = response.json()
            print("\n✅ Response:")
            print(f"City: {result['city']['name']}")
            print(f"Purpose: {result['purpose']}")
            print(f"Generated Tags: {result['generatedTags']}")
            print(f"Total Found: {result['totalFound']}")
            
            print("\n🏛️ Destinations:")
            for i, dest in enumerate(result['destinations'], 1):
                print(f"\n{i}. {dest['title']} (Score: {dest['score']})")
                print(f"   Tags: {dest['tags']}")
                print(f"   Address: {dest['location']['address']}")
                print(f"   Description: {dest['details']['description'][:100]}...")
        else:
            print(f"❌ Error: {response.status_code}")
            print(response.text)
            
    except Exception as e:
        print(f"❌ Exception: {e}")

def test_ingest_destinations():
    """Test API ingest destinations"""
    
    print("\n📥 Testing ingest destinations...")
    
    try:
        response = requests.post(f"{BASE_URL}/v1/ingest-destinations")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Success: {result['message']}")
            print(f"   Processed: {result['destinations_processed']} destinations")
        else:
            print(f"❌ Error: {response.status_code}")
            print(response.text)
            
    except Exception as e:
        print(f"❌ Exception: {e}")

def test_chat_completion():
    """Test API chat completion"""
    
    payload = {
        "messages": [
            {"role": "user", "content": "Tôi muốn tìm địa điểm chụp ảnh đẹp ở Hà Nội, bạn có gợi ý gì không?"}
        ],
        "userId": "test_user",
        "isUseKnowledge": True
    }
    
    print("\n💬 Testing chat completion...")
    print(f"Request: {json.dumps(payload, indent=2, ensure_ascii=False)}")
    
    try:
        response = requests.post(f"{BASE_URL}/v1/chat/completions", json=payload)
        
        if response.status_code == 200:
            result = response.json()
            print("\n✅ Response:")
            print(f"Content: {result['choices'][0]['message']['content'][:200]}...")
            
            if 'documents' in result['choices'][0]['message']:
                docs = result['choices'][0]['message']['documents']
                print(f"Documents used: {len(docs)}")
        else:
            print(f"❌ Error: {response.status_code}")
            print(response.text)
            
    except Exception as e:
        print(f"❌ Exception: {e}")

def test_health_check():
    """Test health check endpoint"""
    
    print("🏥 Testing health check...")
    
    try:
        response = requests.head(f"{BASE_URL}/v1/keep-alive")
        
        if response.status_code == 200:
            print("✅ Server is healthy!")
        else:
            print(f"❌ Server error: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Exception: {e}")

if __name__ == "__main__":
    print("🚀 Travel Destination Search API Test")
    print("=" * 50)
    
    # Test health check first
    test_health_check()
    
    # Test ingest (run this first to populate Pinecone)
    test_ingest_destinations()
    
    # Test search
    test_search_destinations()
    
    # Test chat
    test_chat_completion()
    
    print("\n✨ Test completed!") 