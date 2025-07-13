import requests
import json

# Cấu hình
BASE_URL = "http://localhost:8000"

def test_basic_endpoints():
    """Test các endpoint cơ bản"""
    
    print("🚀 Testing Travel Destination Search RAG System")
    print("=" * 60)
    
    # Test 1: Health check
    print("\n1️⃣ Testing health check...")
    try:
        response = requests.head(f"{BASE_URL}/v1/keep-alive", timeout=5)
        if response.status_code == 200:
            print("✅ Health check passed!")
        else:
            print(f"❌ Health check failed: {response.status_code}")
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to server. Is it running?")
        print("💡 Start server with: uvicorn server:app --reload --host 0.0.0.0 --port 8000")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    
    # Test 2: API info
    print("\n2️⃣ Testing API info...")
    try:
        response = requests.get(f"{BASE_URL}/", timeout=5)
        if response.status_code == 200:
            result = response.json()
            print(f"✅ API info: {result['message']}")
            print(f"   Version: {result['version']}")
            print(f"   Available endpoints: {list(result['endpoints'].keys())}")
        else:
            print(f"❌ API info failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Test 3: Search destinations (sẽ fail nếu chưa có data)
    print("\n3️⃣ Testing search destinations...")
    payload = {
        "citySlug": "ha-noi",
        "purpose": "chụp ảnh",
        "limit": 5
    }
    
    try:
        response = requests.post(f"{BASE_URL}/v1/search-destinations", json=payload, timeout=10)
        if response.status_code == 200:
            result = response.json()
            print("✅ Search destinations successful!")
            print(f"   City: {result['city']['name']}")
            print(f"   Purpose: {result['purpose']}")
            print(f"   Generated Tags: {result['generatedTags']}")
            print(f"   Total Found: {result['totalFound']}")
        elif response.status_code == 404:
            print("⚠️  Search failed: City not found (expected if no data)")
        elif response.status_code == 500:
            print("⚠️  Search failed: Server error (check API keys and database)")
        else:
            print(f"❌ Search failed: {response.status_code}")
            print(f"   Response: {response.text}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Test 4: Chat completion
    print("\n4️⃣ Testing chat completion...")
    payload = {
        "messages": [
            {"role": "user", "content": "Xin chào! Bạn có thể giúp tôi tìm địa điểm du lịch không?"}
        ],
        "userId": "test_user",
        "isUseKnowledge": False
    }
    
    try:
        response = requests.post(f"{BASE_URL}/v1/chat/completions", json=payload, timeout=15)
        if response.status_code == 200:
            result = response.json()
            print("✅ Chat completion successful!")
            content = result['choices'][0]['message']['content']
            print(f"   Response: {content[:100]}...")
        else:
            print(f"❌ Chat failed: {response.status_code}")
            print(f"   Response: {response.text}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print("\n" + "=" * 60)
    print("✨ Basic testing completed!")
    print("\n📝 Next steps:")
    print("1. Configure .env file with your API keys")
    print("2. Start MongoDB and add some test data")
    print("3. Run: uvicorn server:app --reload --host 0.0.0.0 --port 8000")
    print("4. Test again with: python simple_test.py")

if __name__ == "__main__":
    test_basic_endpoints() 