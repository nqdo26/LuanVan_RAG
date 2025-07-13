import requests
import json

BASE_URL = "http://localhost:8000"

def quick_test():
    print("🚀 Quick Test - Travel RAG System")
    print("=" * 40)
    
    # Test 1: Basic endpoint
    try:
        response = requests.get(f"{BASE_URL}/", timeout=5)
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Server running: {result['message']}")
        else:
            print(f"❌ Server error: {response.status_code}")
    except Exception as e:
        print(f"❌ Connection error: {e}")
        return
    
    # Test 2: Chat without knowledge (should work)
    print("\n💬 Testing chat without knowledge...")
    payload = {
        "messages": [
            {"role": "user", "content": "Xin chào! Bạn có thể giúp tôi không?"}
        ],
        "userId": "test_user",
        "isUseKnowledge": False
    }
    
    try:
        response = requests.post(f"{BASE_URL}/v1/chat/completions", json=payload, timeout=30)
        if response.status_code == 200:
            result = response.json()
            content = result['choices'][0]['message']['content']
            print(f"✅ Chat successful!")
            print(f"   Response: {content[:100]}...")
        else:
            print(f"❌ Chat failed: {response.status_code}")
            print(f"   Error: {response.text}")
    except Exception as e:
        print(f"❌ Chat error: {e}")
    
    # Test 3: Search destinations (sử dụng cityId thực tế)
    print("\n🔍 Testing search destinations...")
    payload = {
        "cityId": "687098188c1b001339b73aa8",  # Hồ Chí Minh - ID thực tế
        "purpose": "chụp ảnh",
        "limit": 5
    }
    
    try:
        response = requests.post(f"{BASE_URL}/v1/search-destinations", json=payload, timeout=30)
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Search successful!")
            print(f"   City: {result['city']['name']}")
            print(f"   Found: {result['totalFound']} destinations")
        elif response.status_code == 404:
            print("⚠️  Search failed: City not found (expected without data)")
        elif response.status_code == 500:
            print("⚠️  Search failed: Server error (check logs)")
            print(f"   Error: {response.text}")
        else:
            print(f"❌ Search failed: {response.status_code}")
            print(f"   Error: {response.text}")
    except Exception as e:
        print(f"❌ Search error: {e}")
    
    print("\n" + "=" * 40)
    print("✨ Quick test completed!")

if __name__ == "__main__":
    quick_test() 