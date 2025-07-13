import requests
import json

BASE_URL = "http://localhost:8000"

def detailed_test():
    print("🔍 Detailed Test - Travel Destination Search")
    print("=" * 50)
    
    # Test 1: Hồ Chí Minh với purpose khác
    print("1️⃣ Testing Hồ Chí Minh with 'du lịch' purpose")
    payload1 = {
        "cityId": "687098458c1b001339b73ab4",  # Hồ Chí Minh
        "purpose": "du lịch",
        "limit": 10
    }
    
    test_search(payload1)
    
    # Test 2: Cần Thơ
    print("\n2️⃣ Testing Cần Thơ with 'chụp ảnh' purpose")
    payload2 = {
        "cityId": "687098188c1b001339b73aa8",  # Cần Thơ
        "purpose": "chụp ảnh",
        "limit": 10
    }
    
    test_search(payload2)
    
    # Test 3: Hồ Chí Minh với purpose đơn giản
    print("\n3️⃣ Testing Hồ Chí Minh with 'ăn uống' purpose")
    payload3 = {
        "cityId": "687098458c1b001339b73ab4",  # Hồ Chí Minh
        "purpose": "ăn uống",
        "limit": 10
    }
    
    test_search(payload3)

def test_search(payload):
    print(f"\n🔍 Searching destinations in city for: {payload['purpose']}")
    print(f"   City ID: {payload['cityId']}")
    print(f"   Limit: {payload['limit']}")
    
    try:
        response = requests.post(f"{BASE_URL}/v1/search-destinations", json=payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            
            print(f"✅ Search successful!")
            print(f"📍 City: {result['city']['name']}")
            print(f"🎯 Purpose: {result['purpose']}")
            print(f"🏷️  Generated Tags: {result['generatedTags']}")
            print(f"📊 Total Found: {result['totalFound']}")
            
            if result['destinations']:
                print(f"🏛️  Destinations found:")
                print("-" * 40)
                
                for i, dest in enumerate(result['destinations'], 1):
                    print(f"\n{i}. {dest['title']}")
                    print(f"   📍 Address: {dest['location']['address']}")
                    print(f"   🏷️  Tags: {dest['tags']}")
                    print(f"   ⭐ Score: {dest['score']}")
                    print(f"   📝 Description: {dest['details']['description'][:80]}...")
                    
                    if dest['details']['highlight']:
                        print(f"   ✨ Highlights: {', '.join(dest['details']['highlight'][:2])}")
                    
                    print("-" * 20)
            else:
                print("⚠️  No destinations found")
                
        elif response.status_code == 404:
            print("❌ City not found")
        elif response.status_code == 500:
            print("❌ Server error")
            print(f"   Error: {response.text}")
        else:
            print(f"❌ Search failed: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    detailed_test()
    print("\n" + "=" * 50)
    print("✨ Detailed test completed!") 