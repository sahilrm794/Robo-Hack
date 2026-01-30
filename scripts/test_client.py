"""
Test Client for Voice Agent API.
Simple script to test the API endpoints.
"""

import asyncio
import httpx
import json


BASE_URL = "http://localhost:8000"


async def test_health():
    """Test health endpoints."""
    print("\n🏥 Testing Health Endpoints...")
    
    async with httpx.AsyncClient() as client:
        # Basic health
        response = await client.get(f"{BASE_URL}/api/health")
        print(f"   /api/health: {response.status_code}")
        print(f"   {response.json()}")
        
        # Readiness
        response = await client.get(f"{BASE_URL}/api/health/ready")
        print(f"   /api/health/ready: {response.status_code}")
        print(f"   {response.json()}")


async def test_conversation():
    """Test conversation endpoint."""
    print("\n💬 Testing Conversation Endpoint...")
    
    test_messages = [
        {"text": "Hello, I need help with my order", "language": "en"},
        {"text": "What is your return policy?", "language": "en"},
        {"text": "मुझे मेरा ऑर्डर ट्रैक करना है", "language": "hi"},
        {"text": "Do you have any wireless headphones?", "language": "en"},
    ]
    
    session_id = None
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        for msg in test_messages:
            payload = {**msg}
            if session_id:
                payload["session_id"] = session_id
            
            print(f"\n   📤 User: {msg['text']}")
            
            response = await client.post(
                f"{BASE_URL}/api/conversation/message",
                json=payload
            )
            
            if response.status_code == 200:
                data = response.json()
                session_id = data.get("session_id")
                print(f"   🤖 Agent: {data['response'][:100]}...")
                print(f"   ⏱️  Latency: {data['latency_ms']}ms")
                
                if data.get("tool_calls"):
                    for tc in data["tool_calls"]:
                        print(f"   🔧 Tool: {tc['tool']}")
            else:
                print(f"   ❌ Error: {response.status_code}")
                print(f"   {response.text}")


async def test_order_tracking():
    """Test order tracking scenario."""
    print("\n📦 Testing Order Tracking Scenario...")
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        # Ask about order
        response = await client.post(
            f"{BASE_URL}/api/conversation/message",
            json={
                "text": "I want to track my order. My phone number is 9876543210",
                "language": "en"
            }
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"   🤖 Agent: {data['response']}")
            print(f"   ⏱️  Latency: {data['latency_ms']}ms")
            
            if data.get("tool_calls"):
                for tc in data["tool_calls"]:
                    print(f"   🔧 Tool Used: {tc['tool']}")
                    print(f"      Input: {tc['input']}")


async def test_product_search():
    """Test product search scenario."""
    print("\n🔍 Testing Product Search Scenario...")
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        # Search for products
        response = await client.post(
            f"{BASE_URL}/api/conversation/message",
            json={
                "text": "Show me some laptops under 1 lakh rupees",
                "language": "en"
            }
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"   🤖 Agent: {data['response'][:200]}...")
            print(f"   ⏱️  Latency: {data['latency_ms']}ms")


async def test_faq():
    """Test FAQ lookup scenario."""
    print("\n❓ Testing FAQ Scenario...")
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            f"{BASE_URL}/api/conversation/message",
            json={
                "text": "What payment methods do you accept?",
                "language": "en"
            }
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"   🤖 Agent: {data['response'][:200]}...")
            print(f"   ⏱️  Latency: {data['latency_ms']}ms")


async def test_multilingual():
    """Test multilingual support."""
    print("\n🌍 Testing Multilingual Support...")
    
    messages = [
        {"text": "मुझे अपना ऑर्डर रद्द करना है", "language": "hi"},
        {"text": "আমার অর্ডার কোথায়?", "language": "bn"},
        {"text": "माझा ऑर्डर कुठे आहे?", "language": "mr"},
    ]
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        for msg in messages:
            print(f"\n   📤 [{msg['language']}] {msg['text']}")
            
            response = await client.post(
                f"{BASE_URL}/api/conversation/message",
                json=msg
            )
            
            if response.status_code == 200:
                data = response.json()
                print(f"   🤖 Agent: {data['response'][:100]}...")


async def main():
    """Run all tests."""
    print("=" * 60)
    print("🧪 Voice Agent API Test Client")
    print("=" * 60)
    print(f"Target: {BASE_URL}")
    
    try:
        await test_health()
        await test_conversation()
        await test_order_tracking()
        await test_product_search()
        await test_faq()
        await test_multilingual()
        
        print("\n" + "=" * 60)
        print("✅ All tests completed!")
        print("=" * 60)
        
    except httpx.ConnectError:
        print("\n❌ Cannot connect to server. Make sure it's running:")
        print("   uvicorn app.main:app --reload")
    except Exception as e:
        print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
