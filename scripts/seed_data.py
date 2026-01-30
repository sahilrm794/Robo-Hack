"""
Seed Database with Sample Data.
Populates the database with demo products, FAQs, policies, and orders.
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime, timedelta
import uuid

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.db.database import init_db, get_db
from app.db.models import Product, Order, OrderItem, FAQ, Policy


def gen_id():
    """Generate a short unique ID."""
    return str(uuid.uuid4())[:8]


async def seed_products():
    """Seed sample products."""
    print("📦 Seeding products...")
    
    products = [
        {
            "id": "PHONE-001",
            "name": "SmartPhone Pro X",
            "name_hi": "स्मार्टफोन प्रो एक्स",
            "name_bn": "স্মার্টফোন প্রো এক্স",
            "name_mr": "स्मार्टफोन प्रो एक्स",
            "category": "Electronics",
            "price": 49999.00,
            "description": "Latest flagship smartphone with 5G, 128GB storage, and 48MP camera",
            "description_hi": "5G, 128GB स्टोरेज और 48MP कैमरा वाला नवीनतम फ्लैगशिप स्मार्टफोन",
            "specifications": {"storage": "128GB", "ram": "8GB", "camera": "48MP", "5g": True},
            "stock_quantity": 50
        },
        {
            "id": "LAPTOP-001",
            "name": "UltraBook 15 Pro",
            "name_hi": "अल्ट्राबुक 15 प्रो",
            "category": "Electronics",
            "price": 89999.00,
            "description": "Thin and light laptop with Intel i7, 16GB RAM, and 512GB SSD",
            "specifications": {"processor": "Intel i7", "ram": "16GB", "storage": "512GB SSD"},
            "stock_quantity": 25
        },
        {
            "id": "HEADPHONES-001",
            "name": "Wireless ANC Headphones",
            "name_hi": "वायरलेस एएनसी हेडफोन",
            "category": "Electronics",
            "price": 15999.00,
            "description": "Premium wireless headphones with Active Noise Cancellation",
            "specifications": {"anc": True, "battery": "30 hours", "bluetooth": "5.2"},
            "stock_quantity": 100
        },
        {
            "id": "SHIRT-001",
            "name": "Cotton Casual Shirt",
            "name_hi": "कॉटन कैजुअल शर्ट",
            "category": "Clothing",
            "price": 1299.00,
            "description": "Comfortable 100% cotton shirt for casual wear",
            "specifications": {"material": "100% Cotton", "sizes": ["S", "M", "L", "XL"]},
            "stock_quantity": 200
        },
        {
            "id": "JEANS-001",
            "name": "Slim Fit Denim Jeans",
            "name_hi": "स्लिम फिट डेनिम जींस",
            "category": "Clothing",
            "price": 1999.00,
            "description": "Classic slim fit jeans with stretch comfort",
            "specifications": {"material": "Denim", "fit": "Slim"},
            "stock_quantity": 150
        },
        {
            "id": "MIXER-001",
            "name": "Kitchen Mixer Grinder 750W",
            "name_hi": "किचन मिक्सर ग्राइंडर 750W",
            "category": "Home & Kitchen",
            "price": 3499.00,
            "description": "Powerful mixer grinder with 3 jars for all kitchen needs",
            "specifications": {"power": "750W", "jars": 3, "warranty": "2 years"},
            "stock_quantity": 75
        },
        {
            "id": "COOKER-001",
            "name": "Stainless Steel Pressure Cooker 5L",
            "name_hi": "स्टेनलेस स्टील प्रेशर कुकर 5L",
            "category": "Home & Kitchen",
            "price": 2199.00,
            "description": "Durable stainless steel pressure cooker with safety valve",
            "specifications": {"capacity": "5L", "material": "Stainless Steel"},
            "stock_quantity": 120
        },
        {
            "id": "CREAM-001",
            "name": "Moisturizing Face Cream",
            "name_hi": "मॉइस्चराइजिंग फेस क्रीम",
            "category": "Beauty",
            "price": 599.00,
            "description": "Hydrating face cream for all skin types",
            "specifications": {"size": "100ml", "skin_type": "All"},
            "stock_quantity": 300
        },
        {
            "id": "BOOK-001",
            "name": "Learn Python Programming",
            "name_hi": "पायथन प्रोग्रामिंग सीखें",
            "category": "Books",
            "price": 499.00,
            "description": "Comprehensive guide to Python programming for beginners",
            "specifications": {"pages": 450, "language": "English"},
            "stock_quantity": 80
        },
        {
            "id": "CRICKET-001",
            "name": "Cricket Bat - Kashmir Willow",
            "name_hi": "क्रिकेट बैट - कश्मीर विलो",
            "category": "Sports",
            "price": 2499.00,
            "description": "Professional grade Kashmir willow cricket bat",
            "specifications": {"wood": "Kashmir Willow", "weight": "1.2kg"},
            "stock_quantity": 45
        }
    ]
    
    async with get_db() as db:
        for product_data in products:
            product = Product(**product_data)
            db.add(product)
        await db.commit()
    
    print(f"   ✅ Added {len(products)} products")


async def seed_faqs():
    """Seed sample FAQs."""
    print("❓ Seeding FAQs...")
    
    faqs = [
        # General FAQs
        {
            "id": gen_id(),
            "question": "What are your delivery charges?",
            "question_hi": "आपकी डिलीवरी चार्जेस क्या हैं?",
            "answer": "Delivery is FREE for orders above ₹499. For orders below ₹499, a delivery charge of ₹49 applies.",
            "answer_hi": "₹499 से ऊपर के ऑर्डर पर डिलीवरी मुफ्त है। ₹499 से कम के ऑर्डर पर ₹49 की डिलीवरी चार्ज लागू होती है।",
            "topic": "shipping",
            "is_general": True
        },
        {
            "id": gen_id(),
            "question": "How long does delivery take?",
            "question_hi": "डिलीवरी में कितना समय लगता है?",
            "answer": "Standard delivery takes 3-5 business days. Express delivery (₹99 extra) delivers within 1-2 business days for metro cities.",
            "answer_hi": "स्टैंडर्ड डिलीवरी में 3-5 कार्यदिवस लगते हैं।",
            "topic": "shipping",
            "is_general": True
        },
        {
            "id": gen_id(),
            "question": "What is your return policy?",
            "question_hi": "आपकी रिटर्न पॉलिसी क्या है?",
            "answer": "We offer 7-day easy returns for most products. Electronics have a 7-day replacement policy. Some items like innerwear are non-returnable.",
            "answer_hi": "हम अधिकांश उत्पादों के लिए 7-दिन की आसान वापसी प्रदान करते हैं।",
            "topic": "returns",
            "is_general": True
        },
        {
            "id": gen_id(),
            "question": "What payment methods do you accept?",
            "question_hi": "आप कौन से भुगतान के तरीके स्वीकार करते हैं?",
            "answer": "We accept Credit/Debit cards, UPI, Net Banking, Wallets (Paytm, PhonePe), and Cash on Delivery (COD).",
            "answer_hi": "हम क्रेडिट/डेबिट कार्ड, UPI, नेट बैंकिंग, वॉलेट और कैश ऑन डिलीवरी स्वीकार करते हैं।",
            "topic": "payment",
            "is_general": True
        },
        {
            "id": gen_id(),
            "question": "How can I track my order?",
            "question_hi": "मैं अपना ऑर्डर कैसे ट्रैक कर सकता हूं?",
            "answer": "You can track your order by providing your order ID or registered phone number. I can help you track it right now!",
            "answer_hi": "आप अपना ऑर्डर आईडी या रजिस्टर्ड फोन नंबर देकर ट्रैक कर सकते हैं।",
            "topic": "orders",
            "is_general": True
        },
        {
            "id": gen_id(),
            "question": "How do I cancel my order?",
            "question_hi": "मैं अपना ऑर्डर कैसे रद्द करूं?",
            "answer": "You can cancel your order before it is shipped. Once shipped, you'll need to reject delivery or return after receiving.",
            "answer_hi": "आप शिपमेंट से पहले अपना ऑर्डर रद्द कर सकते हैं।",
            "topic": "orders",
            "is_general": True
        },
        {
            "id": gen_id(),
            "question": "Do you have EMI options?",
            "question_hi": "क्या आपके पास EMI विकल्प हैं?",
            "answer": "Yes! We offer No-Cost EMI on orders above ₹3000 with select bank credit cards. Tenure options: 3, 6, 9, or 12 months.",
            "answer_hi": "हां! हम ₹3000 से ऊपर के ऑर्डर पर नो-कॉस्ट EMI प्रदान करते हैं।",
            "topic": "payment",
            "is_general": True
        },
        # Product-specific FAQs
        {
            "id": gen_id(),
            "product_id": "PHONE-001",
            "question": "Does this phone support 5G?",
            "answer": "Yes, the SmartPhone Pro X fully supports 5G connectivity on all major carriers.",
            "topic": "specifications",
            "is_general": False
        },
        {
            "id": gen_id(),
            "product_id": "LAPTOP-001",
            "question": "What is the battery life of this laptop?",
            "answer": "The UltraBook 15 Pro offers up to 10 hours of battery life under normal usage.",
            "topic": "specifications",
            "is_general": False
        },
        {
            "id": gen_id(),
            "product_id": "MIXER-001",
            "question": "What warranty does this mixer come with?",
            "answer": "The Kitchen Mixer Grinder comes with a 2-year manufacturer warranty covering motor and electrical defects.",
            "topic": "warranty",
            "is_general": False
        }
    ]
    
    async with get_db() as db:
        for faq_data in faqs:
            faq = FAQ(**faq_data)
            db.add(faq)
        await db.commit()
    
    print(f"   ✅ Added {len(faqs)} FAQs")


async def seed_policies():
    """Seed sample policies."""
    print("📜 Seeding policies...")
    
    policies = [
        {
            "id": gen_id(),
            "type": "return",
            "title": "Return and Refund Policy",
            "title_hi": "वापसी और रिफंड नीति",
            "content": """## Return Policy

### Eligibility
- Most products can be returned within 7 days of delivery
- Products must be unused, in original packaging with all tags attached
- Electronics have a 7-day replacement-only policy

### Refund Timeline
- Refund will be processed within 5-7 business days after return pickup
- Bank refunds may take additional 5-10 business days
- Store credit is instant""",
            "content_hi": "7 दिनों के भीतर अधिकांश उत्पाद वापस किए जा सकते हैं।"
        },
        {
            "id": gen_id(),
            "type": "shipping",
            "title": "Shipping Policy",
            "title_hi": "शिपिंग नीति",
            "content": """## Shipping Policy

### Delivery Charges
- FREE delivery on orders above ₹499
- ₹49 for orders below ₹499
- Express delivery: ₹99 (1-2 days for metro cities)

### Delivery Timeline
- Standard: 3-5 business days
- Express: 1-2 business days (metro cities)""",
            "content_hi": "₹499 से ऊपर के ऑर्डर पर मुफ्त डिलीवरी।"
        },
        {
            "id": gen_id(),
            "type": "payment",
            "title": "Payment Policy",
            "title_hi": "भुगतान नीति",
            "content": """## Payment Methods

- Credit/Debit Cards (Visa, Mastercard, RuPay)
- UPI (PhonePe, Google Pay, Paytm)
- Net Banking (all major banks)
- Cash on Delivery (COD) up to ₹25,000

### EMI Options
- No-Cost EMI on orders above ₹3000
- Available tenures: 3, 6, 9, 12 months""",
            "content_hi": "क्रेडिट/डेबिट कार्ड, UPI, नेट बैंकिंग, COD स्वीकार्य।"
        },
        {
            "id": gen_id(),
            "type": "warranty",
            "title": "Warranty Policy",
            "title_hi": "वारंटी नीति",
            "content": """## Warranty Policy

### Electronics
- 1-year standard manufacturer warranty
- Covers manufacturing defects only

### Home Appliances
- 2-year motor warranty
- 1-year comprehensive warranty""",
            "content_hi": "इलेक्ट्रॉनिक्स पर 1 साल की वारंटी।"
        },
        {
            "id": gen_id(),
            "type": "cancellation",
            "title": "Cancellation Policy",
            "title_hi": "रद्द करने की नीति",
            "content": """## Cancellation Policy

### Before Shipping
- Full refund on cancellation
- Instant cancellation available
- No cancellation charges

### After Shipping
- Cannot cancel once shipped
- Reject delivery for refund
- Or return after receiving""",
            "content_hi": "शिपिंग से पहले ऑर्डर रद्द किया जा सकता है।"
        }
    ]
    
    async with get_db() as db:
        for policy_data in policies:
            policy = Policy(**policy_data)
            db.add(policy)
        await db.commit()
    
    print(f"   ✅ Added {len(policies)} policies")


async def seed_orders():
    """Seed sample orders."""
    print("📋 Seeding orders...")
    
    orders = [
        {
            "id": "ORD-2024-001234",
            "user_phone": "9876543210",
            "status": "delivered",
            "total_amount": 51498.00,
            "shipping_address": "123, Green Park, New Delhi - 110016",
            "estimated_delivery": datetime.now() - timedelta(days=2),
            "tracking_number": "TRACK123456789",
            "shipping_carrier": "BlueDart"
        },
        {
            "id": "ORD-2024-001235",
            "user_phone": "9876543210",
            "status": "shipped",
            "total_amount": 1798.00,
            "shipping_address": "123, Green Park, New Delhi - 110016",
            "estimated_delivery": datetime.now() + timedelta(days=2),
            "tracking_number": "TRACK987654321",
            "shipping_carrier": "Delhivery"
        },
        {
            "id": "ORD-2024-001236",
            "user_phone": "8765432109",
            "status": "processing",
            "total_amount": 5698.00,
            "shipping_address": "45, MG Road, Mumbai - 400001",
            "estimated_delivery": datetime.now() + timedelta(days=4)
        },
        {
            "id": "ORD-2024-001237",
            "user_phone": "7654321098",
            "status": "cancelled",
            "total_amount": 89999.00,
            "shipping_address": "78, Park Street, Kolkata - 700016"
        },
        {
            "id": "ORD-2024-001238",
            "user_phone": "6543210987",
            "status": "confirmed",
            "total_amount": 3298.00,
            "shipping_address": "22, Banjara Hills, Hyderabad - 500034",
            "estimated_delivery": datetime.now() + timedelta(days=5)
        }
    ]
    
    order_items = [
        # Items for ORD-2024-001234
        {"id": gen_id(), "order_id": "ORD-2024-001234", "product_id": "PHONE-001", "quantity": 1, "unit_price": 49999.00},
        {"id": gen_id(), "order_id": "ORD-2024-001234", "product_id": "HEADPHONES-001", "quantity": 1, "unit_price": 1499.00},
        # Items for ORD-2024-001235
        {"id": gen_id(), "order_id": "ORD-2024-001235", "product_id": "SHIRT-001", "quantity": 1, "unit_price": 1299.00},
        {"id": gen_id(), "order_id": "ORD-2024-001235", "product_id": "BOOK-001", "quantity": 1, "unit_price": 499.00},
        # Items for ORD-2024-001236
        {"id": gen_id(), "order_id": "ORD-2024-001236", "product_id": "MIXER-001", "quantity": 1, "unit_price": 3499.00},
        {"id": gen_id(), "order_id": "ORD-2024-001236", "product_id": "COOKER-001", "quantity": 1, "unit_price": 2199.00},
        # Items for ORD-2024-001237
        {"id": gen_id(), "order_id": "ORD-2024-001237", "product_id": "LAPTOP-001", "quantity": 1, "unit_price": 89999.00},
        # Items for ORD-2024-001238
        {"id": gen_id(), "order_id": "ORD-2024-001238", "product_id": "JEANS-001", "quantity": 1, "unit_price": 1999.00},
        {"id": gen_id(), "order_id": "ORD-2024-001238", "product_id": "SHIRT-001", "quantity": 1, "unit_price": 1299.00},
    ]
    
    async with get_db() as db:
        # Add orders
        for order_data in orders:
            order = Order(**order_data)
            db.add(order)
        
        await db.flush()
        
        # Add order items
        for item_data in order_items:
            item = OrderItem(**item_data)
            db.add(item)
        
        await db.commit()
    
    print(f"   ✅ Added {len(orders)} orders with {len(order_items)} items")


async def main():
    """Seed all data."""
    print("🌱 Starting database seeding...\n")
    
    # Initialize database first
    await init_db()
    
    # Seed data
    await seed_products()
    await seed_faqs()
    await seed_policies()
    await seed_orders()
    
    print("\n✅ Database seeding complete!")
    print("\n📊 Summary:")
    print("   - 10 Products")
    print("   - 10 FAQs")
    print("   - 5 Policies")
    print("   - 5 Orders")


if __name__ == "__main__":
    asyncio.run(main())
