"""
Database Initialization Script.
Creates all tables and sets up the database.
"""

import asyncio
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.db.database import init_db


async def main():
    """Initialize the database."""
    print("🗄️  Initializing database...")
    
    await init_db()
    
    print("✅ Database initialized successfully!")
    print("📁 Database location: data/app.db")


if __name__ == "__main__":
    asyncio.run(main())
