import os
import asyncio
import sys

# Add project root to path to find 'app' package
# Assuming structure: NerdLearn/scripts/this_script.py and NerdLearn/apps/api/app
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
api_root = os.path.join(project_root, "apps", "api")
sys.path.append(api_root)

from app.services.ingestion_service import IngestionService
from app.core.database import AsyncSessionLocal, Base, engine
from app.models import User, Instructor, Course
import app.models # Ensure all models are loaded for Base.metadata
from sqlalchemy import select, text

async def seed_dependencies(db):
    """Ensure Schema and Dependencies exist"""
    print("Creating database schema...")
    async with engine.begin() as conn:
        # Enable pgvector extension
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.run_sync(Base.metadata.create_all)

    print("Seeding SQL dependencies...")
    try:
        # 1. User
        res = await db.execute(select(User).filter_by(username="admin_ingest"))
        user = res.scalars().first()
        if not user:
            print("Creating Admin User...")
            user = User(
                username="admin_ingest", 
                email="admin_ingest@nerdlearn.io", 
                hashed_password="mock_password",
                is_active=True
            )
            db.add(user)
            await db.flush()
        
        # 2. Instructor
        res = await db.execute(select(Instructor).filter_by(user_id=user.id))
        instructor = res.scalars().first()
        if not instructor:
            print("Creating Instructor...")
            instructor = Instructor(user_id=user.id)
            db.add(instructor)
            await db.flush()
            
        # 3. Course
        res = await db.execute(select(Course).filter_by(id=1))
        course = res.scalars().first()
        if not course:
            print("Creating Course 1...")
            course = Course(
                id=1,
                title="Research Ingestion",
                instructor_id=instructor.id,
                description="Ingested Research Papers"
            )
            db.add(course)
        
        await db.commit()
        print("Dependencies seeded.")
    except Exception as e:
        print(f"Seeding failed: {e}")
        await db.rollback()
        raise

async def ingest_directory(directory_path: str):
    print(f"Scanning {directory_path} for PDFs...")
    if not os.path.exists(directory_path):
         print(f"Directory not found: {directory_path}")
         return

    files = [f for f in os.listdir(directory_path) if f.lower().endswith('.pdf')]
    
    if not files:
        print("No PDF files found.")
        return

    print(f"Found {len(files)} PDFs. Starting ingestion...")

    # Phase 0: Seed
    async with AsyncSessionLocal() as db:
        await seed_dependencies(db)

    # Phase 1: Ingestion
    print("--- Phase 1: Ingestion ---")
    async with AsyncSessionLocal() as db:
        service = IngestionService(db=db)
        
        # Ensure Graph Course Node exists!
        print("Ensuring Graph Course Node exists...")
        await service.graph_service.create_course_node(1, "Research Ingestion")
        
        for filename in files:
            file_path = os.path.join(directory_path, filename)
            print(f"Ingesting: {filename}")
            try:
                count = await service.ingest_file(file_path, course_id=1)
                print(f"  Successfully indexed {count} chunks.")
            except Exception as e:
                print(f"  FAILED to ingest {filename}: {e}")
                # We don't traceback here to keep output clean, but we could
                
    # Phase 2: Optimization
    print("\n--- Phase 2: Optimization ---")
    async with AsyncSessionLocal() as db:
        service = IngestionService(db=db)
        print("Running GraphRAG optimization (Community Detection)...")
        try:
            await service.optimize_course(course_id=1)
            print("Optimization complete.")
        except Exception as e:
            print(f"Optimization failed: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ingest_pdfs.py <directory_path>")
        sys.exit(1)
    
    directory = sys.argv[1]
    asyncio.run(ingest_directory(directory))
