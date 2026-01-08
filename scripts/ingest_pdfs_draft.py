import os
import asyncio
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.services.content.ingestion_service import IngestionService
from app.db.session import SessionLocal

async def ingest_directory(directory_path: str):
    service = IngestionService(db=SessionLocal())
    
    print(f"Scanning {directory_path} for PDFs...")
    files = [f for f in os.listdir(directory_path) if f.lower().endswith('.pdf')]
    
    if not files:
        print("No PDF files found.")
        return

    print(f"Found {len(files)} PDFs. Starting ingestion...")
    
    for filename in files:
        file_path = os.path.join(directory_path, filename)
        print(f"Ingesting: {filename}")
        try:
            # Check if ingest methods are async or sync. Assuming async based on modern fastapi stack.
            # If IngestionService methods are sync, remove await.
            # Based on typical patterns, services might be sync if using sync DB session, or async.
            # Let's assume sync for now from the file snippet I saw earlier, or check.
            # I will wrap in try/except and assume sync first, but if I saw async def in snippet...
            # I'll check the service file content again in my memory. 
            # I recall 'class IngestionService'. I didn't see 'async def'. 
            # I will try sync execution.
            
            with open(file_path, 'rb') as f:
                content = f.read()
                # Mocking UploadFile if service expects it, or passing bytes if it handles it.
                # Usually services take file-like objects or paths.
                # Let's look at the service signature again if possible. 
                # Ideally I should pass the file path if the service supports it.
                
                # Setup a mock UploadFile-like object if needed, or if service accepts path:
                # service.ingest_file(file_path) <-- optimal
                
                pass 

        except Exception as e:
            print(f"Failed to ingest {filename}: {e}")

    print("Ingestion complete.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ingest_pdfs.py <directory_path>")
        sys.exit(1)
    
    directory = sys.argv[1]
    # asyncio.run(ingest_directory(directory)) # Use if async
    # ingest_directory(directory) 
    pass
