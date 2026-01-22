import os
import asyncio
import sys

# Add project root to path to find 'app' package
# Assuming structure: NerdLearn/scripts/this_script.py and NerdLearn/apps/api/app
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
api_root = os.path.join(project_root, "apps", "api")
sys.path.append(api_root)

from app.services.ingestion_service import IngestionService

async def ingest_directory(directory_path: str):
    # Initialize service (DB optional for this script if not used by vector store directly)
    service = IngestionService()
    
    print(f"Scanning {directory_path} for PDFs...")
    if not os.path.exists(directory_path):
         print(f"Directory not found: {directory_path}")
         return

    files = [f for f in os.listdir(directory_path) if f.lower().endswith('.pdf')]
    
    if not files:
        print("No PDF files found.")
        return

    print(f"Found {len(files)} PDFs. Starting ingestion...")
    
    for filename in files:
        file_path = os.path.join(directory_path, filename)
        print(f"Ingesting: {filename}")
        try:
            # Defaulting to course_id=1 for this script
            count = await service.ingest_file(file_path, course_id=1)
            print(f"  Successfully indexed {count} chunks.")
        except Exception as e:
            print(f"  Failed: {e}")

    print("Ingestion complete.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ingest_pdfs.py <directory_path>")
        sys.exit(1)
    
    directory = sys.argv[1]
    asyncio.run(ingest_directory(directory))
