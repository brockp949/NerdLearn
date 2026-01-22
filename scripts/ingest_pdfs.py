import requests
import os
import sys

API_URL = "http://localhost:8000/api"

def get_or_create_course():
    # List courses
    try:
        response = requests.get(f"{API_URL}/courses/")
        courses = response.json()
        for course in courses:
            if course["title"] == "NerdLearn Research":
                print(f"Found existing course: {course['id']}")
                return course["id"]
    except Exception as e:
        print(f"Error accessing API: {e}. Is the backend running?")
        return None

    # Create course if not found
    print("Creating 'NerdLearn Research' course...")
    payload = {
        "title": "NerdLearn Research",
        "description": "Foundational research papers for the NerdLearn platform.",
        "instructor_id": 1, # Mock instructor
        "difficulty_level": "advanced",
        "tags": ["research", "adaptive-learning", "gamification"],
        "price": 0
    }
    response = requests.post(f"{API_URL}/courses/", json=payload)
    if response.status_code == 201:
        return response.json()["id"]
    else:
        print(f"Failed to create course: {response.text}")
        return None

def ingest_pdfs(directory, course_id):
    files = [f for f in os.listdir(directory) if f.lower().endswith('.pdf')]
    print(f"Found {len(files)} PDFs.")

    for i, filename in enumerate(files):
        print(f"Uploading {filename}...")
        file_path = os.path.join(directory, filename)
        
        with open(file_path, 'rb') as f:
            files_payload = {'file': (filename, f, 'application/pdf')}
            data_payload = {
                'title': filename.replace(".pdf", ""),
                'module_type': 'pdf',
                'order': i
            }
            
            response = requests.post(
                f"{API_URL}/courses/{course_id}/modules",
                files=files_payload,
                data=data_payload
            )
            
            if response.status_code == 201:
                print(f"Success: {filename}")
            else:
                print(f"Failed: {filename} - {response.text}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ingest_pdfs.py <pdf_directory>")
        sys.exit(1)

    course_id = get_or_create_course()
    if course_id:
        ingest_pdfs(sys.argv[1], course_id)
