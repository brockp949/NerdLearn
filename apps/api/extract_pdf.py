
import PyPDF2
import sys

def extract_text(pdf_path):
    try:
        reader = PyPDF2.PdfReader(pdf_path)
        text = ""
        # Read first 10 pages or all if fewer
        num_pages = len(reader.pages)
        print(f"Total pages: {num_pages}")
        
        for i in range(min(num_pages, 10)):
            page = reader.pages[i]
            text += page.extract_text() + "\n\n"
            
        return text
    except Exception as e:
        return f"Error: {e}"

if __name__ == "__main__":
    pdf_path = r"C:\Users\brock\Documents\NerdLearn\NerdLearn\Neural ODEs for Personalized Memory Decay.pdf"
    print(extract_text(pdf_path))
