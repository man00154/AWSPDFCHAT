from fastapi import FastAPI, UploadFile, File, HTTPException
from app.utils import process_pdf, upload_to_s3, generate_chat_response
from app.models import create_db_and_tables, PDFContent

app = FastAPI(title="AWS PDF Chat by Manish Singh")

@app.on_event("startup")
async def startup_event():
    """
    Initialize the database tables during startup.
    """
    create_db_and_tables()

@app.post("/upload/")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload a PDF file, process it, and store its contents.

    Parameters:
    - file (UploadFile): The PDF file to be uploaded.

    Returns:
    - dict: Message indicating success.
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    try:
        # Process the PDF file to extract text
        text = process_pdf(file.file)
        
        # Reset file pointer for re-uploading
        file.file.seek(0)
        
        # Upload the file to AWS S3
        upload_to_s3(file.file, file.filename)

        # Save extracted text to the database
        pdf_content = PDFContent(filename=file.filename, content=text)
        pdf_content.save()

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

    return {"message": "PDF uploaded and processed successfully by AWS PDF Chat by Manish Singh"}

@app.post("/chat/")
async def chat_with_pdf(query: str, file_id: int):
    """
    Chat with the contents of an uploaded PDF file.

    Parameters:
    - query (str): The user's question/query.
    - file_id (int): The ID of the uploaded PDF file.

    Returns:
    - dict: Chat response generated from the PDF content.
    """
    try:
        pdf_content = PDFContent.get(file_id)
        if not pdf_content:
            raise HTTPException(status_code=404, detail="PDF not found")
        
        # Generate a response using the GPT model
        response = generate_chat_response(query, pdf_content.content)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
    
    return {"response": f"Chat response (AWS PDF Chat by Manish Singh): {response}"}

if __name__ == "__main__":
    # Use uvicorn to run the application on port 8080
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
