"""
AWS PDF Chat Application by Manish Singh
This script allows users to upload PDFs, process them, and use chat-based interaction to query the content of the PDF using AWS services.
"""
import boto3
import PyPDF2
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI

# AWS and OpenAI Configuration
AWS_REGION = "us-east-1"
S3_BUCKET_NAME = "aws-pdf-chat-app-manish-singh"
OPENAI_API_KEY = "your_openai_api_key"

def upload_pdf_to_s3(file_path, bucket_name, file_key):
    """
    Upload a PDF file to an AWS S3 bucket.
    """
    s3_client = boto3.client('s3', region_name=AWS_REGION)
    s3_client.upload_file(file_path, bucket_name, file_key)
    print(f"Uploaded {file_path} to S3 bucket {bucket_name} as {file_key}.")

def download_pdf_from_s3(bucket_name, file_key, download_path):
    """
    Download a PDF file from an AWS S3 bucket.
    """
    s3_client = boto3.client('s3', region_name=AWS_REGION)
    s3_client.download_file(bucket_name, file_key, download_path)
    print(f"Downloaded {file_key} from S3 bucket {bucket_name} to {download_path}.")

def extract_text_from_pdf(file_path):
    """
    Extract text from a PDF file using PyPDF2.
    """
    with open(file_path, 'rb') as pdf_file:
        reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
    return text

def initialize_chat_system(text_data):
    """
    Initialize a conversational chat system with the PDF text data.
    """
    # Load the PDF text into a retriever
    loader = PyPDFLoader(text_data)
    documents = loader.load()
    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
    vector_store = FAISS.from_documents(documents, embeddings)
    
    # Create a chat chain
    retriever = vector_store.as_retriever()
    llm = OpenAI(api_key=OPENAI_API_KEY)
    chain = ConversationalRetrievalChain(retriever=retriever, llm=llm)
    return chain

def chat_with_pdf(chain, user_query):
    """
    Handle chat interactions with the PDF using the initialized chain.
    """
    response = chain.run(user_query)
    return response

# Example Usage
if __name__ == "__main__":
    # Step 1: Upload PDF to S3
    file_path = "example.pdf"
    file_key = "uploaded_files/example.pdf"
    upload_pdf_to_s3(file_path, S3_BUCKET_NAME, file_key)

    # Step 2: Download PDF from S3
    download_path = "downloaded_example.pdf"
    download_pdf_from_s3(S3_BUCKET_NAME, file_key, download_path)

    # Step 3: Extract text from PDF
    pdf_text = extract_text_from_pdf(download_path)

    # Step 4: Initialize Chat System
    chat_chain = initialize_chat_system(pdf_text)

    # Step 5: Chat with PDF
    user_query = "What is the main topic of the PDF?"
    response = chat_with_pdf(chat_chain, user_query)
    print("Chat Response:", response)
