JD (for genrating dashboards)

import gradio as gr
import google.generativeai as genai
import warnings
from langchain_community.document_loaders import PyPDFLoader  # Updated import path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma  # Updated import path
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.prompts import PromptTemplate
from docx import Document
import os

warnings.filterwarnings("ignore")

# Set API key directly in the script (only for local testing, not recommended for production)
os.environ['GOOGLE_API_KEY'] = 'AIzaSyCdC68JCX4Om-GF7LCCcsey-PAbXPyQO9g'  # Replace with your actual API key

# Retrieve the API key
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

if not GOOGLE_API_KEY:
    raise ValueError("Please set the GOOGLE_API_KEY environment variable.")

genai.configure(api_key=GOOGLE_API_KEY)
model = ChatGoogleGenerativeAI(
    model="gemini-pro", 
    google_api_key=GOOGLE_API_KEY, 
    temperature=0.2, 
    convert_system_message_to_human=True
)

qa_template = """You are an expert in analyzing job descriptions and creating customized dashboards. Your task is to thoroughly review the provided job description for various roles and design multiple, concise, and insightful dashboards that highlight key responsibilities, required skills, and other relevant details.
{context}
Your Task:"""
QA_CHAIN_PROMPT = PromptTemplate.from_template(qa_template)

def process_job_description(context):
    try:
        # Split the job description into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
        texts = text_splitter.split_text(context)

        # Initialize Google Generative AI embeddings and vector store
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
        vector_index = Chroma.from_texts(texts, embeddings)

        # Set up the retriever for QA
        n_elements = len(texts)
        search_results = max(5, min(10, n_elements))  # Ensure at least 5 search results if possible
        retriever = vector_index.as_retriever(search_kwargs={"k": search_results})

        # Create a QA chain using the prompt template
        qa_chain = RetrievalQA.from_chain_type(
            model,
            retriever=retriever,
            chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
        )

        return qa_chain

    except Exception as e:
        return f"Error during QA extraction: {e}"

def extract_text_from_pdf(pdf_file_path):
    # Verify if the file exists
    if not os.path.isfile(pdf_file_path):
        return f"Error: File not found at {pdf_file_path}"

    loader = PyPDFLoader(pdf_file_path)
    pages = loader.load()
    text = ""
    for page in pages:
        text += page.page_content
    return text

def extract_text_from_docx(docx_file_path):
    # Verify if the file exists
    if not os.path.isfile(docx_file_path):
        return f"Error: File not found at {docx_file_path}"

    doc = Document(docx_file_path)
    full_text = []
    for paragraph in doc.paragraphs:
        full_text.append(paragraph.text)
    return "\n".join(full_text)

def format_output(text):
    """
    Formats the output text to ensure neat alignment and readability.
    """
    # Remove asterisks and replace bullet points for list items
    formatted_text = text.replace("**", "")  # Remove bold formatting markers
    formatted_text = formatted_text.replace("\n*", "\n-")  # Replace bullet points with hyphens
    formatted_text = formatted_text.replace("\n-", "\n\n-")  # Add spacing between bullet points
    formatted_text = formatted_text.replace("\n\n", "\n")  # Reduce excessive newlines

    # Ensure paragraphs are well-spaced
    formatted_text = "\n\n".join([line.strip() for line in formatted_text.splitlines() if line.strip() != ""])

    return formatted_text

def remove_dashboard_names(text):
    """
    Removes any dashboard names or headers from the generated text.
    """
    # Define patterns or specific phrases that represent dashboard names
    unwanted_phrases = ["Dashboard:", "Responsibilities Dashboard", "Skills Dashboard", "Dashboard"]

    # Remove lines that contain unwanted phrases
    filtered_lines = [line for line in text.splitlines() if not any(phrase in line for phrase in unwanted_phrases)]

    return "\n".join(filtered_lines)

def generate_dashboards(file):
    # Determine the file type and extract text accordingly
    file_extension = os.path.splitext(file.name)[1].lower()
    if file_extension == ".pdf":
        job_description = extract_text_from_pdf(file.name)
    elif file_extension == ".docx":
        job_description = extract_text_from_docx(file.name)
    else:
        return "Unsupported file type. Please upload a PDF or DOCX file."

    # Check if extraction was successful
    if job_description.startswith("Error:"):
        return job_description

    # Process the job description
    qa_chain = process_job_description(job_description)
    if isinstance(qa_chain, str):  # Error case
        return qa_chain

    # Generate dashboards
    answers = qa_chain.run(job_description)

    # Remove any dashboard titles or labels from the answers
    clean_answers = remove_dashboard_names(answers)

    # Format the output for neat alignment
    formatted_output = format_output(clean_answers)

    # Return the formatted generated dashboards
    return formatted_output

# Create the Gradio interface
interface = gr.Interface(
    fn=generate_dashboards,
    inputs=gr.File(label="Upload Job Description PDF or DOCX"),  # Updated label to include DOCX
    outputs=gr.Textbox(label="Generated Dashboards"),
    title="Job Description Dashboard Generator",
    description="Upload a job description in PDF or DOCX format, and this tool will generate at least 5 customized dashboards based on the provided job description."
)

# Launch the interface
interface.launch()
