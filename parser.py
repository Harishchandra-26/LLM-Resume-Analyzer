import os
import sys
import tempfile
from typing import Optional

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_groq import ChatGroq

from models import Resume
from prompts import resume_prompt


# -----------------------------------------------------
# Load Environment Variables
# -----------------------------------------------------

load_dotenv()


# -----------------------------------------------------
# Create Groq LLM
# -----------------------------------------------------

def get_llm(
    model_name: str = "openai/gpt-oss-120b",
    temperature: float = 0,
) -> ChatGroq:
    """
    Returns a configured ChatGroq instance.
    """

    # Load .env file
    load_dotenv()

    # Get Groq API key
    api_key = os.getenv("GROQ_API_KEY")

    # Check API key
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY not found. "
            "Please add it to your .env file."
        )

    # Create Groq LLM
    return ChatGroq(
        model=model_name,
        temperature=temperature,
        groq_api_key=api_key,
    )


# -----------------------------------------------------
# Read Resume PDF
# -----------------------------------------------------

def load_resume_text_from_path(pdf_path: str) -> str:
    """
    Load a resume PDF and return all extracted text.
    """

    if not os.path.exists(pdf_path):
        raise FileNotFoundError(
            f"Resume file not found: {pdf_path}"
        )

    loader = PyPDFLoader(pdf_path)

    documents = loader.load()

    text = "\n".join(
        document.page_content
        for document in documents
    )

    if not text.strip():
        raise ValueError(
            "No text could be extracted from the resume PDF."
        )

    return text


# -----------------------------------------------------
# Parse Resume
# -----------------------------------------------------

def parse_resume(
    resume_text: str,
    llm: Optional[ChatGroq] = None,
) -> Resume:
    """
    Convert resume text into a structured Resume object.
    """

    if not resume_text.strip():
        raise ValueError(
            "Resume text is empty."
        )

    # Create LLM if one was not provided
    if llm is None:
        llm = get_llm()

    # Enable structured output
    structured_llm = llm.with_structured_output(
        Resume,
        method="function_calling",
    )

    # Create LangChain chain
    chain = resume_prompt | structured_llm

    # Invoke the chain
    result = chain.invoke(
        {
            "resume_text": resume_text,
        }
    )

    return result


# -----------------------------------------------------
# Convenience Function
# -----------------------------------------------------

def parse_resume_from_path(
    pdf_path: str,
) -> Resume:
    """
    Parse a resume directly from a PDF path.
    """

    text = load_resume_text_from_path(pdf_path)

    return parse_resume(text)


# -----------------------------------------------------
# Streamlit Helper
# -----------------------------------------------------

def load_resume_text_from_bytes(
    file_bytes: bytes,
    suffix: str = ".pdf",
) -> str:
    """
    Convert uploaded PDF bytes into text.
    """

    temp_file = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix,
    )

    try:
        # Write uploaded bytes
        temp_file.write(file_bytes)
        temp_file.close()

        # Extract text
        return load_resume_text_from_path(
            temp_file.name
        )

    finally:
        # Delete temporary file
        if os.path.exists(temp_file.name):
            os.remove(temp_file.name)


# -----------------------------------------------------
# CLI Support
# -----------------------------------------------------

if __name__ == "__main__":

    load_dotenv()

    # Check command-line argument
    if len(sys.argv) < 2:
        print(
            "Usage: python parser.py resume.pdf"
        )
        sys.exit(1)

    pdf_path = sys.argv[1]

    try:
        # Parse resume
        resume = parse_resume_from_path(
            pdf_path
        )

        # Print structured JSON
        print(
            resume.model_dump_json(
                indent=4
            )
        )

    except Exception as e:
        print(
            f"\nError while parsing resume:\n{e}"
        )
        sys.exit(1)