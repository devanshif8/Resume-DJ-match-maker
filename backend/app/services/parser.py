from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_resume_chunks(pdf_path: str):
    """
    Load a PDF resume and split it into chunks.
    Returns LangChain document chunks.
    """
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120
    )

    chunks = splitter.split_documents(pages)
    return chunks


def resume_chunks_to_text(resume_chunks):
    """
    Combine all chunk text into one lowercase resume text blob
    for quick keyword matching.
    """
    return "\n".join(chunk.page_content for chunk in resume_chunks)