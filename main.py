from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os

# Configuration
DATA_PATH = "data/"
DB_FAISS_PATH = "vectorstore/db_faiss"

def load_pdf_files(data_path):
    """
    Load all PDF files from the specified directory
    """
    print(f"Loading PDFs from: {data_path}")
    
    # Check if directory exists
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Directory {data_path} does not exist")
    
    # Load PDFs using DirectoryLoader
    loader = DirectoryLoader(
        data_path,
        glob='*.pdf',
        loader_cls=PyPDFLoader
    )
    
    documents = loader.load()
    print(f"Loaded {len(documents)} pages from PDFs")
    
    # Print information about loaded documents
    pdf_files = set()
    for doc in documents:
        if hasattr(doc, 'metadata') and 'source' in doc.metadata:
            pdf_files.add(os.path.basename(doc.metadata['source']))
    
    print(f"PDF files processed: {list(pdf_files)}")
    return documents

def create_chunks(documents, chunk_size=500, chunk_overlap=50):
    """
    Split documents into smaller chunks for better retrieval
    """
    print(f"Creating chunks with size={chunk_size}, overlap={chunk_overlap}")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    text_chunks = text_splitter.split_documents(documents)
    print(f"Created {len(text_chunks)} text chunks")
    return text_chunks

def get_embedding_model():
    """
    Initialize the embedding model
    """
    print("Loading embedding model...")
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    print("Embedding model loaded successfully")
    return embedding_model

def create_vector_store(text_chunks, embedding_model, save_path):
    """
    Create FAISS vector store from text chunks and save it locally
    """
    print("Creating FAISS vector store...")
    
    # Create vector store
    db = FAISS.from_documents(text_chunks, embedding_model)
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Save the vector store
    db.save_local(save_path)
    print(f"Vector store saved to: {save_path}")
    return db

def main():
    """
    Main function to orchestrate the PDF processing pipeline
    """
    print("=== PDF Chatbot Setup ===")
    
    try:
        # Step 1: Load raw PDF files
        print("\n1. Loading PDF files...")
        documents = load_pdf_files(DATA_PATH)
        
        if not documents:
            print("No PDFs found. Please add PDF files to the data/ directory")
            return
        
        # Step 2: Create chunks
        print("\n2. Creating text chunks...")
        text_chunks = create_chunks(documents)
        
        # Step 3: Create vector embeddings
        print("\n3. Loading embedding model...")
        embedding_model = get_embedding_model()
        
        # Step 4: Store embeddings in FAISS
        print("\n4. Creating and saving vector store...")
        db = create_vector_store(text_chunks, embedding_model, DB_FAISS_PATH)
        
        print("\n✅ Setup completed successfully!")
        print(f"   - Processed {len(documents)} pages")
        print(f"   - Created {len(text_chunks)} chunks")
        print(f"   - Vector store saved to: {DB_FAISS_PATH}")
        
        return db
        
    except Exception as e:
        print(f"❌ Error during setup: {str(e)}")
        return None

# Alternative method: Load specific PDF files by name
def load_specific_pdfs(pdf_paths):
    """
    Load specific PDF files by their paths
    """
    print(f"Loading specific PDFs: {pdf_paths}")
    all_documents = []
    
    for pdf_path in pdf_paths:
        if not os.path.exists(pdf_path):
            print(f"Warning: {pdf_path} does not exist")
            continue
            
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        all_documents.extend(documents)
        print(f"Loaded {len(documents)} pages from {os.path.basename(pdf_path)}")
    
    print(f"Total pages loaded: {len(all_documents)}")
    return all_documents

if __name__ == "__main__":
    # Method 1: Load all PDFs from directory
    db = main()
    
    # Method 2: Load specific PDFs (uncomment if needed)
    # specific_pdfs = ["data/document1.pdf", "data/document2.pdf"]
    # documents = load_specific_pdfs(specific_pdfs)
    # text_chunks = create_chunks(documents)
    # embedding_model = get_embedding_model()
    # db = create_vector_store(text_chunks, embedding_model, DB_FAISS_PATH)