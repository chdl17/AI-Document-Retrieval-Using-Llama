### Importing all the libraries
from dotenv import load_dotenv
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain.schema import Document
from langchain.chains import RetrievalQA
from transformers import LlamaForCausalLM, LlamaTokenizer
from chromadb import Client
import torch

load_dotenv()
# function to load PDF pages
class RagPipeline():
    def __init__(self, pdf_file):
        self.pdf_file = pdf_file
        self.model_name = os.getenv("MODEL_NAME")
        self.documents=[]
        self.token = os.getenv("HUGGINGFACE")
        self.vector_stores = None
        self.llama_model = None
        self.chat_prompt = None
        self.retriever_qa_chain = None

    def pdfloader(self):
        loader = PyPDFLoader(self.pdf_file)
        documents = []
        
        # Load pages using lazy_load
        try:
            for page in loader.lazy_load():
                # Create a Document object with proper content
                document = Document(page_content=page.page_content, metadata={"source": self.pdf_file})
                documents.append(document)
        except Exception as e:
            print(f"Error: {e}")
        
        return documents
            
    # function to process data (split into chunks)
    def split_documents(self, chunk_size=500, chunk_overlap=50):
        # Initialize the text splitter
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        
        # Split documents into smaller chunks
        chunks = text_splitter.split_documents(self.documents)
        
        return chunks
        
    def create_vector_stores(self, documents):        
        try:
            # Step 1: Split the documents into smaller chunks
            chunks = self.split_documents()  # Assuming `self.documents` is used for splitting

            # Step 2: Extract text from the chunks
            texts = [doc['text'] for doc in chunks]  # Extracting text from chunked documents

            # Step 3: Initialize the Hugging Face embedding model
            embeddings = HuggingFaceEmbeddings(model_name="nomic-ai/nomic-embed-text-v1", model_kwargs={'trust_remote_code': True})
            
            # Step 4: Generate embeddings for the chunks
            embeddings_list = embeddings.embed_documents(texts)
            
            # Debugging: Check if embeddings are valid
            print(f"Generated embeddings: {embeddings_list[:5]}")  # Print first 5 embeddings for inspection
            
            # Step 5: Ensure embeddings are not empty or malformed
            if not embeddings_list or any(len(embedding) == 0 for embedding in embeddings_list):
                raise ValueError("Generated embeddings are empty or invalid.")
            
            # Step 6: Initialize Chroma client and create collection
            client = Client()  # Initialize chromadb client
            chroma_collection = client.create_collection(name="my_vector_store")  # Create a custom collection if necessary

            # Step 7: Add the embeddings to Chroma (if necessary)
            chroma_collection.add(embeddings=embeddings_list, metadatas=None, ids=None)
            
            # Step 8: Store the vector store in the class variable
            self.vector_stores = Chroma.from_documents(chunks, embeddings, collection_name="my_vector_store")
        
        except Exception as e:
            print(f"Error creating vector stores: {str(e)}")


    def load_llama(self):
        # Initialize the tokenizer and model
        self.tokenizer = LlamaTokenizer.from_pretrained(self.model_name, token = self.token, legacy= False)  # Load tokenizer for the model
        self.llama_model = LlamaForCausalLM.from_pretrained(self.model_name, token = self.token)  # Load the model
        
        # Optionally, move to device (e.g., CUDA or CPU) if using large models
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.llama_model.to(device)
    
    def create_chat_prompt(self):
        self.chat_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant in Telecom Industry"),
            ("user", "The user asked: {user_query}. Here are the some of the retrieved document:{retrieved_documents}. Please provide an answer based on this"),
            ("answer", "The answer is the: {response}")    
        ])
        
    def format_prompt(self, user_query, retrieved_documents):
        return self.chat_prompt.format(user_query=user_query, retrieved_documents= retrieved_documents)
    
    def generate_answer(self, prompt):
        response = self.llama_model(prompt)
        return response
    
    def create_retrieval_qa_chain(self):
        retriever = self.vector_stores.as_retriever(
            search_type ='similarity', search_kwargs={"k": 3})
        self.retriever_qa_chain = RetrievalQA(
            llm = self.llama_model,
            retriever = retriever,
            prompt = self.chat_prompt,
            return_source_documents = True
        )
        
    def run_pipeline(self, user_query):
        self.pdfloader()
        documents = self.split_documents()
        self.create_vector_stores(documents)
        self.load_llama()
        self.create_chat_prompt()
        self.create_retrieval_qa_chain()
        
        result = self.retrieval_qa_chain.run(user_query)
        return result
