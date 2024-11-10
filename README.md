# AI Document Retrieval and Question-Answering Pipeline

This project implements an AI-powered pipeline to retrieve relevant document chunks and answer user queries based on PDF documents. It leverages **LangChain**, **Hugging Face**, and **Chroma** for document processing, embedding generation, and question-answering tasks using a **Llama** language model. The solution is built as a web application using **Streamlit**.

### Table of Contents
- [Overview](#overview)
- [Project Flow](#project-flow)
- [Technologies](#technologies)
- [Installation](#installation)
- [Usage](#usage)
- [File Structure](#file-structure)
- [License](#license)

### Overview

This pipeline takes a PDF file as input, processes it to extract relevant information, and stores the data in vector embeddings. Then, a **Llama language model** is used to answer user queries based on the extracted documents. The key components of the pipeline include:

1. **PDF Document Loading**: Load PDF documents and split them into manageable chunks.
2. **Document Vectorization**: Convert the chunks into vector embeddings using Hugging Face’s model.
3. **Vector Store Creation**: Store these embeddings in a Chroma vector store for efficient retrieval.
4. **Retrieval-Based Question Answering**: A **RetrievalQA** pipeline is used to retrieve relevant document chunks and answer user queries.
5. **Interactive Streamlit UI**: Users can interact with the system to upload PDFs and query the model via a simple Streamlit web interface.

### Project Flow

#### Step-by-Step Process:

1. **PDF Document Upload**:
    - The user uploads a PDF document to the system via the **Streamlit interface**.
    - The PDF is loaded using the **PyPDFLoader** class from LangChain.

2. **Document Chunking**:
    - The loaded PDF content is split into smaller, manageable chunks using **RecursiveCharacterTextSplitter**. This ensures that each chunk fits within the model's input constraints.

3. **Embedding Generation**:
    - The text chunks are embedded using the **Hugging Face embeddings** model.
    - The embeddings are then stored in **Chroma**, a vector store, to allow for efficient retrieval during the question-answering phase.

4. **Model Initialization**:
    - The **Llama model** is loaded using **Hugging Face Transformers** and moved to either the CPU or GPU for processing.
    - The tokenizer and model are initialized.

5. **Retrieval QA Setup**:
    - The **RetrievalQA** chain is created, linking the **Llama model** and the **Chroma vector store**.
    - This setup ensures that the model can query the vector store for the most relevant document chunks when answering user questions.

6. **User Query and Answer Generation**:
    - When the user inputs a query, the **RetrievalQA** chain searches for the most relevant document chunks using cosine similarity.
    - The retrieved document chunks are fed into the **Llama model**, which generates the final answer based on the information retrieved.

7. **Streamlit Interface**:
    - The user interacts with the pipeline through a simple web-based interface built using **Streamlit**.
    - Users can upload PDFs and input their queries, receiving answers directly in the browser.

### Technologies

- **LangChain**: For managing document loading, text splitting, and question-answering chain.
- **Hugging Face**: For using pre-trained models (Llama, embeddings).
- **Chroma**: A vector store for efficient retrieval of document embeddings.
- **Streamlit**: For building the interactive web interface.
- **PyTorch**: For running and managing the Llama model.

### Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/yourusername/AI-Document-Retrieval-QA.git
    cd AI-Document-Retrieval-QA
    ```

2. Create and activate a virtual environment:

    ```bash
    python -m venv venv
    source venv/bin/activate   # On Windows, use venv\Scripts\activate
    ```

3. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

4. **Set up environment variables**:
   - Create a `.env` file in the root directory of the project with the following content (ensure it does not contain any sensitive tokens):

    ```bash
    MODEL_NAME="your_model_name_here"
    HUGGINGFACE="your_huggingface_token_here"
    ```

### Usage

1. **Run the Streamlit application**:

    ```bash
    streamlit run app.py
    ```

2. **Upload a PDF**:
   - Open your browser and navigate to the Streamlit app.
   - Upload a PDF document.

3. **Ask a question**:
   - Once the document is loaded and processed, input a query related to the content in the uploaded PDF.

4. **View the answer**:
   - The model will return an answer based on the content of the document, sourced from the relevant sections.

### File Structure

```bash
.
├── app.py                      # Streamlit app entry point
├── rag_pipeline.py             # Pipeline logic and functions
├── requirements.txt            # Python dependencies
├── .env                        # Environment variables (MODEL_NAME, HUGGINGFACE token)
├── utils/                      # Helper functions (if any)
│   └── ...
└── README.md                   # Project documentation
