import streamlit as st
from rag_pipeline import RagPipeline

# Function to run the application
def run():
    # Display app title
    st.title("RAG Chatbot with LLama-3-1B")

    # Upload PDF file
    uploaded_file = st.file_uploader("Upload a PDF document", type="pdf")

    # Input for user query
    user_query = st.text_input("Ask a question about the document:")

    if uploaded_file is not None and user_query:
        # Save the uploaded PDF to a file
        with open("uploaded_document.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Instantiate the RagPipeline class
        rag_pipeline = RagPipeline(pdf_file="uploaded_document.pdf")

        # Run the pipeline to get the response
        response = rag_pipeline.run_pipeline(user_query)

        # Display the response
        st.write(f"Answer: {response}")

    elif uploaded_file is not None:
        st.write("Please enter a question to get an answer from the document.")
    elif user_query:
        st.write("Please upload a PDF document first.")

# Run the Streamlit app
if __name__ == "__main__":
    run()