import requests
from bs4 import BeautifulSoup
from mistralai import Mistral
import faiss
import numpy as np
from dotenv import load_dotenv
import os
import streamlit as st
from mistralai.models import SDKError


# Load environment variables from .env file
load_dotenv()

def rag_pipeline(url):
    """
    RAG pipeline: Load policy text, chunk it, generate embeddings, and create a FAISS index.
    """
    # Get the API key from the environment
    api_key = st.secrets["MISTRAL_API_KEY"]
    if not api_key:
        raise ValueError("MISTRAL_API_KEY not found in .env file.")
    
    # Step 1: Load policy text
    policy_text = load_policy_text(url)
    
    # Step 2: Chunk the text
    text_chunks = chunk_text(policy_text)
    
    # Step 3: Generate embeddings
    embeddings = get_text_embeddings(text_chunks, api_key)
    
    # Step 4: Create FAISS index
    index = create_faiss_index(embeddings)
    
    return index, text_chunks, api_key


def load_policy_text(url):
    """
    Load and parse policy text from a URL.
    """
    response = requests.get(url)
    html_doc = response.text
    soup = BeautifulSoup(html_doc, "html.parser")
    tag = soup.find("div")
    return tag.text

def chunk_text(text, chunk_size=512):
    """
    Split the text into chunks of a specified size.
    """
    return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]

def get_text_embeddings(text_chunks, api_key):
    """
    Generate embeddings for a list of text chunks using MistralAI.
    """
    client = Mistral(api_key=api_key)
    embeddings_batch_response = client.embeddings.create(
        model="mistral-embed",
        inputs=text_chunks
    )
    return [embedding.embedding for embedding in embeddings_batch_response.data]

def create_faiss_index(embeddings):
    """
    Create a FAISS index from a list of embeddings.
    """
    d = len(embeddings[0])  # Dimension of embeddings
    index = faiss.IndexFlatL2(d)  # L2 distance index
    index.add(np.array(embeddings))  # Add embeddings to the index
    return index

def process_query(query, index, text_chunks, api_key):
    """
    Process a user query: retrieve relevant chunks and generate an answer.
    """
    try:
        # Generate embedding for the query
        query_embedding = np.array([get_text_embeddings([query], api_key)[0]])
        
        # Search for relevant chunks
        D, I = index.search(query_embedding, k=2)  # Retrieve top 2 chunks
        retrieved_chunks = [text_chunks[i] for i in I.tolist()[0]]
        
        # Generate answer using MistralAI
        client = Mistral(api_key=api_key)
        prompt = f"Context: {retrieved_chunks}\\nQuery: {query}\\nAnswer:"
        print("Prompt:", prompt)  # Debugging: Check the prompt content
        
        response = client.chat.complete(
            model="mistral-large-latest",
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.choices[0].message.content
    
    except SDKError as e:
        print(f"MistralAI SDK Error: {e}")
        return "An error occurred with the MistralAI API."
    except Exception as e:
        print(f"Unexpected error: {e}")
        return "An unexpected error occurred."
