import requests
from bs4 import BeautifulSoup
from mistralai import Mistral
import faiss
import numpy as np
from dotenv import load_dotenv
import os
import streamlit as st
from mistralai.models import SDKError
import time


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

def get_text_embeddings(text_chunks, api_key, max_retries=3, retry_delay=5):
    """
    Generate embeddings for a list of text chunks using MistralAI.
    
    Args:
        text_chunks (list): List of text chunks to generate embeddings for.
        api_key (str): MistralAI API key.
        max_retries (int): Maximum number of retries for rate limit errors.
        retry_delay (int): Delay (in seconds) between retries.
    
    Returns:
        list: List of embeddings for the input text chunks.
    """
    client = Mistral(api_key=api_key)
    
    for attempt in range(max_retries):
        try:
            # Attempt to generate embeddings
            embeddings_batch_response = client.embeddings.create(
                model="mistral-embed",
                inputs=text_chunks
            )
            # Return the embeddings
            return [embedding.embedding for embedding in embeddings_batch_response.data]
        
        except SDKError as e:
            # Handle rate limit errors (status code 429)
            if "Status 429" in str(e):
                print(f"Rate limit exceeded. Retrying in {retry_delay} seconds... (Attempt {attempt + 1}/{max_retries})")
                time.sleep(retry_delay)  # Wait before retrying
            else:
                # Re-raise the error if it's not a rate limit issue
                print(f"MistralAI API error: {e}")
                raise e
        
        except Exception as e:
            # Handle other unexpected errors
            print(f"Unexpected error: {e}")
            raise e
    
    # If all retries fail, raise an exception
    raise Exception(f"Failed to generate embeddings after {max_retries} retries due to rate limiting.")

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
        return "Your query isn't related to UDST policies."
    except Exception as e:
        print(f"Unexpected error: {e}")
        return "An unexpected error occurred try again in a couple of minutes."
