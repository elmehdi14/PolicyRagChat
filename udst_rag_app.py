import streamlit as st
from rag_pipeline import rag_pipeline, process_query
from dotenv import load_dotenv
import os
from mistralai.models import SDKError

# Load environment variables from .env file
load_dotenv()

# List of available policies
POLICIES = {
    "Sport and Wellness Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/sport-and-wellness-facilities-and",
    "Student Engagement Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/udst-policies-and-procedures/student-engagement-policy",
    "Student Council Procedure": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/student-council-procedure",
    "International Student Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/udst-policies-and-procedures/international-student-policy",
    "International Student Procedure": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/udst-policies-and-procedures/international-student-procedure",
    "Graduation Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/udst-policies-and-procedures/graduation-policy",
    "Student Counselling Services Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/udst-policies-and-procedures/student-counselling-services-policy",
    "Examination Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/examination-policy",
    "Academic Qualifications Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/academic-qualifications-policy",
    "Intellectual Property Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/intellectual-property-policy",

}

# Streamlit app
st.title("UDST Policy Chatbot")

# Policy selection (list box)
selected_policy = st.selectbox("Select a Policy", list(POLICIES.keys()))

# Load the RAG pipeline for the selected policy
url = POLICIES[selected_policy]
index, text_chunks, api_key = rag_pipeline(url)

# Query input 
query = st.text_input("Enter your query:")

# Button to process the query
if st.button("Submit"):
    if query.strip():  # Ensure the query is not empty
        # Process the query and retrieve the answer
        answer = process_query(query, index, text_chunks, api_key)
        # Display the answer in a text area
        st.text_area("Answer:", value=answer, height=200)
    else:
        st.warning("Please enter a query.")
