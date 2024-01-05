import streamlit as st
import numpy as np
import pandas as pd
import google.generativeai as genai
import textwrap
import PyPDF2
from PyPDF2 import PdfReader
from io import BytesIO
from dotenv import load_dotenv
import os
import re
import requests
import docx

# Function to download a PDF from a URL
def download_pdf(url):
    response = requests.get(url)
    if response.status_code == 200:
        return BytesIO(response.content)
    else:
        return None

def extract_text(file, file_type):
    if file_type == 'pdf':
        reader = PdfReader(file)
        text = "".join([page.extract_text() + "\n" for page in reader.pages])
    elif file_type == 'txt':
        text = file.getvalue().decode("utf-8")
    elif file_type == 'docx':
        doc = docx.Document(file)
        text = "\n".join([para.text for para in doc.paragraphs if para.text])
    else:
        raise ValueError(f"Unsupported file type: {file_type}")
    return text

def process_document(text):
    paragraphs = text.split('\n\n')
    paragraphs = [para.strip() for para in paragraphs if len(para.strip()) > 40]
    return {f"Section {i+1}": para for i, para in enumerate(paragraphs)}



API_KEY =st.secrets["GOOGLE_API_KEY"]
genai.configure(api_key=API_KEY)

def generate_embeddings(text, task_type="retrieval_document"):
    try:
        model = 'models/embedding-001'
        embedding = genai.embed_content(model=model, content=text, task_type=task_type)
        return embedding["embedding"]
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return np.zeros(768)

def find_best_passage(query_embedding, df):
    query_embedding_np = np.array(query_embedding)
    df_embeddings_np = df['Embeddings'].apply(np.array)
    if query_embedding_np.shape[0] != df_embeddings_np.iloc[0].shape[0]:
        raise ValueError("Mismatch in embedding sizes")
    dot_products = np.dot(np.stack(df_embeddings_np.to_numpy()), query_embedding_np)
    idx = np.argmax(dot_products)
    return df.iloc[idx]['Text']

def generate_conversational_answer(query, relevant_passage):
    model = genai.GenerativeModel('gemini-pro')
    prompt = make_prompt(query, relevant_passage)
    result = model.generate_content(prompt)
    if hasattr(result, 'text'):
        return result.text
    elif hasattr(result, 'parts'):
        return ''.join(part.text for part in result.parts if part.text)
    else:
        return "Sorry, I couldn't generate a response."

def make_prompt(query, relevant_passage):
    escaped = relevant_passage.replace("'", "").replace('"', "").replace("\n", " ")
    prompt = textwrap.dedent(f"""
        You are a helpful informative, and document reviewer bot that answers questions based on the provided document text.
        QUESTION: '{query}'
        DOCUMENT TEXT: '{escaped}'
        ANSWER:
    """)
    return prompt

def send_message(message):
    if message:
        st.session_state.chat_history.append(("You", message))
        query_embedding = generate_embeddings(message, task_type="retrieval_query")
        relevant_document = find_best_passage(query_embedding, df)
        response = generate_conversational_answer(message, relevant_document)
        st.session_state.chat_history.append(("Chatbot", response))
        st.experimental_rerun()

# Initialize Streamlit app
st.set_page_config(page_title="Document Q&A Bot")

# Sidebars
st.sidebar.header("Instructions")
st.sidebar.write("""
    Upload a document and ask questions about its content.
    Note: Larger documents may take longer for the model to process.
""")

st.sidebar.header("About App")
st.sidebar.write("""
    This app is powered by the 'Gemini-pro' model and its embeddings.
    Created by a data science enthusiast passionate about solving problems through AI.
""")

whatsapp_url=st.sidebar.header("Contact Me")
if whatsapp_url:
    st.sidebar.markdown("[ Gyimah Gideon](https://wa.link/yvcpjd)")
    

st.header("Document Q&A Bot")

uploaded_file = st.file_uploader("Upload a Document", type=["pdf", "txt", "docx"])

df = pd.DataFrame()

if uploaded_file:
    file_type = uploaded_file.name.split('.')[-1].lower()
    file = BytesIO(uploaded_file.getvalue()) if file_type in ['pdf', 'txt'] else uploaded_file
    text = extract_text(file, file_type)
    DOCUMENTS = process_document(text)
    df = pd.DataFrame(list(DOCUMENTS.items()), columns=['Title', 'Text'])
    df['Embeddings'] = df['Text'].apply(lambda x: generate_embeddings(x))
    st.success("Document processed successfully.")

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Chat UI
st.write("## Chat")
chat_container = st.container()
for author, message in st.session_state.chat_history:
    if author == "You":
        chat_container.markdown(f"<span style='color: blue;'>{author}:</span> {message}", unsafe_allow_html=True)
    else:  # "Chatbot"
        chat_container.markdown(f"<span style='color: green;'>{author}:</span> {message}", unsafe_allow_html=True)

with st.form("chat_form", clear_on_submit=True):
    input_message = st.text_input("Your message:", key="input")
    submit_button = st.form_submit_button(label='Send')

if submit_button and input_message:
    send_message(input_message)

st.write("----")
st.header("Feedback")

with st.form("feedback_form", clear_on_submit=True):
    feedback = st.text_area("Your feedback", key="feedback_text")
    submitted_feedback = st.form_submit_button(label="Submit Feedback")

feedback_file_path = "feedback.txt"

if submitted_feedback and feedback:
    with open(feedback_file_path, "a") as file:
        file.write(f"Feedback: {feedback}\n")
    st.write("Thank you for your feedback!")

st.write("Privacy Notice: Your data is handled with confidentiality.")
