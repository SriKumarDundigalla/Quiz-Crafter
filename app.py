from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory
from Prompt_examples import content1,content2,content3, content4
from werkzeug.utils import secure_filename
import networkx as nx
import matplotlib.pyplot as plt
from community import community_louvain
from itertools import cycle
from matplotlib.lines import Line2D
import subprocess
import os
import PyPDF2
import openai
import tiktoken
from dotenv import load_dotenv
from contextlib import contextmanager

import logging
import nbformat
import jsonpatch

import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.cluster import KMeans

from pinecone import Pinecone
from langchain_community.llms import OpenAI
from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma


import warnings
import shutil
import json
from langchain_text_splitters import TokenTextSplitter
from typing import List
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel

import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from plotly.offline import plot
import logging
import nbformat
import PyPDF2
import ast
import io
import chromadb

from astrapy.db import AstraDB
from langchain_text_splitters import CharacterTextSplitter
import plotly.graph_objects as go
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from werkzeug.utils import secure_filename
from flask import Flask, render_template, request, redirect, url_for, flash, session
from werkzeug.security import generate_password_hash, check_password_hash
import sqlite3
from docx import Document
import warnings
# Configure basic logging
logging.basicConfig(filename='App.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
vectordb = None 
load_dotenv()
app = Flask(__name__)
DATABASE = 'users.db'
persist_directory = 'docs/chroma/'
UPLOAD_FOLDER = os.path.join('..', 'static', 'Files')  # Base folder for all users' files
context_window_size =int(os.getenv('context_window_size'))
encoding = tiktoken.encoding_for_model("gpt-4")
api_key = os.getenv('OPENAI_API_KEY')
app.secret_key = os.getenv('SECRET_KEY', 'fallback_secret_key')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the base upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
def get_db():
    conn = sqlite3.connect(DATABASE)
    return conn

def init_db():
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

# Initialize the database on app start
init_db()

# @app.route('/')
# def index():
#     filenames = os.listdir(UPLOAD_FOLDER)
#     return render_template('index.html', filenames=filenames)

@app.route('/')
def home():
    return redirect(url_for('login'))

def create_logins_table():
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS logins (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            login_date DATE DEFAULT CURRENT_DATE,
            login_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    ''')
    conn.commit()
    conn.close()
create_logins_table() 

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
        user = cursor.fetchone()

        if user and check_password_hash(user[2], password):
            session['user_id'] = user[0]  # Store user ID in session
            flash('Login successful', 'success')

            # Log both login date and time
            cursor.execute("INSERT INTO logins (user_id) VALUES (?)", (user[0],))
            conn.commit()

            return redirect(url_for('main_page'))
        else:
            flash('Login failed. Check your credentials.', 'danger')

        conn.close()

    return render_template('login.html')
@app.route('/view_logins')
def view_logins():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT u.username, l.login_date, l.login_time 
        FROM logins l
        JOIN users u ON u.id = l.user_id
        ORDER BY l.login_time DESC
    ''')
    logins = cursor.fetchall()
    conn.close()

    return render_template('view_logins.html', logins=logins)

@app.route('/index')
def main_page():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    # Get the username
    user_id = session.get('user_id')
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT username FROM users WHERE id = ?", (user_id,))
    result = cursor.fetchone()
    conn.close()

    if result:
        username = result[0]
    else:
        flash("User not found.")
        return redirect(url_for('login'))

    user_folder = get_user_folder(username)
    filenames = os.listdir(user_folder)
    return render_template('index.html', filenames=filenames, username=username)


@app.route('/logout')
def logout():
    session.pop('user_id', None)  # Clear session
    flash('You have been logged out.', 'success')
    return redirect(url_for('login'))  # Redirect to login
def get_user_folder(username):
    """Generate a folder path for a user."""
    user_folder = os.path.join(app.config['UPLOAD_FOLDER'], f'{username}_files')
    if not os.path.exists(user_folder):
        os.makedirs(user_folder)
    return user_folder



@app.route('/upload', methods=['POST'])
def upload():
    # Get the logged-in user's user_id from the session
    user_id = session.get('user_id')
    if not user_id:
        flash("You must be logged in to upload files.")
        return redirect(url_for('login'))

    # Get the username based on the user_id
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT username FROM users WHERE id = ?", (user_id,))
    result = cursor.fetchone()
    conn.close()

    if result:
        username = result[0]
    else:
        flash("User not found.")
        return redirect(url_for('login'))

    # Ensure the user's folder exists
    user_folder = get_user_folder(username)

    if 'files' not in request.files:
        flash('No file part')
        return redirect(request.url)
    
    files = request.files.getlist('files')

    filenames = []
    for file in files:
        if file and file.filename != '':
            filename = secure_filename(file.filename)
            file.save(os.path.join(user_folder, filename))
            filenames.append(filename)

    flash('File(s) successfully uploaded')

    return render_template('index.html', filenames=filenames, username=username)


@app.route('/delete/<filename>', methods=['POST'])
def delete_file(filename):
    if 'user_id' not in session:
        return redirect(url_for('login'))

    # Get the username
    user_id = session.get('user_id')
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT username FROM users WHERE id = ?", (user_id,))
    result = cursor.fetchone()
    conn.close()

    if result:
        username = result[0]
    else:
        flash("User not found.")
        return redirect(url_for('login'))

    user_folder = get_user_folder(username)
    file_path = os.path.join(user_folder, filename)
    if os.path.exists(file_path):
        os.remove(file_path)
    return redirect(url_for('main_page'))


@app.route('/generate_outcomes', methods=['POST'])
def generate_outcomes():
    global vectordb
    if 'user_id' not in session:
        return redirect(url_for('login'))
    prompt_type = request.form.get('promptType')
    # Get the username
    user_id = session.get('user_id')
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT username FROM users WHERE id = ?", (user_id,))
    result = cursor.fetchone()
    conn.close()

    if result:
        username = result[0]
    else:
        flash("User not found.")
        return redirect(url_for('login'))

    user_folder = get_user_folder(username)
    filenames = os.listdir(user_folder)
    file_details = [{'path': os.path.join(user_folder, name)} for name in filenames]
    logging.info(file_details)
    file_contents = get_file_contents(file_details)
    logging.info(file_contents)
    # Proceed only if file_contents is not empty
    if not file_contents:
        return "No valid files to process.", 400
    vectordb = process_and_insert_contents(file_contents, persist_directory)
    summarized_contents = summarize_files(file_contents)
    chunked_contents = create_chunks_from_content_greedy(summarized_contents, context_window_size)
    list_of_learning_outcomes = generate_learning_outcomes_for_chunks(chunked_contents)
    filtered_los, fig = filter_learning_outcomes(list_of_learning_outcomes)
    questions = process_outcomes(list_of_learning_outcomes,prompt_type)
    logging.info(questions)
    # Convert the Plotly figure to HTML div
    graph_html = plot(fig, output_type='div', include_plotlyjs=True)
    return render_template('edit_learning_outcomes.html', learning_outcomes=filtered_los, graph=graph_html,outcome_list=questions)

def generate_outcomes():
    global vectordb
    filenames = os.listdir(UPLOAD_FOLDER)
    file_details = [{'path': os.path.join(UPLOAD_FOLDER, name)} for name in filenames]
    logging.info(file_details)
    file_contents=get_file_contents(file_details)
    vectordb = process_and_insert_contents(file_contents,persist_directory)
    summarized_contents = summarize_files(file_contents)
    chunked_contents = create_chunks_from_content_greedy(summarized_contents,context_window_size)
    list_of_learning_outcomes = generate_learning_outcomes_for_chunks(chunked_contents)
    filtered_los, fig =filter_learning_outcomes(list_of_learning_outcomes)
    # Convert the Plotly figure to HTML div
    graph_html = plot(fig, output_type='div', include_plotlyjs=True)

    return render_template('edit_learning_outcomes.html', learning_outcomes=filtered_los, graph=graph_html)

@app.route('/process_outcomes', methods=['POST'])
def process_outcomes(list_of_learning_outcomes,promptStyle):
    global vectordb
    # Retrieve the list of learning outcomes from the form data
    edited_outcomes = list_of_learning_outcomes
    prompt_style = promptStyle
    Quetions = format_learning_outcomes_with_identifiers(vectordb,edited_outcomes,prompt_style,5)
    # # Process each list of questions
    # for key in Quetions:
    #     Quetions[key] = process_questions(Quetions[key])
    #  # Adding index information directly into the structure for easier handling in the template
    # for category, questions in Quetions.items():
    #     for idx, question in enumerate(questions):
    #         question['index'] = idx
    #         question['options_with_indices'] = [{'index': opt_idx, 'option': option} 
    #                                             for opt_idx, option in enumerate(question['options'])]

    # Now Quetions contains structured questions, use or display them as needed
    return Quetions   

@app.route('/update_questions', methods=['POST'])
def update_questions():
    form_data = request.form.to_dict(flat=False)  # Get data as dictionary with values as list
    organized_data = {}

    # Step 1: Initialize categories with empty lists
    for key, values in form_data.items():
        if key.startswith('category_'):
            for value in values:
                if value not in organized_data:
                    organized_data[value] = []

    # Step 2: Process questions and assign them to categories
    for index in range(5):  # Assuming there are 5 sets of questions as per your example
        category_key = f'category_{index + 1}'  # Adjust this if category indexing is different
        category_name = form_data.get(category_key, ['Unknown Category'])[0]  # Default to 'Unknown Category'

        organized_data.setdefault(category_name, [])  # Ensure the category exists

        for i in range(5):  # Assuming each question set contains 5 questions
            question_key = f'question_{i}'
            options = [form_data[f'option_{i}_{j}'][index] for j in range(4)]
            answer = form_data[f'answer_{i}'][index]

            question_dict = {
                'question': form_data[question_key][index],
                'options': options,
                'answer': answer,
            }

            organized_data[category_name].append(question_dict)
    list_of_file_paths=[]
    a=1
    for key,value in organized_data.items():
        output =generate_QTI_Files(value,key)
        list_of_file_paths.append(save_output_to_file(output,f"Learning_outcome{a}"))
        list_of_file_paths = [path.replace('\\', '/') for path in list_of_file_paths]
        a+=1
    input_folder = './QTI_Text_Files'
  
    
    # Generate ZIP files from text files
    run_text2qti(input_folder)
    
    # List the generated ZIP files
    zip_files = [f for f in os.listdir(input_folder) if f.endswith('.zip')]

    return render_template('successful.html', files=zip_files, folder=input_folder)



@app.route('/successful')
def successful():
    # If you need to handle a direct visit to '/successful' without form submission, redirect back
    return redirect(url_for('main_page'))  # Redirect to the index page or another appropriate page

@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory('./QTI_Text_Files', filename, as_attachment=True)

def clean_content(content):
    """
    Performs cleaning of the file content, including trimming whitespace and removing non-printable characters.
    :param content: Raw content string to be cleaned.
    :return: Cleaned content string.
    """
    content = content.strip()  # Remove leading and trailing whitespace
    content = content.replace('\x00', '')  # Remove null bytes if present
    # Normalize line breaks and whitespace
    content = content.replace('\n', ' ')  # Replace new lines with spaces
    content = re.sub(r'\s+', ' ', content)  # Replace multiple spaces with a single space

    # Remove non-printable characters
    content = ''.join(char for char in content if char.isprintable() or char in ('\n', '\t', ' '))
    # Remove non-printable characters, including the replacement character
    content = re.sub(r'[^\x20-\x7E]+', '', content)
    return content

@contextmanager
def change_dir(destination):
    try:
        cwd = os.getcwd()  # store the current working directory
        os.chdir(destination)  # change to the destination directory
        yield
    finally:
        os.chdir(cwd)  # restore the original directory

def run_text2qti(input_folder):
    with change_dir(input_folder):
        # List all text files in the current folder
        text_files = [f for f in os.listdir('.') if f.endswith('.txt')]
        
        for text_file in text_files:
            try:
                # Run text2qti on each text file within the same directory
                subprocess.run(["text2qti", text_file], check=True)
                print(f"text2qti has successfully converted '{text_file}' to QTI format.")
            except subprocess.CalledProcessError as e:
                print(f"An error occurred while running text2qti: {e}")
            except FileNotFoundError:
                print("text2qti is not installed or not found in the system path.")

def read_file_content(file_path):
    """
    Reads the content of a file based on its extension and returns the cleaned content as a string.
    :param file_path: Path to the file.
    :return: Cleaned content of the file as a string.
    """
    file_name = os.path.basename(file_path)
    content = ''
    file_type = os.path.splitext(file_name)[1].lower()
    print(file_path)
    try:
        if file_type == '.pdf':
            # Handle PDF files
            with open(file_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    page_text = page.extract_text() if page.extract_text() else ''
                    content += clean_content(page_text)
        elif file_type == '.docx':
            # Handle DOCX files
            doc = Document(file_path)
            for para in doc.paragraphs:
                content += clean_content(para.text + '\n')
        elif file_type == '.ipynb':
            # Handle Jupyter Notebook files
            with open(file_path, "r", encoding='utf-8') as f:
                nb = nbformat.read(f, as_version=4)
                for cell in nb.cells:
                    if cell.cell_type in ('code', 'markdown'):
                        cell_content = cell['source'] + '\n\n'
                        content += clean_content(cell_content)
        else:
            # Handle other text files
            with open(file_path, "r", encoding='utf-8') as f:
                content = clean_content(f.read())

        logging.info(f"Successfully read and cleaned content from: {file_name}")
    except Exception as e:
        logging.exception(f"Error reading {file_name}: {e}")

    return content


def get_file_contents(file_details):
    """
    Retrieves the contents of each file based on the provided file details.

    :param file_details: List of dictionaries containing file details.
    :return: A list of dictionaries, each containing 'path' and 'content' of the file.
    """
    content_details = []
    for file_info in file_details:
        # Extract the file path from the dictionary
        file_path = file_info['path']
        file_content = read_file_content(file_path)
        if file_content:
            content_details.append({
                'path': file_path,
                'content': file_content
            })
    return content_details



def process_and_insert_contents(file_contents, persist_directory='./docs/chroma/'):
    """
    Processes the contents of each file, splits them, embeds, and inserts into a database.

    :param file_contents: List of dictionaries containing file paths and their contents.
    :param persist_directory: The directory to persist any necessary data for database insertion.
    """
    # Initialize the text splitter and embedding tools
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=150,
        length_function=len
    )
    embedding = OpenAIEmbeddings(api_key=api_key)
    all_documents = []  # Collect all documents for insertion

    for file in file_contents:
        content = file['content']  # Extract the content string
        if not content:
            continue  # Skip if content is empty

        # Split the content into documents
        documents = text_splitter.create_documents([content])
        all_documents.extend(documents)

        logging.info(f"Processed and inserted content from: {file['path']}")

    # Insert all documents into the vector database at once
    vectordb = Chroma.from_documents(
        documents=all_documents,
        embedding=embedding,
        persist_directory=persist_directory
    )

    return vectordb


def summarize_files(file_details):
    """
    Processes the content of files whose content exceeds a specified global token size,
    by splitting the content into chunks. Each chunk's size is determined to ensure it 
    doesn't exceed the global token size limit. The function returns a list of dictionaries 
    with the filename/path, chunked content, and the token size of each chunk.

    :param file_details: List of dictionaries with file details.
    :return: A list of dictionaries with content and token size.
    """
    global_token_size = int(os.getenv('GLOBAL_TOKEN_SIZE'))
    Overlap = 500  # Example overlap size, adjust as needed
    summarized_files = []

    for file in file_details:
        content = file['content']  # Extract the content string from the dictionary
        original_token_count = len(tiktoken.encoding_for_model("gpt-4").encode(content))

        if original_token_count > global_token_size:
            # Calculate the number of chunks needed
            N = 1 + (original_token_count - global_token_size) // (global_token_size - Overlap)
     
            # Initialize the splitter with calculated chunk size and overlap
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=original_token_count // N,
                chunk_overlap=Overlap,
                length_function=len,
                is_separator_regex=False
            )
            # Split the content into documents/chunks
            documents = splitter.create_documents([content])
            for document in documents:
                summarized_files.append({
                    'content': document.page_content,
                    'token_size': len(tiktoken.encoding_for_model("gpt-4").encode(document.page_content))
                })   
        else:
            # If the content does not exceed global token size, add it directly
            summarized_files.append({
                'content': content,
                'token_size': original_token_count
            })

    return summarized_files


def create_chunks_from_content_greedy(file_contents, context_window_size):
    """
    Creates content chunks from a list of file content dictionaries using a Greedy approach, 
    ensuring that each chunk does not exceed a specified context window size in terms of tokens.

    Parameters:
    - file_contents (list of dict): A list of dictionaries, where each dictionary contains 
      'content' (str) and 'token_size' (int) keys. 'content' is the text of the file, and 
      'token_size' is the number of tokens that text consists of.
    - context_window_size (int): The maximum number of tokens that a single chunk can contain. 
      It defines the upper limit for the size of each content chunk.

    Returns:
    - list of str: A list of content chunks, where each chunk is a string composed of file contents 
      that together do not exceed the context window size.
    """

    all_chunks = []  # Initialize the list to hold all content chunks
    current_chunk = ""  # Initialize the current chunk as an empty string
    current_token_count = 0  # Initialize the current token count to 0

    # Sort file_contents by 'token_size' in descending order
    sorted_file_contents = sorted(file_contents, key=lambda x: x['token_size'], reverse=True)

    for content in sorted_file_contents:
        # If adding this content exceeds the context window size, start a new chunk
        if current_token_count + content['token_size'] > context_window_size:
            if current_chunk:  # Ensure the current chunk is not empty
                all_chunks.append(current_chunk)  # Add the current chunk to all_chunks
                current_chunk = ""  # Reset the current chunk
                current_token_count = 0  # Reset the token count for the new chunk

        # Add the content to the current chunk if it fits
        if current_token_count + content['token_size'] <= context_window_size:
            current_chunk += content['content'] + "\n"  # Append content and a newline for readability
            current_token_count += content['token_size']
    
    # Add the last chunk if it contains any content
    if current_chunk:
        all_chunks.append(current_chunk)

    return all_chunks


def generate_learning_outcomes_for_chunks(documents):
    api_key = os.getenv('OPENAI_API_KEY')
    prompt_type = request.form.get('promptType')
    delimiter = "###"
    chunk_LOs = {}  # Dictionary to hold learning outcomes for each chunk

    # Initialize OpenAI client with your API key
    client = openai.OpenAI(api_key=api_key)


    # The number of outcomes to generate per chunk, adjust as needed or dynamically set
    number_of_outcomes = int(os.getenv('LOs_PER_CHUNK', 5))
    if prompt_type == 'zero_shot':
        logging.info("zero_shot")
        system_message = f"""
      As a Professor with expertise in curriculum development and crafting learning outcomes, your task is to extract and enumerate {number_of_outcomes} distinct learning outcomes from the provided course content. This content includes programming code, with each topic or code example distinctly separated by triple backticks ```. Your challenge is to recognize and interpret these segmented topics, especially those conveyed through programming code, to determine their thematic and practical contributions to the course. These outcomes should address the comprehensive skills and knowledge base essential to the subject matter, with a special emphasis on the interpretation and application of programming concepts as demonstrated within the code segments. The learning outcomes should be formatted as a Python list, precisely containing {number_of_outcomes} entries. Each entry must represent a unique learning outcome that students are expected to achieve by the end of the course, directly informed by the theoretical content and the practical programming code examples provided.
    """
    elif prompt_type == 'few_shot':
        logging.info("few_shot")
        system_message = f"""
        As a curriculum developer and professor, you are tasked with dissecting educational material encapsulated within triple backticks, spanning an extensive range of topics. This may include, but is not limited to, theoretical discussions, practical applications, programming examples, and any other form of academic content. Your primary objective is to distill and articulate {number_of_outcomes} specific learning outcomes from the presented content. These outcomes should comprehensively reflect the essential skills and knowledge students are expected to master by the conclusion of the course. You are to ensure a balanced emphasis on both the theoretical insights and practical competencies demonstrated through the material. 
        The extracted learning outcomes are to be strictly organized as a Python list, comprising exactly {number_of_outcomes} distinct items. Each item within this list should represent a unique learning goal directly derived from the course content, encapsulating both conceptual understanding and applicable skills.

        
        Do not venture beyond the content provided. Stick only to the content provided to you. Proceed to analyze the given content and structure your response in the as python list as follows example output response structure:
            Example_input1: Generate Lerning outcomes for the following content enclosed by triple hashtag{delimiter}{content1}{delimiter}
            Expected_output1 :  [
                "Understand and Apply Various Data Mining Techniques: Learners will be able to describe and apply different data mining techniques such as association rules, classification, clustering, decision trees, K-Nearest Neighbor (KNN), neural networks, and predictive analysis. They will understand how these techniques are used to extract valuable insights from large datasets, including market basket analysis, object classification, grouping based on similarities, and forecasting future outcomes.",
               "Comprehend the Data Mining Process: Learners will gain an understanding of the structured process involved in data mining, including the importance of understanding the business context, data preparation, model building, result evaluation, and implementing changes based on findings. This outcome ensures that learners can navigate through the complexities of data analysis projects systematically and efficiently.",
               "Execute Predictive Modeling using Linear Regression: Learners will develop the skills to implement predictive models using linear regression, including data preparation, model fitting, and performance analysis. They will be proficient in using Python libraries such as matplotlib, numpy, pandas, and sklearn for data manipulation and visualization, fitting a linear regression model, and evaluating its performance through metrics like the coefficient of determination (R^2).",
                "Data Preparation and Visualization: Learners will be skilled in preparing data for analysis, including cleaning, standardizing, and checking for outliers. They will also learn how to visualize data and regression model outcomes using scatter plots to identify trends and patterns. This outcome emphasizes the importance of visual data exploration as a preliminary step to modeling.",
                "Critical Evaluation and Application of Data Mining Findings: Upon completing their learning, individuals will be capable of critically evaluating the results of data mining and predictive modeling efforts. They will understand how to make informed decisions based on the analysis, implement changes in the business strategy, and monitor the impact of these changes. This learning outcome bridges the gap between technical analysis and practical business applications, preparing learners to contribute strategically to business growth and innovation."
                ]

            Example_input2 : Generate Lerning outcomes for the following content enclosed by triple hashtag{delimiter}{content2}{delimiter}
            Expected_output2 : [
                    "Understand Database Fundamentals and MongoDB's Place in the Ecosystem: Learners will grasp the concept of databases, differentiating between relational (SQL) and non-relational (NoSQL) databases. They will understand MongoDB's role as a powerful, flexible NoSQL document database that supports diverse data storage needs with scalability, performance, and high availability.",
                    "Develop Skills in Performing CRUD Operations with MongoDB using Python: Participants will acquire the ability to connect to a MongoDB database using Python's `pymongo` library, create databases and collections, and perform CRUD (Create, Read, Update, Delete) operations to manage documents within MongoDB collections.",
                    "Master Advanced MongoDB Operations and Database Management: Learners will delve into advanced MongoDB functionalities such as aggregation, indexing, and query optimization. They will learn how to use these operations to enhance query performance and ensure efficient data retrieval, forming a solid foundation for developing data-driven applications.",
                    "Implement Security and Efficient Data Handling Techniques: Students will learn secure methods of handling sensitive information by reading authentication details from external files, thus enhancing security and simplifying credential management. They will also understand the importance of indexing and the capabilities of MongoDB in supporting large-scale, efficient data management and retrieval.",
                    "Acquire Comprehensive Knowledge on MongoDB's Features, Advantages, and Usage in Industry: Learners will explore the features, advantages, and practical applications of MongoDB, understanding its schema-less nature, document orientation, high performance, scalability, and support for various data types. They will also recognize MongoDB's significance in the tech industry, supported by examples of major companies utilizing MongoDB for their data storage needs."
                ]

            Example_input3 : Generate Lerning outcomes for the following content enclosed by triple hashtag{delimiter}{content3}{delimiter}
            Expected_output3 :
                            [
                    "Understand the Process of Setting Up a Connection to DataStax Astra with Cassandra: Learners will understand how to use Python to establish a secure connection to a Cassandra database hosted on DataStax Astra, including the steps for loading authentication details from a JSON file and utilizing a secure connection bundle.",
                    "Gain Knowledge on Cassandra Database Schema Creation: Learners will gain the ability to create and manipulate database schemas within Cassandra, specifically learning how to set keyspaces and create tables with various data types as primary keys, which is crucial for database design and management.",
                    "Learn to Insert and Query Data in Cassandra: Learners will acquire skills in inserting sample data into tables and querying data from those tables using the Cassandra Query Language (CQL), emphasizing the importance of understanding how to manage and retrieve data efficiently.",
                    "Develop an Understanding of Parameterized Queries for Security and Efficiency: Learners will understand the significance of using parameterized queries to prevent SQL injection attacks and ensure data type accuracy, which promotes writing secure and maintainable code.",
                    "Master the Initial Setup for Using Cassandra with DataStax Astra: Learners will be able to navigate the initial setup process for using Cassandra in a cloud environment, including creating an account on DataStax Astra, setting up a serverless database, and understanding the tools and environments (like Anaconda and VS Code) needed to connect to and work with Cassandra databases."
                ]
                
            Example_input4 :Generate Lerning outcomes for the following content enclosed by triple hashtag{delimiter}{content4}{delimiter}
            Expected_output4 :
                            [
                    "Understanding the Fundamentals of AI and its Branches: Learners will grasp the basic concepts and distinctions between Artificial Intelligence (AI), Machine Learning (ML), Deep Learning, and Generative AI, and how these technologies are interconnected.",
                    "Recognizing AI's Impact Across Various Fields: Participants will identify potential areas where AI is expected to exceed human performance in the near future, such as language translation, creative writing, driving, and even complex tasks like surgery, demonstrating AI's growing influence in diverse sectors.",
                    "Exploring the Role of Adversarial AI in Cybersecurity: Students will gain insights into how adversarial AI operates by creating inputs designed to fool AI systems, understanding its significance in cybersecurity through examples like manipulating images to bypass classifiers, and discussing its implications for malware detection and network security.",
                    "Developing Machine Learning Models with Practical Exercises: Through hands-on exercises, learners will experience the process of training a machine learning model, including data preprocessing, feature extraction, model training with perceptrons, and evaluating accuracy, using Python libraries such as NumPy, pandas, and PyTorch.",
                    "Ethical Considerations and Future Directions of AI: Participants will discuss the ethical aspects of AI, including the challenges of AI brittleness, opacity, and the lack of commonsense reasoning, as well as future advancements like text-to-image generation, text-to-voice conversion, and the potential for AI in generating real-world objects through 3-D printing."
                ]


"""
    elif prompt_type == 'cot':
        logging.info("Chain_Of_Thought")
        system_message = f"""
\"\"\"
As a curriculum developer and professor, you are tasked with a specific challenge. The educational material, spanning a broad range of topics, is provided to you encapsulated within triple backticks. This material might include theoretical discussions, practical applications, programming examples, and more. Your primary objective is to extract and articulate exactly {number_of_outcomes} learning outcomes from the content provided. These outcomes should be comprehensive, reflecting the skills and knowledge students are expected to master by the end of the course. It's crucial to balance theoretical insights with practical competencies demonstrated through the material.

Given this role and task, let's breakdown the approach step-by-step to ensure clarity and precision in the outcomes extracted:

1. **Content Review**: Start by meticulously reviewing each piece of educational material provided within triple backticks. Whether it's a piece of code or a theoretical discussion, understanding each element in-depth is key.

2. **Outcome Identification**: From each piece of content, identify the key learning outcomes. These should be the most critical skills or knowledge that a student needs to acquire. The focus should be on outcomes that are directly related to the material reviewed, ensuring relevance and direct derivation from the content.

3. **Outcome Selection and Prioritization**: Out of the potential outcomes identified, prioritize and select the {number_of_outcomes} most important ones. These selected outcomes should span the spectrum of the material, ensuring a diverse and comprehensive reflection of the course content.

4. **List Organization**: Organize these {number_of_outcomes} outcomes into a Python list. Each item in this list should be a distinct learning goal, articulately representing a blend of conceptual understanding and applicable skills derived directly from the course content.

5. **Adherence to Content**: Throughout this process, it's vital to confine your analysis to the material provided within the triple backticks. Extrapolation beyond the provided content should be avoided to maintain accuracy and relevance. Make sure The Learning Outcomes are covering all the concepts in content given from each section.

An example structure for organizing the learning outcomes in a Python list could look like this:

learning_outcomes = [
    "Outcome 1: Understanding the fundamental concepts of [Topic A], demonstrating both theoretical insights and practical applications.",
    "Outcome 2: Developing proficiency in [Skill B] through guided practice and real-world applications found in [Topic B].",
    ...
    "Outcome {number_of_outcomes}: Achieving a comprehensive understanding and application of [Concepts from Topic X] combined with [Skills from Topic Y]."
]
\"\"\"
\"\"\"This structured approach ensures that as a curriculum developer and professor, you systematically dissect and translate the educational material into tangible learning outcomes, aligning with the course's objectives.\"\"\"
"""

    
    
    all_out_comes=[]
    
    # Generate learning outcomes for each chunk
    for index, chunk in enumerate(documents, start=1):
        delimiter = "```"
        user_message = f"""
                        \"\"\"
                        As a curriculum developer and professor, I am tasked with creating learning outcomes based on the provided educational material. Below is a segment of educational content enclosed within triple backticks, which spans a wide range of topics potentially including theoretical discussions, practical applications, programming examples, and other forms of academic content. My goal is to distill this material into specific, actionable learning outcomes that reflect the essential skills and knowledge students are expected to master. 

                        Please generate learning outcomes for the following content:

                        {delimiter}{chunk}{delimiter}

                        The learning outcomes should be comprehensive, covering both conceptual understanding and applicable skills directly derived from the course content. Each outcome should be unique and relevant to the provided material.
                        \"\"\"
                        """

        response = client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {"role": "system", "content": system_message.strip()},
                {"role": "user", "content": user_message.strip()}
            ],
            temperature=0
        )
        
        summary = response.choices[0].message.content
        start = summary.find("[")
        end = summary.rfind("]") + 1
        outcome_list = eval(summary[start:end])
        logging.info(outcome_list)

        all_out_comes.append(outcome_list)

    # Flatten each list of outcomes into a single string per list to simplify the example
    documents = [item for outcome_list in all_out_comes for item in outcome_list]

    return documents

def create_cluster_visualization(clusters):
    G = nx.Graph()
    pos = {}
    labels = {}
    full_texts = {}
    for cluster_id, texts in clusters.items():
        for text in texts:
            node_name = f"{cluster_id}: {text[:30]}..."
            G.add_node(node_name)
            pos[node_name] = (hash(text) % 100, hash(cluster_id) % 100)
            labels[node_name] = node_name
            full_texts[node_name] = text

    for cluster_id, texts in clusters.items():
        texts_iter = iter(texts)
        first_text = next(texts_iter, None)
        if first_text:
            for text in texts_iter:
                G.add_edge(f"{cluster_id}: {first_text[:30]}...", f"{cluster_id}: {text[:30]}...")

    pos = nx.spring_layout(G, seed=42)
    edge_x, edge_y = [], []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=0.5, color='#888'), hoverinfo='none', mode='lines')
    node_x, node_y, node_hover_texts = [], [], []
    for node, (x, y) in pos.items():
        node_x.append(x)
        node_y.append(y)
        node_hover_texts.append(full_texts[node])

    node_trace = go.Scatter(x=node_x, y=node_y, mode='markers', hoverinfo='text', hovertext=node_hover_texts,
                            marker=dict(showscale=True, colorscale='Viridis', size=20, color=list(range(len(G.nodes()))),
                                        colorbar=dict(thickness=15, title='Node Connections', xanchor='left', titleside='right'), line_width=2))
    fig = go.Figure(data=[edge_trace, node_trace], layout=go.Layout(title='Clusters of Learning Outcomes', titlefont_size=16, showlegend=False, hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40), xaxis=dict(showgrid=False, zeroline=False, showticklabels=False), yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        paper_bgcolor='black', plot_bgcolor='black'))
    return fig


def filter_learning_outcomes(documents,num_clusters=5):
    api_key = os.getenv('OPENAI_API_KEY')
    client = openai.OpenAI(api_key=api_key)
    # Generate embeddings for each document using OpenAI's API
    embeddings = []
    for doc in documents:
        response = client.embeddings.create(
                input=doc,
                model="text-embedding-3-large"  # You can choose different models based on your specific needs
            )

        embeddings.append(response.data[0].embedding)

    # Apply k-means clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(embeddings)

    # Get the cluster labels
    labels = kmeans.labels_

    # Creating a dictionary to store clusters and their learning outcomes
    clusters = {}
    for i in range(num_clusters):
        clusters[f"Cluster {i+1}"] = []

    # Assigning learning outcomes to their respective clusters
    for idx, label in enumerate(labels):
        cluster_name = f"Cluster {label + 1}"
        clusters[cluster_name].append(documents[idx])
    logging.info(clusters)
    # Create and display the graph
    fig = create_cluster_visualization(clusters)

    system_message = f"""
###As a professor specializing in the creation of learning outcomes, you are tasked with analyzing a dictionary where each key corresponds to a cluster of topics, and each value lists detailed learning outcomes for that cluster. 

###Your objectives are:

1.Identify the overarching theme or common objective within each cluster's learning outcomes.
2.Summarize the diverse topics within each cluster into a cohesive and informative statement that captures the essence of the educational goals, ensuring that even if topics differ, they are represented under the unified theme of the cluster.
3.Craft each summary to be distinct and specifically tailored to its cluster, steering clear of generic descriptions. Give me in one complete sentence.
4.Compile these summaries into a Python list, with each entry corresponding to a particular cluster, ensuring the number of elements in the list matches the number of clusters in the input dictionary.
5.Dont add anything new to summary make sure everting is from provided cluster list and basic concepts must be included in the Outccome. From give cluster list cover all essential concepts.

###Output:

Return a Python list containing the summarized learning outcomes. Each element of the list should correspond to a cluster, tailored to reflect the comprehensive educational goals of that cluster, inclusive of all topics, regardless of their diversity within the same cluster. Dont repeate word cluster at starting of learning outcomes.
"""
    user_message = f"""
                        \"\"\"
                            Given an input dictionary of groups of learning outcomes, please summarize the list of learning outcomes for each cluster. The desired output is a Python list that covers all topics provided for each cluster's learning outcomes. Use the enclosed cluster dictionary provided between triple backticks:
```
{clusters} ```

Ensure that each element in the Python list reflects a comprehensive summary of the learning outcomes for each cluster, incorporating all associated topics
                        \"\"\"
                        """
    response = client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {"role": "system", "content": system_message.strip()},
                {"role": "user", "content": user_message.strip()}
            ],
            temperature=0
        )
        
    summary = response.choices[0].message.content
    logging.info(summary)
    start = summary.find("[")
    end = summary.rfind("]") + 1
    outcome_list = eval(summary[start:end])


    return [outcome_list,fig]

def process_questions(questions):
    structured_questions = []
    for q in questions:
        parts = q.split("#")
        question_text = parts[0].strip("*")
        options = parts[1:-1]
        answer = parts[-1].split("**Answer: ")[1].strip("**")

        question_data = {
            "question": question_text,
            "options": options,
            "answer": answer
        }
        structured_questions.append(question_data)
    
    return structured_questions

def find_most_relevant_learning_outcome_document(vectordb, learning_outcomes,number_of_docs=5):
    """
    Uses vectordb to find the most relevant learning outcome document from the database for each topic.

    :param vectordb: The vectordb instance configured for retrieval.
    :param learning_outcomes: A list of lists, where each sublist represents learning outcomes related to a specific topic.
    :return: A list of tuples, each containing the most relevant document's content and its relevance score for each list of learning outcomes.
    """
    # Initialize the vectordb retriever with 'k' set to 1 to retrieve the most relevant document
    retriever = vectordb.as_retriever(search_type="mmr", search_kwargs={"k": number_of_docs})

    documents =[]
    docs = retriever.get_relevant_documents(learning_outcomes)
    for i in range(0,len(docs)):
        documents.append("```"+docs[i].page_content+"```")
    return documents



def format_learning_outcomes_with_identifiers(vectordb,learning_outcomes,prompt_style,number_of_docs=5,):
    api_key = os.getenv('OPENAI_API_KEY')
    client = openai.OpenAI(api_key=api_key)
    num_of_quetions = os.getenv('NUMBER_OF_QUETIONS')
    Quetion_Learning_outcomes=dict()
    logging.info(prompt_style)
    for i in learning_outcomes:
        docs = find_most_relevant_learning_outcome_document(vectordb,i,number_of_docs)
        if prompt_style=='cot' :
            system_message=f"""
    **Task:** You are a professor tasked with developing a concise series of multiple-choice quiz questions for students. Each question must be aligned with a distinct learning outcome and directly related to specific content sections . This content may include programming code relevant to the learning outcomes. The correct answer to each question should be unequivocally found within the provided content, which is enclosed in triple backticks for each section. Generate {num_of_quetions} multiple-choice questions from each section of content, ensuring each question can be answered based on the given content and relates closely to the learning outcomes.

    **Follow these tasks to generate multiple-choice questions:**
    1. **Analyze the Learning Outcome and Content:** For each provided learning outcome, identify the core concept that needs to be assessed. Review the content and programming code enclosed within triple backticks to understand how it supports the learning outcome.
    2. **Content and Code Review:** Thoroughly examine the textual content and programming code. Extract key facts, figures, themes, and functional aspects of the code related to the learning outcome.
    3. **Question Development:** Based on the comprehensive review, craft questions that test the student's understanding of specific facts, themes, or the functionality demonstrated by the code. The Quetions hsould be on concepts covered in content.
    4. **Formulating Options:** Create multiple-choice options where one is the correct answer (directly taken from the content or inferred from understanding the code) and the others are plausible but incorrect, closely related to the content to ensure a comprehensive understanding.
    5. **Avoid Repetition:** Ensure that each question is unique and there is no repetition across the sections.
    6. **Verification:** Confirm that the output quiz meets the guidelines and each question links clearly back to the provided content and its associated learning outcome. Generate Only {num_of_quetions} multiple choice Questions.

    **Output Requirements:**
    Each question should follow this format and reflect both core and supplementary themes:

    [
        "**Question1. [Question here]?**#A) [Option A]#B) [Option B]#C) [Option C]#D) [Option D]#**Answer: [Correct full Option including text]**",
        ... (Repeat for all questions)
    ]

    Proceed with this format for all questions, ensuring each is answerable based on the provided content, including understanding of the programming code. This comprehensive approach guarantees a focused, educational, and thematic quiz that effectively assesses students' understanding and engagement with the material.
    """
            user_message = f"""Please generate multiple-choice questions from the provided content associated with specific learning outcomes. learning outcome is enclosed within triple hashtags, denoted as `###{i}###`, represents the learning outcome. The content relevant to each learning outcome is enclosed within triple backticks, ` ```{docs}``` `,  represents the content related to that learning outcome. Ensure that each multiple-choice question is directly derived from the relevant content and clearly aligns with its respective learning outcome."""
            response = client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[
                    {"role": "system", "content": system_message.strip()},
                    {"role": "user", "content": user_message.strip()}
                ],
                temperature=0
            )   
            summary = response.choices[0].message.content

        elif prompt_style=='zero_shot':
            system_message=f"""
                    You are a professor tasked with developing a concise series of multiple-choice quiz \
                    questions for your students, each aligned with a distinct learning outcome and \
                    directly related to specific content. Your task is to ensure that each question not \
                    only integrates with the learning material but also has its correct answer \
                    unequivocally found within the provided content. The user will provide \
                    you with content enclosed in triple backticks for each section related \
                    to a given learning outcome. Generate {num_of_quetions} multiple-choice questions \
                    from each section of content divided by ```. Ensure the question can be answered\
                    from the given content. The remaining options in multiple-choice should be \
                    closely related to the content. Output must only contain list with single line string as shown in example for each question, no other explanation text strictly.

                    **Output Structure of python list Expected:**
                   
                        [
                            "**Question 1. [Insert your first question here]?**\n#A) [Insert Option A]\n#B) [Insert Option B]\n#C) [Insert Option C]\n#D) [Insert Option D]\n**Answer: [Correct Answer Text: Option X - Full Answer Description]**",
                            "**Question 2. [Insert your second question here]?**\n#A) [Insert Option A]\n#B) [Insert Option B]\n#C) [Insert Option C]\n#D) [Insert Option D]\n**Answer: [Correct Answer Text: Option Y - Full Answer Description]**",  
                            "**Question 3. [Insert your third question here]?**\n#A) [Insert Option A]\n#B) [Insert Option B]\n#C) [Insert Option C]\n#D) [Insert Option D]\n**Answer: [Correct Answer Text: Option Z - Full Answer Description]**",
                            "**Question 4. [Insert your fourth question here]?**\n#A) [Insert Option A]\n#B) [Insert Option B]\n#C) [Insert Option C]\n#D) [Insert Option D]\n**Answer: [Correct Answer Text: Option W - Full Answer Description]**",
                            "**Question 5. [Insert your fifth question here]?**\n#A) [Insert Option A]\n#B) [Insert Option B]\n#C) [Insert Option C]\n#D) [Insert Option D]\n**Answer: [Correct Answer Text: Option V - Full Answer Description]**"
                        ]

                Proceed with this format for all questions, ensuring they are answerable based on the provided content. This comprehensive approach ensures a focused, educational, and thematic quiz that effectively assesses students' understanding and engagement with the material.
                """
            user_message = f"""Please generate multiple-choice questions from the given content and learning outcome. The learning outcome is enclosed within triple hashtags ###{i}###, and the content relevant to the learning outcome is enclosed within triple backticks ```{docs}```"""
            response = client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[
                    {"role": "system", "content": system_message.strip()},
                    {"role": "user", "content": user_message.strip()}
                ],
                temperature=0
            )
            
            summary = response.choices[0].message.content
            logging.info(summary)

        try:
            # Regular expression pattern to match each question block
            pattern = r"\*\*Question\s\d+.*?\*\*Answer:.*?(?=\*\*Question\s\d+|\]$)"
            matches = re.findall(pattern, summary, re.DOTALL)

            outcome_list = []
            for match in matches:
                # Extract question and answer separately
                question_match = re.search(r"(\*\*Question\s\d+.*?)(?=\*\*Answer:)", match, re.DOTALL)
                answer_match = re.search(r"\*\*Answer:(.*)", match, re.DOTALL)

                if question_match and answer_match:
                    question_with_options = question_match.group(1).strip()
                    answer = answer_match.group(1).strip()
                    
                    # Extract question text
                    question_text_match = re.search(r'\*\*Question\s\d+\.\s*(.*?)\*\*', question_with_options, re.DOTALL)
                    if question_text_match:
                        question_text = question_text_match.group(1).strip()
                        # Now get the options string
                        options_string = question_with_options[question_text_match.end():].strip()
                        # Parse the options
                        options = re.findall(r'#([A-Z])\)\s*(.*?)(?=\s*#|\s*\Z)', options_string, re.DOTALL)
                        options_dict = {option_letter: option_text.strip() for option_letter, option_text in options}
                    else:
                        logging.warning(f"Could not parse question text in: {question_with_options}")
                        question_text = question_with_options
                        options_dict = {}
                    
                    # Clean the answer
                    answer = answer.strip('**",\n"').strip()
                    # Extract correct option
                    answer_option_match = re.match(r'([A-D])\s*[-–]\s*(.*)', answer)
                    if answer_option_match:
                        correct_option = answer_option_match.group(1)
                        answer_text = answer_option_match.group(2).strip()
                    else:
                        correct_option = None
                        answer_text = answer
                    
                    outcome_list.append({
                        'question': question_text,
                        'options': options_dict,
                        'correct_option': correct_option,
                        'answer_text': answer_text
                    })
                else:
                    logging.warning(f"Could not parse question or answer in: {match}")

            # Assuming Quetion_Learning_outcomes is a list or dictionary
            Quetion_Learning_outcomes[i] = outcome_list
            logging.info(Quetion_Learning_outcomes)

        except Exception as e:
            logging.error(f"Error processing the summary: {e}")
            Quetion_Learning_outcomes[i]=summary

    
    return Quetion_Learning_outcomes

def generate_QTI_Files(Quetions_object,lo):
    api_key = os.getenv('OPENAI_API_KEY')
    client = openai.OpenAI(api_key=api_key)
    system_message="""
#Objective: Convert a JSON object into a text2qti formatted text string. The JSON object contains a key representing a learning outcome and a list of questions with options and correct answers related to this outcome.

#Instructions:
1. Extract the learning outcome from the JSON key to use as the quiz title.provide a shorter and concise title from learning outcomes fro quiz title.
2. For each question in the list:
   - Number the questions sequentially starting from 1.
   - List the question followed by feedback indications for correct and incorrect answers.
   - Identify and mark the correct answer with an asterisk (*) before the option letter. This is must be followed
   - List all four options for the question as provided in the JSON.
   - Provide specific feedback for each option according to its correctness:
     a) If the option is the correct answer, follow it with "Correct answer feedback.". Provide your feedback after three dots (...) below each option. so where there is (... Correct answer feedback.) you need to replace that wit your feedback with correct answer feedback.
     b) If the option is incorrect, follow it with "Incorrect answer feedback.".Provide your feedback after three dots (...) below each option. so where there is (... Incorrect answer feedback.) you need to replace that wit your feedback on Incorrect answer feedback.
3. Make sure in respose has no other inputs or notes as this really important task to follow structure. 
Important - ###Take time and Make sure that there is  an asterisk (*) before the option letter of correct answer for all quetions.

#Example JSON Input:
{
  "Mastering data visualization techniques through the creation of scatter plots with matplotlib, including the use of custom colors and sizes for data points based on conditional logic.": [
    {
      "question": "Question1. What determines the color of the data points in the scatter plot?",
      "options": ["A) The size of the data point", "B) The x value of the data point", "C) The y value of the data point", "D) The label of the data point"],
      "answer": "C) The y value of the data point"
    },
    {
      "question": "Question2. What is the shape of the markers used in the scatter plot?",
      "options": ["A) Square", "B) Triangle", "C) Circle", "D) Diamond"],
      "answer": "C) Circle"
    },
    {
      "question": "Question3. Which line is added to the scatter plot as a reference?",
      "options": ["A) y = 50", "B) y = 2x + 3", "C) x = 50", "D) Both A and B"],
      "answer": "D) Both A and B"
    },
    {
      "question": "Question4. How are the sizes of the data points in the scatter plot determined?",
      "options": ["A) Based on their x values", "B) Randomly", "C) Based on their y values", "D) All data points have the same size"],
      "answer": "B) Randomly"
    },
    {
      "question": "Question5. What does the linestyle '--' signify in the plot?",
      "options": ["A) A dotted line", "B) A dashed line", "C) A solid line", "D) A dash-dot line"],
      "answer": "B) A dashed line"
    }
  ]
}


#Look at this Example of expected Response style:
    #Example 1:

    Quiz title: Acquire skills in annotating and customizing plot features

    1. What color are the scatter plot points that have a y-value greater than 50?
    a) Green
    ... Incorrect answer feedback: Green is not the color used to indicate points with a y-value greater than 50.
    b) Blue
    ... Incorrect answer feedback: Blue does not represent points with a y-value greater than 50 in this context.
    *c) Red
    ... Correct answer feedback: Red is used to highlight points with a y-value greater than 50, making them stand out.
    d) Yellow
    ... Incorrect answer feedback: Yellow is not the color chosen for points with a y-value greater than 50.

    2. What is the shape of the markers used in the scatter plot?
    a) Squares
    ... Incorrect answer feedback: Squares are not the shape chosen for markers in this scatter plot.
    b) Triangles
    ... Incorrect answer feedback: Triangles, while a valid shape, are not used for markers in this instance.
    *c) Circles
    ... Correct answer feedback: Circles are the shape used for markers, providing a clear and distinct representation of data points.
    d) Diamonds
    ... Incorrect answer feedback: Diamonds are not the shape used for markers in this scatter plot.

    3. Which of the following best describes the linestyle of the line representing y = 2x + 3?
    a) Dotted
    ... Incorrect answer feedback: A dotted line does not represent the line y = 2x + 3 in this plot.
    b) Solid
    ... Incorrect answer feedback: The line y = 2x + 3 is not depicted with a solid linestyle in this context.
    *c) Dashed
    ... Correct answer feedback: A dashed line accurately represents y = 2x + 3, helping to distinguish it from other plot elements.
    d) Dash-dot
    ... Incorrect answer feedback: The line y = 2x + 3 is not represented with a dash-dot linestyle.

    4. What does the horizontal line at y = 50 represent in the plot?
    a) The average value of y
    ... Incorrect answer feedback: The horizontal line at y = 50 does not represent the average value of y.
    *b) A threshold value
    ... Correct answer feedback: The horizontal line at y = 50 represents a threshold value, marking a specific level of interest within the plot.
    c) The maximum value of y
    ... Incorrect answer feedback: The line at y = 50 does not indicate the maximum value of y.
    d) The minimum value of y
    ... Incorrect answer feedback: The line at y = 50 is not used to denote the minimum value of y.

    5. How is the size of the scatter plot points determined?
    a) Based on the x-value
    ... Incorrect answer feedback: The size of scatter plot points is not determined by their x-value.
    *b) Randomly
    ... Correct answer feedback: The size of scatter plot points is determined randomly, adding variety and visual interest to the plot.
    c) Based on the y-value
    ... Incorrect answer feedback: The y-value does not determine the size of scatter plot points in this case.
    d) All points are the same size
    ... Incorrect answer feedback: Not all points are the same size; their size varies according to a specific criterion, which in this case is random.

    #Example 2: 

    Quiz title: Master Scatter Plot Customization

    1. What determines the color of the points in the scatter plot?
    a) The size of the point
    ... Incorrect answer feedback: The size of the point does not determine its color in a scatter plot.
    b) The x value of the point
    ... Incorrect answer feedback: The x value of the point is not typically used to determine color in scatter plots.
    *c) The y value of the point
    ... Correct answer feedback: The y value of the point often determines its color to visualize data distributions effectively.
    d) The label of the plot
    ... Incorrect answer feedback: The label of the plot does not influence the color of its points.

    2. What is the shape of the markers used in the scatter plot?
    a) Square
    ... Incorrect answer feedback: Squares are not the default shape for markers in scatter plots.
    b) Triangle
    ... Incorrect answer feedback: Triangles, while possible, are not the shape in question here.
    *c) Circle
    ... Correct answer feedback: Circles are commonly used as the shape for markers in scatter plots for clear visualization.
    d) Diamond
    ... Incorrect answer feedback: Diamonds are not the shape typically used for markers in this context.

    3. Which line is added to the plot to indicate a specific y value?
    a) y = 2x + 3
    ... Incorrect answer feedback: This line represents a mathematical relationship, not a specific y value.
    *b) y = 50
    ... Correct answer feedback: The line y = 50 is added to indicate a specific y value within the scatter plot.
    c) x = 50
    ... Incorrect answer feedback: The line x = 50 would indicate a specific x value, not a y value.
    d) x = 2y + 3
    ... Incorrect answer feedback: This line represents a mathematical relationship, not a specific y value.

    4. How are the sizes of the points in the scatter plot determined?
    a) Based on their x values
    ... Incorrect answer feedback: The x values of points are not typically used to determine their sizes in scatter plots.
    b) Based on their y values
    ... Incorrect answer feedback: While possible, this is not the method used for determining point sizes in this context.
    *c) Randomly assigned
    ... Correct answer feedback: Randomly assigning sizes to points can help in visualizing data distributions in a more dynamic and interesting way.
    d) All points are the same size
    ... Incorrect answer feedback: Having all points the same size would not allow for the dynamic visualization of data distributions.

    5. What does the line 'y = 2x + 3' represent in the context of the plots?
    a) A boundary separating different colors of points
    ... Incorrect answer feedback: The line 'y = 2x + 3' does not serve as a boundary for color separation in scatter plots.
    *b) A trend line for the scatter plot data
    ... Correct answer feedback: The line 'y = 2x + 3' represents a trend line, helping to visualize the relationship between data points in the scatter plot.
    c) An arbitrary line for aesthetic purposes
    ... Incorrect answer feedback: The line is not arbitrary; it has a specific purpose in the context of data visualization.
    d) A reference line to highlight specific data points
    ... Incorrect answer feedback: While it could highlight specific data points, its primary role is to represent a trend within the scatter plot data.
    # Example3

        Quiz title: Enhance Plot Annotations and Legends

        1. What color is used to indicate values of y greater than 50 in the scatter plot?
        a) Green
        ... Incorrect answer feedback: Green does not specifically indicate values of y greater than 50 in scatter plots.
        b) Blue
        ... Incorrect answer feedback: Blue is not the color used for highlighting y values greater than 50.
        *c) Red
        ... Correct answer feedback: Red is used to distinctly indicate values of y that are greater than 50, making them stand out visually.
        d) Yellow
        ... Incorrect answer feedback: Yellow is not designated for values of y greater than 50 in these visualizations.

        2. What linestyle is used for the line at y = 50?
        a) Solid
        ... Incorrect answer feedback: A solid line is not the specific choice for marking the y = 50 line in scatter plots.
        b) Dotted
        ... Incorrect answer feedback: The dotted linestyle is not used for the y = 50 reference line.
        *c) Dashed
        ... Correct answer feedback: A dashed line is used to indicate the y = 50 line, providing a clear visual reference within the plot.
        d) Dashdot
        ... Incorrect answer feedback: The dashdot pattern is not the linestyle chosen for the y = 50 line.

        3. Which of the following best describes the purpose of 'plt.legend()' in the plots?
        a) To create a line graph
        ... Incorrect answer feedback: 'plt.legend()' does not create line graphs; it serves a different purpose.
        b) To add a title to the plot
        ... Incorrect answer feedback: Adding a title is not the function of 'plt.legend()' in plot customization.
        *c) To display a legend on the plot
        ... Correct answer feedback: 'plt.legend()' is crucial for displaying a legend, which helps in identifying various plot elements for clarity.
        d) To change the plot background color
        ... Incorrect answer feedback: Changing the background color is not the purpose of 'plt.legend()' in plot design.

        4. What is the relationship between 'x' and 'y' in the line 'y = 2x + 3' as plotted?
        a) y is twice x minus 3
        ... Incorrect answer feedback: This description does not accurately represent the relationship between 'x' and 'y' in the given equation.
        b) y is three times x plus 2
        ... Incorrect answer feedback: This is not the correct interpretation of the relationship between 'x' and 'y' in the equation.
        *c) y is twice x plus 3
        ... Correct answer feedback: This accurately describes the relationship, where 'y' is calculated as twice 'x' plus 3, indicating a direct and proportional increase.
        d) y is half of x plus 3
        ... Incorrect answer feedback: This does not correctly describe the relationship as outlined in the equation 'y = 2x + 3'.

        5. How are the sizes of markers in the scatter plot determined?
        a) Based on the value of x
        ... Incorrect answer feedback: The size of markers is not determined by the value of x in this context.
        *b) Randomly between 0 and 100
        ... Correct answer feedback: Marker sizes are determined randomly within a specified range, adding a dynamic and varied visual aspect to the scatter plot.
        c) They are all the same size
        ... Incorrect answer feedback: Not all markers are the same size; there is variation to enhance visual differentiation.
        d) Based on the value of y
        ... Incorrect answer feedback: The value of y is not the determining factor for the size of markers in these plots.
    
#Response : This is core requiremnt. Alert with this Please make sure the  response is smilar to examples given Examples andEvery correct option should start with * as shown in example.
"""

    user_message = f""""Convert the following JSON object containing learning outcomes and associated multiple-choice questions into a QTI text format. Include only the necessary QTI text in your response. JSON: {Quetions_object} and Here is learning outcome :{lo}. Please make sure correct option to start with asterisks * for the give multiple chocie quetions."""
    print(Quetions_object)
    response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_message.strip()},
                    {"role": "user", "content": user_message.strip()}
                ],
                temperature=0
            )
            
    summary = response.choices[0].message.content
    return summary


def save_output_to_file(output_string, filename, directory=None):
    if directory is None:
        directory = os.path.join('..', 'static', 'Files')
    os.makedirs(directory, exist_ok=True)  # Create the directory if it doesn't exist
    # Ensure the filename ends with '.txt'
    if not filename.endswith('.txt'):
        filename += '.txt'
    file_path = os.path.join(directory, filename)
    with open(file_path, 'w') as file:
        file.write(output_string)
    return file_path  # Return the full path of the saved file for any further use




if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)