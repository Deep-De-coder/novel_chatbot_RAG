import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template,session
from flask_session import Session
import time
import sqlite3
from RetreivingDocument import RetrievingDocument, DocumentProcessor
from preprocessing import Preprocessing
import traceback
import requests
from RAG import RAG
from joblib import load
import logging


def is_title(label, titles_list):
    """
    Determine if a label represents a book title.

    Args:
    label (str): The label to be checked.
    titles_list (list): A list of all possible titles.

    Returns:
    bool: True if the label is a title, False otherwise.
    """
    return label in titles_list

def is_author(label, authors_list):
    """
    Determine if a label represents an author's name.

    Args:
    label (str): The label to be checked.
    authors_list (list): A list of all possible authors.

    Returns:
    bool: True if the label is an author's name, False otherwise.
    """
    return label in authors_list
def init_db():
    conn = sqlite3.connect('chatbot.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS chat_logs (
            id INTEGER PRIMARY KEY, 
            user_query TEXT, 
            bot_response TEXT, 
            query_type TEXT, 
            response_time REAL, 
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

init_db()
print('Flask App Starting')
app = Flask(__name__)

# Configure logging
logging.basicConfig(filename='chatbot_error.log', level=logging.ERROR,
                    format='%(asctime)s:%(levelname)s:%(message)s')

# Load BlenderBot for chit-chat
# model_name = 'facebook/blenderbot-400M-distill'
# tokenizer = BlenderbotTokenizer.from_pretrained(model_name)
# model = BlenderbotForConditionalGeneration.from_pretrained(model_name)

# Load zero-shot classifier
#classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Load models and other resources outside of the route
rf_classifier = load('rf_classifier.joblib')
tfidf_vectorizer = load('tfidf_vectorizer.joblib')
mlb = load('mlb.joblib')
novels_data = pd.read_csv('novels_preprocessed_data.csv')
doc_processor = DocumentProcessor(novels_data,tfidf_vectorizer)
titles_list = novels_data['title'].unique().tolist()
authors_list = novels_data['author'].unique().tolist()
# print("Titles List Sample:", titles_list[:5])  # Print first 5 titles
# print("Authors List Sample:", authors_list[:5])
# print("Is Title Test:", is_title(titles_list[0], titles_list))
# print("Is Author Test:", is_author(authors_list[0], authors_list))

def query_blenderbot_via_api(prompt):
    api_url = "https://api-inference.huggingface.co/models/facebook/blenderbot-400M-distill"
    headers = {"Authorization": "Bearer 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"}

    response = requests.post(api_url, headers=headers, json={"inputs": prompt})
    if response.status_code == 200:
        return response.json()['generated_text']
    else:
        raise Exception(f"API request failed with status code {response.status_code}")
    
def query_zero_shot_classification(prompt, candidate_labels):
    api_url = "https://api-inference.huggingface.co/models/facebook/bart-large-mnli"
    headers = {"Authorization": "Bearer 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"}

    payload = {
        "inputs": prompt,
        "parameters": {"candidate_labels": candidate_labels}
    }

    response = requests.post(api_url, headers=headers, json=payload)
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"API request failed with status code {response.status_code}")



class NoLabelsException(Exception):
    """Exception raised when no labels are returned by the classifier."""
    pass

@app.route('/chat', methods=['POST'])
def chat():
    start_time = time.time()  # Start time

    try:    
        user_input = request.json.get('query', '')
        categories = ["Chit-chat", "Novel-related"] 
        processed_input = Preprocessing.preprocess_query(user_input) 
        # Call the zero-shot classification API
        results = query_zero_shot_classification(user_input, categories)
        # print(results)
        novel_related_score = results['scores'][results['labels'].index("Novel-related")]
        total_retrieved=0

        if novel_related_score >= 0.4:
            query_type = "Novel-related"            
            selected_topics = request.json.get('topics', [])  # Expecting a list of topics
            document_ids = novels_data['Document ID'].tolist()
          #  processed_input = Preprocessing.preprocess_query(user_input)
            # print("Processed Query:", processed_input)
            tfidf_matrix = tfidf_vectorizer.transform(novels_data['contents_preprocessed'])
            transformed_input = tfidf_vectorizer.transform([processed_input])
            # print("Transformed Input Shape:", transformed_input.shape)
            features_name = tfidf_vectorizer.get_feature_names_out()
            inverted_index = RetrievingDocument.build_inverted_index(tfidf_matrix,features_name,document_ids)

            if selected_topics:
                top_k_per_topic = 5
                total_limit = 5
                relevant_documents, total_retrieved = retrieve_balanced_documents(
                    selected_topics, doc_processor, processed_input, novels_data,
                    tfidf_vectorizer, inverted_index, top_k_per_topic, total_limit)
                # print(f"Total number of documents retrieved: {total_retrieved}")
                # print('Response from Topics:',relevant_documents)
                response = RAG.get_answer(relevant_documents, user_input)
                # print('After RAG:',response)
                # final_response = f"Response based on selected topics:{', '.join(selected_topics)}:\n\n{response}"
                final_response = f"{response}"

            else:
                binary_predictions = rf_classifier.predict(tfidf_vectorizer.transform([processed_input]))
                model_output = mlb.inverse_transform(binary_predictions)[0]
                if not model_output:
                    raise NoLabelsException("Classifier returned no labels.")
                # print("Classifier Output:", model_output)
                predicted_title = next((label for label in model_output if is_title(label, titles_list)), None)
                predicted_author = next((label for label in model_output if is_author(label, authors_list)), None)
                # print('Title:',predicted_title)
                # print('Author: ',predicted_author)
                top_documents, total_retrieved = RetrievingDocument.retrieve_documents_for_novel(
                processed_input, doc_processor, predicted_title, predicted_author, 
                novels_data, tfidf_vectorizer, inverted_index, top_k=5)
                # print(f'Documents Retrieved: {top_documents}, Total Retrieved: {total_retrieved}')
                response = RAG.get_answer(top_documents, user_input)
                # print('After RAG: ',response)
                # final_response = f"Title: {predicted_title}, Author: {predicted_author}. \n{response}"
                final_response = f"{response}"
        else:
            query_type = "Chit-chat"
            # Chit-chat response
            # inputs = tokenizer([user_input], return_tensors='pt')
            # reply_ids = model.generate(**inputs, max_length=1000, num_beams=5, temperature=0.7)
            # final_response = tokenizer.decode(reply_ids[0], skip_special_tokens=True)
            final_response = query_blenderbot_via_api(user_input)

        end_time = time.time()
        response_time = end_time - start_time 
        # Database insertion now includes the total_retrieved column
        conn = sqlite3.connect('chatbot.db')
        c = conn.cursor()
        c.execute('''
            INSERT INTO chat_logs (user_query, bot_response, query_type, response_time, total_retrieved) 
            VALUES (?, ?, ?, ?, ?)
        ''', (user_input, final_response, query_type, response_time, total_retrieved))
        conn.commit()
        conn.close()      

        return jsonify({'response': final_response})
    except NoLabelsException as e:
        log_error(e)
        return jsonify({'response': 'No relevant labels found. Would you like to try another question or exit?', 'error': True}), 500
    except Exception as e:
        log_error(e)
        return jsonify({'response': 'Something went wrong. Would you like to try another question or exit?', 'error': True}), 500
@app.route('/get-topics', methods=['GET'])
def get_topics():
    topics = novels_data['title'].unique().tolist()# Example topics
    return jsonify(topics)
@app.route('/logs', methods=['GET'])
def get_logs():
    conn = sqlite3.connect('chatbot.db')
    c = conn.cursor()
    c.execute("SELECT * FROM chat_logs")
    logs = c.fetchall()
    conn.close()
    return jsonify({'logs': logs})



# def update_db():
#     conn = sqlite3.connect('chatbot.db')
#     c = conn.cursor()
#     c.execute('ALTER TABLE chat_logs ADD COLUMN total_retrieved INTEGER DEFAULT 0')
#     conn.commit()
#     conn.close()

# # Call this function once to update your database
# update_db()


def log_error(e):
    logging.error(str(e))
    traceback.print_exc()

def retrieve_balanced_documents(selected_topics, doc_processor, processed_input, novels_data, tfidf_vectorizer, inverted_index, top_k_per_topic, total_limit):
    topic_pools = {
        topic: RetrievingDocument.retrieve_documents_for_topic(topic, doc_processor, processed_input, novels_data, tfidf_vectorizer, inverted_index, top_k=top_k_per_topic) 
        for topic in selected_topics
    }
    
    balanced_documents = []
    total_retrieved_count = 0  # Initialize a counter for the total retrieved documents
    
    # Iterate over the topics to accumulate the documents and the total count
    for topic, (documents, count) in topic_pools.items():
        # print(f"Topic: {topic}, Documents Retrieved: {len(documents)}, Total Count: {count}")
        total_retrieved_count += count  # Add the count from each topic
        for doc in documents:
            balanced_documents.append(doc)  # Add the document to the balanced documents
            if len(balanced_documents) >= total_limit:
                break
        if len(balanced_documents) >= total_limit:
            break

    print(f"Total Retrieved Count across all topics: {total_retrieved_count}")
    return balanced_documents, total_retrieved_count

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/visualization')
def visualization():
    return render_template('i.html')

@app.route('/data')
def data():
    conn = sqlite3.connect('chatbot.db')
    cur = conn.cursor()
    
    cur.execute("SELECT query_type, COUNT(*) as count FROM chat_logs GROUP BY query_type")
    query_types_data = cur.fetchall()
    
    cur.execute("SELECT id, response_time FROM chat_logs ORDER BY id")
    response_times_data = cur.fetchall()
    
    cur.execute("SELECT id, total_retrieved FROM chat_logs ORDER BY id")
    total_retrieved_data = cur.fetchall()
    
    conn.close()    
    query_types = [{'query_type': qt[0], 'count': qt[1]} for qt in query_types_data]
    response_times = [{'query_id': rt[0], 'response_time': rt[1]} for rt in response_times_data]
    total_retrieved = [{'query_id': tr[0], 'total_retrieved': tr[1]} for tr in total_retrieved_data]
    
    return jsonify({'query_types': query_types, 'response_times': response_times, 'total_retrieved': total_retrieved})

if __name__ == '__main__':
    app.run(debug=False)
# response = qa_bot(user_input_query, rf_classifier, retrieval_index, top_k=3) inde