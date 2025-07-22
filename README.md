# Novel Chatbot RAG

## Overview

This project is a Retrieval-Augmented Generation (RAG) chatbot designed to answer questions about classic novels and engage in general chit-chat. It leverages a combination of information retrieval, machine learning, and large language models to provide contextually relevant answers based on a corpus of novels. The system is built with Flask for the web interface and uses various NLP and ML libraries for backend processing.

## Features

- **Chatbot Interface:** Web-based chat interface for user interaction.
- **Novel Q&A:** Answers questions related to the content, authors, and topics of classic novels.
- **Chit-chat:** Handles general conversation using BlenderBot.
- **Topic Selection:** Users can select topics to focus the retrieval process.
- **Visualization:** Provides visual analytics of chatbot usage and response times.
- **Logging:** Stores chat logs and analytics in a local SQLite database.

## File Structure

- `final_code.py` / `Project_code.py`: Main Flask app and backend logic.
- `RetreivingDocument.py`: Document retrieval and ranking logic.
- `RAG.py`: Retrieval-Augmented Generation (calls OpenAI API for answer synthesis).
- `preprocessing.py`: Text preprocessing utilities.
- `Classifier.py`: Training and saving the multi-label classifier.
- `novels_extraction.py`: Script to extract and preprocess novels from online sources.
- `novels_preprocessed_data.csv`: Preprocessed dataset of novels.
- `mlb.joblib`, `tfidf_vectorizer.joblib`, `rf_classifier.joblib`: Saved ML models and vectorizers.
- `chatbot.db`: SQLite database for chat logs.
- `templates/`: HTML templates for the web interface.

## Installation

1. **Clone the repository:**
   ```bash
   git clone <repo-url>
   cd novel_chatbot_RAG
   ```
2. **Install dependencies:**
   Create a `requirements.txt` with the following content (see below), then run:
   ```bash
   pip install -r requirements.txt
   ```

### Example `requirements.txt`
```
flask
flask-session
requests
pandas
numpy
scikit-learn
nltk
joblib
bs4
```

3. **Download NLTK data:**
   The first run will download required NLTK corpora (stopwords, punkt, wordnet).

4. **Prepare data and models:**
   - Ensure `novels_preprocessed_data.csv` and model files (`mlb.joblib`, `tfidf_vectorizer.joblib`, `rf_classifier.joblib`) are present. If not, run `novels_extraction.py` and `Classifier.py`.

## Usage

1. **Run the Flask app:**
   ```bash
   python final_code.py
   ```
   The app will start on `http://0.0.0.0:2000/` by default.

2. **Access the web interface:**
   Open your browser and go to `http://localhost:2000/`.

3. **Chat:**
   - Type your question or message in the chat box.
   - Select topics if desired.
   - View analytics via the visualization page.

## API Endpoints

- `/chat` (POST): Main chat endpoint. Expects JSON `{ "query": "...", "topics": [ ... ] }`.
- `/get-topics` (GET): Returns available novel topics.
- `/logs` (GET): Returns chat logs.
- `/data` (GET): Returns analytics data.
- `/visualization` (GET): Visualization page.
- `/` (GET): Main chat interface.

## Credits

- Uses [Project Gutenberg](https://www.gutenberg.org/) for novel data.
- Utilizes HuggingFace and OpenAI APIs for language models.

## License

This project is for educational and research purposes only. 