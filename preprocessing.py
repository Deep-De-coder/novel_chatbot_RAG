import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
class Preprocessing:
    def preprocess_query(text):
        lemmatizer = WordNetLemmatizer()
        stop_words = set(stopwords.words('english'))
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r"[^a-zA-Z0-9.,!?':;]", ' ', text)
        text = text.lower()
        tokens = text.split()
        filtered_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
        return ' '.join(filtered_tokens)
    def preprocess_content(text):
        # Similar preprocessing as preprocess_query, but for content
        lemmatizer = WordNetLemmatizer()
        stop_words = set(stopwords.words('english'))
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r"[^a-zA-Z0-9.,!?':;]", ' ', text)
        text = text.lower()
        tokens = text.split()
        filtered_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
        return ' '.join(filtered_tokens)
