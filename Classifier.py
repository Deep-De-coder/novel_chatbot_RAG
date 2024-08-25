import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import numpy as np

from sklearn.metrics import accuracy_score, hamming_loss, classification_report
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from joblib import dump
from sklearn.ensemble import RandomForestClassifier
from preprocessing import Preprocessing
# class svm_classifier:
#     # def __init__(self,path_to_data):
#     #     self.data = path_to_data
#     @staticmethod
#     def run_classifier(novels_data):
        # novels_data = pd.read_csv(self.data)
novels_data = pd.read_csv('novels_data.csv')
novels_data['title_author'] = novels_data['title'] + " by " + novels_data['author']
novels_data['contents_preprocessed'] = novels_data['content_original'].apply(Preprocessing.preprocess_content)
novels_data = novels_data.drop(['content_preprocessed'],axis=1)
novels_data.to_csv('novels_preprocessed_data.csv', index=False)
        # Initialize TF-IDF Vectorizer and Label Encoder
tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        # label_encoder = LabelEncoder()
print('tfidf_vectorization:',tfidf_vectorizer)
        # Encode target variable and extract features
        # y_encoded = label_encoder.fit_transform(novels_data['title_author'])
X_tfidf = tfidf_vectorizer.fit_transform(novels_data['contents_preprocessed'])
print('X_tdidf:',X_tfidf)

        # Assuming 'topic' and 'author' are columns in your dataset
y = novels_data[['title', 'author']]

        # Use MultiLabelBinarizer for multi-label encoding
mlb = MultiLabelBinarizer()
y_encoded = mlb.fit_transform(y.values)

# Initialize the Random Forest classifier
rf_classifier = OneVsRestClassifier(RandomForestClassifier(n_estimators=100, random_state=42))

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y_encoded, test_size=0.2, random_state=42)

# Train the classifier
rf_classifier.fit(X_train, y_train)

# Predict and calculate accuracy
y_pred_rf = rf_classifier.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
hamming = hamming_loss(y_test, y_pred_rf)
report = classification_report(y_test, y_pred_rf, target_names=mlb.classes_)

print("Random Forest Accuracy:", accuracy_rf)
print("Hamming Loss: ", hamming)
print("Classification Report:\n", report)

# Save the classifier, vectorizer, and label binarizer
dump(rf_classifier, 'rf_classifier.joblib')
dump(tfidf_vectorizer, 'tfidf_vectorizer.joblib')
dump(mlb, 'mlb.joblib')