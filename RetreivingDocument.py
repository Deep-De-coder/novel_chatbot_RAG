from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class DocumentProcessor:

    def __init__(self, novels_data, tfidf_vectorizer):
        self.novels_data = novels_data
        self.tfidf_vectorizer = tfidf_vectorizer
        self.tfidf_vectors = {}  # Dictionary to store TF-IDF vectors
        self.preprocess_documents()

    def preprocess_documents(self):
        # Compute TF-IDF matrix
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.novels_data['contents_preprocessed'])
        # Store each document's TF-IDF vector in the dictionary
        for doc_id, vector in zip(self.novels_data['Document ID'], tfidf_matrix):
            self.tfidf_vectors[doc_id] = vector

    def get_tfidf_vector(self, doc_id):
        # Retrieve the precomputed TF-IDF vector for a given document
        return self.tfidf_vectors.get(doc_id, None)

# Initialize the processor with your novels data
#document_processor = DocumentProcessor(novels_data)

class RetrievingDocument:

    @staticmethod
    def build_inverted_index(tfidf_matrix, feature_names, document_ids):
        inverted_index = defaultdict(list)
        for doc_idx, doc in enumerate(tfidf_matrix):
            doc_id = document_ids[doc_idx]  # Get the document ID for the current document
            for word_idx in doc.indices:
                word = feature_names[word_idx]
                inverted_index[word].append(doc_id)  # Append the document ID instead of the index
        return inverted_index

    @staticmethod
    def rerank_documents(filtered_data, top_indices, additional_criteria):
        reranked_indices = sorted(top_indices, key=lambda idx: additional_criteria[idx], reverse=True)
        return reranked_indices
    @staticmethod
    def calculate_content_score(document, query_terms):
        score = 0
        for term in query_terms:
            score += document.count(term)
        return score
    @staticmethod
    def retrieve_documents_for_novel(processed_query,doc_processor,predicted_topic, predicted_author, novels_data, tfidf_vectorizer, inverted_index, top_k=5):
     #   predicted_topic, predicted_author = model_output
        #print('Predicted_topic:',predicted_topic)
       # print('Predicted_author:',predicted_author)
        # Use inverted index for initial retrieval
        query_terms = processed_query.split()
      #  print('Query terms:',query_terms)
        relevant_doc_ids = set()
        for term in query_terms:
            if term in inverted_index:
                relevant_doc_ids.update(inverted_index[term])
      #  print('Relevant Docs:',relevant_doc_ids)

        if not relevant_doc_ids:
            return []

        # Filter the dataset based on relevant document IDs
        filtered_data = novels_data[novels_data['Document ID'].isin(relevant_doc_ids)]

        # Modify this line to use an OR condition
        filtered_data = filtered_data[
            (filtered_data['title'] == predicted_topic) | 
            (filtered_data['author'] == predicted_author)
        ]

        if filtered_data.empty:
            return []

        # Calculate cosine similarities for the filtered data
        query_tfidf = doc_processor.tfidf_vectorizer.transform([processed_query])
        cos_similarities = []
        for doc_id in filtered_data['Document ID']:
            doc_tfidf_vector = doc_processor.get_tfidf_vector(doc_id)
            if doc_tfidf_vector is not None:
                similarity = cosine_similarity(query_tfidf, doc_tfidf_vector)
                cos_similarities.append(similarity[0][0])
            else:
                cos_similarities.append(0)
       # print('Cosine Similarity:',cos_similarities)
       # print(filtered_data.columns)
        # Get the top k indices with the highest similarity scores
        filtered_data['content_score'] = [RetrievingDocument.calculate_content_score(doc, query_terms) for doc in filtered_data['contents_preprocessed']]
        
        # Combine cosine similarity with content score for re-ranking
        filtered_data['combined_score'] = cos_similarities + filtered_data['content_score']
        top_indices = np.argsort(filtered_data['combined_score'])[-top_k:][::-1]
        top_documents = filtered_data.iloc[top_indices]['content_original']
        # print('Top Documents:', top_documents)
        return top_documents.tolist(), len(filtered_data)

       # return top_documents.tolist()
    @staticmethod
    def retrieve_documents_for_topic(selected_topic,doc_processor, processed_query, novels_data, tfidf_vectorizer, inverted_index, top_k=5):
    # Use inverted index for initial retrieval
        query_terms = processed_query.split()
        relevant_doc_ids = set()
        for term in query_terms:
            if term in inverted_index:
                relevant_doc_ids.update(inverted_index[term])

        if not relevant_doc_ids:
            return []

        # Filter the dataset based on relevant document IDs
        filtered_data = novels_data[novels_data['Document ID'].isin(relevant_doc_ids)]

        # Filter the data further based on the selected topic
        filtered_data = filtered_data[filtered_data['title'] == selected_topic]

        if filtered_data.empty:
            return []

        # Calculate cosine similarities for the filtered data
        query_tfidf = doc_processor.tfidf_vectorizer.transform([processed_query])
        cos_similarities = []
        for doc_id in filtered_data['Document ID']:
            doc_tfidf_vector = doc_processor.get_tfidf_vector(doc_id)
            if doc_tfidf_vector is not None:
                similarity = cosine_similarity(query_tfidf, doc_tfidf_vector)
                cos_similarities.append(similarity[0][0])
            else:
                cos_similarities.append(0)
       # print('Cosine Similarity:',cos_similarities)
       # print(filtered_data.columns)
        # Get the top k indices with the highest similarity scores
        filtered_data['content_score'] = [RetrievingDocument.calculate_content_score(doc, query_terms) for doc in filtered_data['contents_preprocessed']]
        
        # Combine cosine similarity with content score for re-ranking
        filtered_data['combined_score'] = cos_similarities + filtered_data['content_score']
        top_indices = np.argsort(filtered_data['combined_score'])[-top_k:][::-1]
        top_documents = filtered_data.iloc[top_indices]['content_original']

        return top_documents.tolist(), len(filtered_data)