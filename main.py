import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import io
from collections import Counter
import re
from gensim.models import Word2Vec
from gensim.models.phrases import Phrases, Phraser

# Download necessary NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Initialize lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # Convert to lowercase and tokenize
    words = word_tokenize(text.lower())
    # Remove stopwords and lemmatize
    return ' '.join([lemmatizer.lemmatize(word) for word in words if word.isalnum() and word not in stop_words])

def train_word2vec(texts):
    sentences = [text.split() for text in texts]
    phrases = Phrases(sentences, min_count=1, threshold=1)
    bigram = Phraser(phrases)
    sentences = [bigram[sent] for sent in sentences]
    model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)
    return model

def get_optimal_clusters(X, max_clusters=20):
    silhouette_scores = []
    for n_clusters in range(2, max_clusters+1):
        clusterer = AgglomerativeClustering(n_clusters=n_clusters)
        cluster_labels = clusterer.fit_predict(X)
        silhouette_avg = silhouette_score(X, cluster_labels)
        silhouette_scores.append(silhouette_avg)
    return silhouette_scores.index(max(silhouette_scores)) + 2

def get_cluster_name(cluster_keywords, word2vec_model, min_words=1, max_words=3):
    # Clean and split keywords
    words = [word for keyword in cluster_keywords for word in re.findall(r'\b\w+\b', keyword.lower())]
    
    # Count word frequencies
    word_counts = Counter(words)
    
    # Get the most common words
    common_words = [word for word, count in word_counts.most_common() 
                    if word not in stop_words and len(word) > 1]
    
    # Use Word2Vec to find related words
    related_words = []
    for word in common_words[:5]:  # Consider top 5 common words
        if word in word2vec_model.wv:
            related_words.extend([w for w, _ in word2vec_model.wv.most_similar(word, topn=3)])
    
    # Combine common and related words
    candidate_words = common_words + related_words
    
    # Get the most representative words
    representative_words = []
    for word in candidate_words:
        if len(representative_words) >= max_words:
            break
        if word not in representative_words and not any(word in w or w in word for w in representative_words):
            representative_words.append(word)
        if len(representative_words) >= min_words:
            if len(candidate_words) > len(representative_words) and word_counts.get(candidate_words[len(representative_words)], 0) < word_counts.get(representative_words[-1], 1) / 2:
                break
    
    return ' '.join(representative_words)

# Title and Instructions
st.title("Keyword Clustering Tool")
st.markdown("""
    **Instructions:**
    1. Upload a CSV file with your keywords in a column named 'Keywords'.
    2. The tool will automatically determine the optimal number of clusters.
    3. Click 'Classify and Cluster Keywords' to process your data.
    4. Download the results as a CSV file.
""")

# File upload
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    if "Keywords" not in df.columns:
        st.error("CSV must contain 'Keywords' column.")
    else:
        # Preprocess keywords
        df['Processed_Keywords'] = df['Keywords'].apply(preprocess_text)

        # Train Word2Vec model
        word2vec_model = train_word2vec(df['Processed_Keywords'])

        # Vectorize the processed keywords
        vectorizer = TfidfVectorizer(ngram_range=(1, 2))
        X = vectorizer.fit_transform(df['Processed_Keywords'])

        # Determine optimal number of clusters
        optimal_clusters = get_optimal_clusters(X.toarray())
        st.write(f"Optimal number of clusters: {optimal_clusters}")

        if st.button("Classify and Cluster Keywords"):
            # Perform clustering
            clustering = AgglomerativeClustering(n_clusters=optimal_clusters)
            df['Cluster'] = clustering.fit_predict(X.toarray())

            # Generate cluster names
            cluster_names = []
            for cluster in range(optimal_clusters):
                cluster_keywords = df[df['Cluster'] == cluster]['Keywords'].tolist()
                cluster_name = get_cluster_name(cluster_keywords, word2vec_model)
                cluster_names.append({'Cluster': cluster, 'Cluster Name': cluster_name})

            cluster_names_df = pd.DataFrame(cluster_names)
            
            # Merge cluster names with main dataframe
            final_df = pd.merge(df, cluster_names_df, on='Cluster', how='left')
            
            # Select and order the final columns
            final_columns = ['Keywords', 'Cluster', 'Cluster Name']
            final_df = final_df[final_columns]
        
            # Prepare CSV for download
            output = io.BytesIO()
            final_df.to_csv(output, index=False)
            output.seek(0)
            
            st.download_button(
                label="Download Clustered Keywords CSV",
                data=output,
                file_name="clustered_keywords.csv",
                mime="text/csv"
            )
        
            st.dataframe(final_df)

# Run this app with: streamlit run script_name.py
