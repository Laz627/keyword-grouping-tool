import shutil
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import nltk
from nltk.stem import PorterStemmer
import io
from collections import Counter
import re
import os

# Path to NLTK data directory
nltk_data_dir = os.path.join(os.getcwd(), 'nltk_data')

# Remove the existing punkt tokenizer directory to clear any corrupted files
try:
    punkt_path = os.path.join(nltk_data_dir, 'tokenizers', 'punkt')
    if os.path.exists(punkt_path):
        shutil.rmtree(punkt_path)
except Exception as e:
    print(f"Error removing punkt directory: {e}")

# Download necessary NLTK data
nltk.download('punkt', download_dir=nltk_data_dir, quiet=True)
nltk.download('stopwords', download_dir=nltk_data_dir, quiet=True)

# Point NLTK to the custom data directory
nltk.data.path.append(nltk_data_dir)

# Initialize stemmer and stopwords
stemmer = PorterStemmer()
stop_words = set(nltk.corpus.stopwords.words('english'))

def preprocess_text(text):
    # Convert to lowercase and tokenize
    words = nltk.word_tokenize(text.lower())
    # Remove stopwords and stem
    return ' '.join([stemmer.stem(word) for word in words if word.isalnum() and word not in stop_words])

def get_cluster_name(cluster_keywords, min_words=1, max_words=3):
    # Clean and split keywords
    words = [word for keyword in cluster_keywords for word in re.findall(r'\b\w+\b', keyword.lower())]
    
    # Count word frequencies
    word_counts = Counter(words)
    
    # Define words to exclude
    exclude_words = set(['for', 'what', 'why', 'how', 'when', 'where', 'it', 'which', 'who', 'whom', 'whose', 'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'of', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'shall', 'should', 'may', 'might', 'must', 'can', 'could', 'that'])
    
    # Get the most common words, excluding certain words
    common_words = [word for word, count in word_counts.most_common() 
                    if word not in exclude_words and len(word) > 1]
    
    # Function to get the most representative words
    def get_representative_words(words, min_n=min_words, max_n=max_words):
        word_set = set()
        result = []
        for word in words:
            if len(result) >= max_n:
                break
            if word not in word_set and not any(word in w or w in word for w in word_set):
                word_set.add(word)
                result.append(word)
            if len(result) >= min_n:
                # Check if the next word is significantly less common
                if len(words) > len(result) and word_counts[words[len(result)]] < word_counts[result[-1]] / 2:
                    break
        return result
    
    # Get the most representative words
    representative_words = get_representative_words(common_words)
    
    return ' '.join(representative_words)
    
    # If we don't have enough words, add generic terms
    while len(representative_words) < 3:
        if 'system' not in representative_words and 'system' in word_counts:
            representative_words.append('system')
        elif 'pos' not in representative_words and 'pos' in word_counts:
            representative_words.append('pos')
        else:
            break  # If we can't add 'system' or 'pos', we'll stop here
    
    return ' '.join(representative_words)

# Title and Instructions
st.title("Keyword Clustering Tool")
st.markdown("""
    **Instructions:**
    1. Upload a CSV file with your keywords and optional fields (Search Volume, CPC, Ranked Position, URL).
    2. Adjust the number of clusters for categorization. The higher your clustering number, the more specific the clusters will be, which can make categorization less meaningful as granularity increases.
    3. The tool will classify and cluster the keywords, and generate a downloadable CSV file with the results.
""")

# Template download
def generate_template():
    template = pd.DataFrame({
        "Keywords": ["Example keyword 1", "Example keyword 2"],
        "Search Volume": [1000, 500],
        "CPC": [0.5, 0.7],
        "Ranked Position": [1, 2],
        "URL": ["http://example.com/1", "http://example.com/2"]
    })
    return template.to_csv(index=False).encode('utf-8')

st.download_button(
    label="Download Template CSV",
    data=generate_template(),
    file_name="keyword_template.csv",
    mime="text/csv"
)

# File upload
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    required_columns = ["Keywords"]
    optional_columns = ["Search Volume", "CPC", "Ranked Position", "URL"]
    
    if "Keywords" not in df.columns:
        st.error("CSV must contain 'Keywords' column.")
    else:
        # Ensure optional columns are present and fill NaN with empty strings
        for col in optional_columns:
            if col not in df.columns:
                df[col] = ''
            else:
                df[col] = df[col].fillna('')
        
        df['Keywords'] = df['Keywords'].astype(str)

        # Preprocess keywords
        df['Processed_Keywords'] = df['Keywords'].apply(preprocess_text)

        # User input for number of clusters
        num_clusters = st.slider("Select Number of Clusters", min_value=2, max_value=150, value=10, step=1)

        if st.button("Classify and Cluster Keywords"):
            # Vectorize the processed keywords
            vectorizer = TfidfVectorizer()
            X = vectorizer.fit_transform(df['Processed_Keywords'])
        
            # Perform clustering
            kmeans = KMeans(n_clusters=num_clusters, random_state=42)
            df['Cluster'] = kmeans.fit_predict(X)
        
            # Generate cluster names
            cluster_names = []
            for cluster in range(num_clusters):
                cluster_keywords = df[df['Cluster'] == cluster]['Keywords'].tolist()
                cluster_name = get_cluster_name(cluster_keywords, min_words=1, max_words=3)
                cluster_names.append({'Cluster': cluster, 'Cluster Name': cluster_name})
        
            cluster_names_df = pd.DataFrame(cluster_names)
            
            # Merge cluster names with main dataframe
            final_df = pd.merge(df, cluster_names_df, on='Cluster', how='left')
            
            # Select and order the final columns
            final_columns = ['Keywords', 'Search Volume', 'CPC', 'Ranked Position', 'URL', 'Cluster', 'Cluster Name']
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
