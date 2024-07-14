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

# Download necessary NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

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

def get_sub_cluster_name(sub_cluster_keywords, main_cluster_name, min_words=1, max_words=3):
    main_cluster_words = set(main_cluster_name.split())
    words = [word for keyword in sub_cluster_keywords for word in re.findall(r'\b\w+\b', keyword.lower()) if word not in main_cluster_words]
    
    word_counts = Counter(words)
    
    exclude_words = set(['for', 'what', 'why', 'how', 'when', 'where', 'it', 'which', 'who', 'whom', 'whose', 'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'of', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'shall', 'should', 'may', 'might', 'must', 'can', 'could', 'that'])
    
    common_words = [word for word, count in word_counts.most_common() 
                    if word not in exclude_words and len(word) > 1]
    
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
                if len(words) > len(result) and word_counts[words[len(result)]] < word_counts[result[-1]] / 2:
                    break
        return result
    
    representative_words = get_representative_words(common_words)
    
    return ' '.join(representative_words) if representative_words else 'Other'

# Title and Instructions
st.title("Keyword Clustering Tool")
st.markdown("""
    **Instructions:**
    1. Upload a CSV file with your keywords and optional fields (Search Volume, CPC, Ranked Position, URL).
    2. Adjust the number of clusters for categorization.
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

        # Remove duplicates
        df = df.drop_duplicates(subset=['Keywords'])

        # Preprocess keywords
        df['Processed_Keywords'] = df['Keywords'].apply(preprocess_text)

        # User input for number of clusters
        num_clusters = st.slider("Select Number of Main Clusters", min_value=2, max_value=50, value=10, step=1)

        if st.button("Classify and Cluster Keywords"):
            # Vectorize the processed keywords
            vectorizer = TfidfVectorizer()
            X = vectorizer.fit_transform(df['Processed_Keywords'])

            # First-level clustering
            kmeans = KMeans(n_clusters=num_clusters, random_state=42)
            df['Main_Cluster'] = kmeans.fit_predict(X)

            # Second-level clustering
            sub_clusters = []
            for cluster in range(num_clusters):
                cluster_mask = df['Main_Cluster'] == cluster
                cluster_X = X[cluster_mask]
                
                if cluster_X.shape[0] > 1:  # Only sub-cluster if there's more than one item
                    n_subclusters = min(int(np.sqrt(cluster_X.shape[0])), 5)  # Limit number of subclusters
                    sub_kmeans = KMeans(n_clusters=n_subclusters, random_state=42)
                    sub_cluster_labels = sub_kmeans.fit_predict(cluster_X.toarray())
                    sub_clusters.extend([f"{cluster}.{label}" for label in sub_cluster_labels])
                else:
                    sub_clusters.extend([f"{cluster}.0"])

            df['Sub_Cluster'] = sub_clusters

            # Generate cluster names for both levels
            main_cluster_names = []
            sub_cluster_names = []

            for cluster in range(num_clusters):
                main_cluster_keywords = df[df['Main_Cluster'] == cluster]['Keywords'].tolist()
                main_cluster_name = get_cluster_name(main_cluster_keywords, min_words=1, max_words=3)
                main_cluster_names.append({'Main_Cluster': cluster, 'Main_Cluster_Name': main_cluster_name})

                sub_cluster_ids = df[df['Main_Cluster'] == cluster]['Sub_Cluster'].unique()
                for sub_cluster in sub_cluster_ids:
                    sub_cluster_keywords = df[df['Sub_Cluster'] == sub_cluster]['Keywords'].tolist()
                    sub_cluster_name = get_sub_cluster_name(sub_cluster_keywords, main_cluster_name, min_words=1, max_words=3)
                    sub_cluster_names.append({'Sub_Cluster': sub_cluster, 'Sub_Cluster_Name': sub_cluster_name})

            main_cluster_names_df = pd.DataFrame(main_cluster_names)
            sub_cluster_names_df = pd.DataFrame(sub_cluster_names)

            # Merge cluster names with main dataframe
            final_df = pd.merge(df, main_cluster_names_df, on='Main_Cluster', how='left')
            final_df = pd.merge(final_df, sub_cluster_names_df, on='Sub_Cluster', how='left')

            # Select and order the final columns
            final_columns = ['Keywords', 'Search Volume', 'CPC', 'Ranked Position', 'URL', 'Main_Cluster', 'Main_Cluster_Name', 'Sub_Cluster', 'Sub_Cluster_Name']
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
