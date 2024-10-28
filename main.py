import streamlit as st
import pandas as pd
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np
import re

st.title("Keyword Clustering Using Semantic Similarity")
st.markdown("Upload a CSV file with a 'Keywords' column and specify a seed keyword to refine theme extraction.")

# Optional input for seed keyword
seed_keyword = st.text_input("Enter Seed Keyword for Context (Optional)", value="")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv", "xls", "xlsx"])

def cluster_keywords_semantically(df, seed_keyword='', num_clusters=10):
    """
    Clusters keywords into specified number of clusters based on semantic similarity.

    Parameters:
    - df: pandas DataFrame with a 'Keywords' column.
    - seed_keyword: optional string to provide context for theme extraction.
    - num_clusters: number of clusters to form.

    Returns:
    - pandas DataFrame with original keywords, assigned clusters, and cluster labels.
    """
    # Validate input
    if 'Keywords' not in df.columns:
        st.error("Error: The dataframe must contain a column named 'Keywords'.")
        return None

    # Prepare the list of words to exclude (seed keyword and its components)
    seed_words = []
    if seed_keyword:
        seed_words = seed_keyword.lower().split()

    # Remove seed words from keywords to focus on other terms
    def clean_keyword(keyword):
        cleaned = keyword.lower()
        if seed_keyword:
            # Remove the seed keyword phrase
            pattern = rf'\b{re.escape(seed_keyword.lower())}\b'
            cleaned = re.sub(pattern, '', cleaned)
        if seed_words:
            # Remove individual seed words
            for word in seed_words:
                pattern = rf'\b{re.escape(word)}\b'
                cleaned = re.sub(pattern, '', cleaned)
        cleaned = cleaned.strip()
        return cleaned if cleaned else keyword.lower()  # Retain original keyword if empty after removal

    df['Cleaned Keywords'] = df['Keywords'].apply(clean_keyword)

    # Initialize the sentence transformer model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Compute embeddings for each keyword
    embeddings = model.encode(df['Cleaned Keywords'].tolist(), show_progress_bar=True)

    # Determine optimal number of clusters using silhouette score (optional)
    # Uncomment the following block to compute optimal clusters
    """
    max_clusters = min(len(df), 20)
    silhouette_scores = []
    cluster_range = range(2, max_clusters+1)
    for n_clusters in cluster_range:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(embeddings)
        score = silhouette_score(embeddings, labels)
        silhouette_scores.append(score)
    optimal_clusters = cluster_range[silhouette_scores.index(max(silhouette_scores))]
    num_clusters = optimal_clusters
    """

    # Perform clustering using KMeans
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(embeddings)
    df['Cluster'] = cluster_labels

    # Assign cluster names based on top terms in the cluster
    cluster_names = {}
    for cluster_num in range(num_clusters):
        # Get indices of keywords in this cluster
        cluster_indices = np.where(cluster_labels == cluster_num)[0]
        cluster_keywords = df.iloc[cluster_indices]['Cleaned Keywords'].tolist()

        # Extract key terms from cluster keywords
        all_terms = ' '.join(cluster_keywords)
        # Use KeyBERT to extract representative terms for the cluster
        keywords = kw_model.extract_keywords(all_terms, keyphrase_ngram_range=(1, 2), stop_words='english', top_n=3)
        cluster_name = ', '.join([kw[0] for kw in keywords])
        cluster_names[cluster_num] = cluster_name if cluster_name else f"Cluster {cluster_num}"

    # Map cluster numbers to names
    df['Theme'] = df['Cluster'].map(cluster_names)

    # Reorder columns for clarity
    output_columns = ['Keywords', 'Theme']

    return df[output_columns]

if uploaded_file:
    # Load the uploaded file into a DataFrame
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    # Input for number of clusters
    num_clusters = st.number_input("Enter the number of clusters to form", min_value=2, max_value=50, value=10, step=1)

    with st.spinner("Clustering keywords..."):
        df_with_clusters = cluster_keywords_semantically(df, seed_keyword, num_clusters=num_clusters)

    if df_with_clusters is not None:
        st.write("Clustered Keywords:")
        st.dataframe(df_with_clusters)

        # Option to download the modified DataFrame
        csv = df_with_clusters.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Clustered Keywords CSV",
            data=csv,
            file_name="clustered_keywords.csv",
            mime="text/csv"
        )
