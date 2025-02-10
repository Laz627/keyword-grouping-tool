import streamlit as st
import pandas as pd
import re
from collections import Counter

# KeyBERT and clustering imports
from keybert import KeyBERT
from sklearn.cluster import KMeans
import numpy as np

# Import SentenceTransformer and create a dedicated encoder instance
from sentence_transformers import SentenceTransformer
# This instance is used for generating embeddings
embedding_model_encode = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize KeyBERT with the encoding model
kw_model = KeyBERT(model=embedding_model_encode)

# For optional adjectives extraction using NLTK
import nltk
from nltk import pos_tag, word_tokenize
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# -----------------------------
# Streamlit UI Configuration
# -----------------------------
st.title("Automated Keyword Clustering & Tagging with KeyBERT")
st.markdown(
    """
    **Instructions:**
    
    - Upload a CSV or Excel file containing a **Keywords** column.
    - Optionally enter a seed keyword (or phrase) to remove from the results.
    - Choose the number of clusters (themes) and decide whether to extract adjectives.
    """
)

# User inputs
seed_keyword = st.text_input("Enter Seed Keyword for Context (Optional)", value="")
n_clusters = st.number_input("Number of Clusters for Themes", min_value=2, value=5, step=1)
extract_adjectives = st.checkbox("Extract Adjectives from Keywords", value=False)
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv", "xls", "xlsx"])

# -----------------------------
# Helper Functions
# -----------------------------
def extract_keyphrases(text):
    """
    Extracts keyphrases using KeyBERT for 1-gram, 2-gram, and 3-gram.
    Returns a dictionary with keys '1-gram', '2-gram', and '3-gram'.
    """
    keyphrases = {}
    for n in range(1, 4):
        keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(n, n), stop_words='english')
        keyphrase = keywords[0][0] if keywords else ''
        keyphrases[f'{n}-gram'] = keyphrase
    return keyphrases

def clean_phrase(phrase, seed_keyword, seed_words):
    """
    Removes the seed keyword and its component words from a given phrase.
    """
    cleaned = phrase
    if seed_keyword:
        pattern = rf'\b{re.escape(seed_keyword)}\b'
        cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)
    if seed_words:
        for word in seed_words:
            pattern = rf'\b{re.escape(word)}\b'
            cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.strip()
    return cleaned if cleaned else phrase  # Return original if cleaning results in an empty string

def extract_adjectives_from_text(text):
    """
    Uses NLTK to extract adjectives from the provided text.
    """
    tokens = word_tokenize(text)
    tagged = pos_tag(tokens)
    adjectives = [word for word, tag in tagged if tag in ['JJ', 'JJR', 'JJS']]
    return adjectives

def process_keywords(df, seed_keyword, n_clusters, extract_adjectives):
    """
    Processes the DataFrame by:
      - Extracting and cleaning keyphrases for each row.
      - Combining phrases and clustering them using KMeans.
      - Assigning each row a theme based on the most frequent cluster among its phrases.
      - Optionally extracting adjectives from the original keyword text.
      
    Returns:
      - The updated DataFrame.
      - A mapping of cluster labels to theme names.
    """
    if 'Keywords' not in df.columns:
        st.error("Error: The dataframe must contain a column named 'Keywords'.")
        return None, None

    # Create a list of seed words (lowercase) for removal
    seed_words = seed_keyword.lower().split() if seed_keyword else []
    
    progress_bar = st.progress(0)
    all_cleaned_phrases = []  # To store all cleaned phrases across rows
    extracted_data = []       # To store per-row data (keywords and extracted keyphrases)

    # Loop through each row and process the keywords
    for idx, row in df.iterrows():
        kp = extract_keyphrases(row['Keywords'])
        # Clean each extracted phrase by removing seed keyword(s)
        cleaned_kp = {k: clean_phrase(v, seed_keyword, seed_words) for k, v in kp.items()}
        # Combine the cleaned phrases (converted to lowercase) for this row
        combined_phrases = list(set([phrase.lower() for phrase in cleaned_kp.values() if phrase]))
        all_cleaned_phrases.extend(combined_phrases)
        extracted_data.append({
            'Keywords': row['Keywords'],
            'Keyphrases': kp,
            'Cleaned': cleaned_kp,
            'Combined': combined_phrases
        })
        progress_bar.progress((idx + 1) / len(df))
    progress_bar.empty()

    # Get unique phrases across all rows
    unique_phrases = list(set(all_cleaned_phrases))
    if not unique_phrases:
        st.error("No keyphrases were extracted from the data.")
        return None, None

    # Compute embeddings using the dedicated encoder
    embeddings = embedding_model_encode.encode(unique_phrases)

    # Cluster the unique phrases using KMeans
    k = min(n_clusters, len(unique_phrases))
    kmeans = KMeans(n_clusters=k, random_state=42)
    cluster_labels = kmeans.fit_predict(embeddings)

    # Map each unique phrase to its cluster label
    phrase_to_cluster = {phrase: label for phrase, label in zip(unique_phrases, cluster_labels)}
    cluster_to_phrases = {}
    for phrase, label in phrase_to_cluster.items():
        cluster_to_phrases.setdefault(label, []).append(phrase)

    # For each cluster, select a representative theme phrase (using frequency)
    cluster_themes = {}
    for label, phrases in cluster_to_phrases.items():
        freq = {phrase: all_cleaned_phrases.count(phrase) for phrase in phrases}
        theme_phrase = max(freq, key=freq.get) if freq else phrases[0]
        cluster_themes[label] = theme_phrase.capitalize()

    # Assign a theme to each row based on the clusters of its keyphrases
    assigned_themes = []
    for item in extracted_data:
        clusters_in_row = [phrase_to_cluster[phrase] for phrase in item['Combined'] if phrase in phrase_to_cluster]
        if clusters_in_row:
            # Use the most frequent cluster for the row as its theme
            cluster_label = max(set(clusters_in_row), key=clusters_in_row.count)
            theme = cluster_themes[cluster_label]
        else:
            theme = "Other"
        assigned_themes.append(theme)

    # Build the final DataFrame with additional columns for keyphrases and themes
    df_result = df.copy()
    df_result['Core (1-gram)'] = [data['Keyphrases']['1-gram'] for data in extracted_data]
    df_result['Core (2-gram)'] = [data['Keyphrases']['2-gram'] for data in extracted_data]
    df_result['Core (3-gram)'] = [data['Keyphrases']['3-gram'] for data in extracted_data]
    df_result['Theme'] = assigned_themes

    # Optionally, extract adjectives from the original keywords
    if extract_adjectives:
        df_result['Adjectives'] = df_result['Keywords'].apply(
            lambda text: ", ".join(extract_adjectives_from_text(text))
        )
    
    return df_result, cluster_themes

# -----------------------------
# Process the Uploaded File
# -----------------------------
if uploaded_file:
    # Load the file into a DataFrame based on its extension
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    with st.spinner("Processing keywords and clustering themes..."):
        result_df, cluster_themes = process_keywords(df, seed_keyword, n_clusters, extract_adjectives)

    if result_df is not None:
        st.write("### Identified Cluster Themes")
        st.write(cluster_themes)
        
        st.write("### Keywords with Assigned Themes")
        st.dataframe(result_df)

        # Provide an option to download the resulting DataFrame
        csv_data = result_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Keywords with Themes CSV",
            data=csv_data,
            file_name="keywords_with_themes.csv",
            mime="text/csv"
        )
