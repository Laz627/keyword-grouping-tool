import streamlit as st
import pandas as pd
import re
from collections import Counter
import numpy as np

# Imports for KeyBERT and sentence embeddings
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

# Imports for NLP (if needed)
import nltk
from nltk import word_tokenize
nltk.download('punkt')

# Initialize the SentenceTransformer and KeyBERT model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
kw_model = KeyBERT(model=embedding_model)

# Streamlit App Title and Description
st.title("KeyBERT Theme Extraction for Tagging")
st.markdown(
    """
    This tool uses KeyBERT to extract candidate tagging themes from your keyword list.
    
    **Workflow:**
    
    1. Upload a CSV/Excel file with a **Keywords** column.
    2. (Optionally) Limit the processing to the first N keywords.
    3. Choose the number of keyphrases to extract per keyword.
    4. Set a minimum frequency threshold (themes occurring less frequently will be ignored).
    5. (Optionally) Specify a number of clusters to group similar candidate themes.
    
    The candidate themes (with their frequencies) are then displayed for your review.
    Use these suggestions to inform the tagging rules in your final programmatic tagging system.
    """
)

# User inputs for controlling extraction
uploaded_file = st.file_uploader("Upload CSV/Excel File", type=["csv", "xls", "xlsx"])
num_keywords = st.number_input("Process first N keywords (0 = all)", min_value=0, value=0, step=1)
top_n = st.number_input("Keyphrases per keyword", min_value=1, value=3, step=1)
min_freq = st.number_input("Minimum frequency for candidate theme", min_value=1, value=2, step=1)
num_clusters = st.number_input("Number of clusters (0 = skip clustering)", min_value=0, value=0, step=1)

if uploaded_file:
    # Load file into a DataFrame
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Error loading file: {e}")
        st.stop()
    
    # Check that the file has a "Keywords" column.
    if "Keywords" not in df.columns:
        st.error("The file must contain a column named 'Keywords'.")
    else:
        keywords_list = df["Keywords"].tolist()
        # If num_keywords is specified (non-zero), restrict to that many.
        if num_keywords > 0:
            keywords_list = keywords_list[:num_keywords]
        
        st.write("Extracting keyphrases from keywords...")
        all_phrases = []
        progress_bar = st.progress(0)
        
        # Process each keyword in the list.
        for idx, kw in enumerate(keywords_list):
            # Extract keyphrases for the current keyword.
            # We extract keyphrases of lengths 1 to 3 and take the top_n candidates.
            keyphrases = kw_model.extract_keywords(kw, keyphrase_ngram_range=(1, 3), stop_words='english', top_n=top_n)
            # keyphrases is a list of tuples (phrase, score)
            extracted = [kp[0].lower() for kp in keyphrases if kp[0]]
            all_phrases.extend(extracted)
            progress_bar.progress((idx + 1) / len(keywords_list))
        progress_bar.empty()
        
        # Count the frequency of each candidate theme.
        phrase_counts = Counter(all_phrases)
        # Only keep those with frequency >= min_freq.
        candidate_themes = {phrase: freq for phrase, freq in phrase_counts.items() if freq >= min_freq}
        
        st.write("### Candidate Themes and Frequencies")
        if candidate_themes:
            candidate_df = pd.DataFrame(list(candidate_themes.items()), columns=["Theme", "Frequency"])
            candidate_df = candidate_df.sort_values(by="Frequency", ascending=False)
            st.dataframe(candidate_df)
        else:
            st.write("No candidate themes met the minimum frequency threshold.")
        
        # Optionally, cluster the candidate themes if a positive number of clusters is provided.
        if num_clusters > 0 and len(candidate_themes) >= num_clusters:
            st.write("### Clustering Candidate Themes")
            themes = list(candidate_themes.keys())
            embeddings = embedding_model.encode(themes)
            kmeans = KMeans(n_clusters=num_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(embeddings)
            clusters = {}
            for label, theme in zip(cluster_labels, themes):
                clusters.setdefault(label, []).append(theme)
            for label, group in clusters.items():
                st.write(f"**Cluster {label}:** {', '.join(group)}")
        
        st.markdown(
            """
            **Next Steps:**  
            Use the candidate themes above as a starting point for designing your programmatic tagging rules.
            For example, you can copy these themes into a tagging document, refine or merge clusters,
            and then implement your final tagging system based on your reviewed themes.
            """
        )
