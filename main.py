import streamlit as st
import pandas as pd
from keybert import KeyBERT
from collections import Counter
import re

# Initialize the KeyBERT model
kw_model = KeyBERT()

st.title("Keyword Theme Extraction with Conditional Seed Keyword Removal")
st.markdown("Upload a CSV file with a 'Keywords' column and specify a seed keyword to refine theme extraction.")

# Optional input for seed keyword
seed_keyword = st.text_input("Enter Seed Keyword for Context (Optional)", value="")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv", "xls", "xlsx"])

def cluster_keywords(df, seed_keyword=''):
    """
    Clusters keywords into meaningful groups based on keyphrase extraction.

    Parameters:
    - df: pandas DataFrame with a 'Keywords' column.
    - seed_keyword: optional string to provide context for theme extraction.

    Returns:
    - pandas DataFrame with original keywords, extracted n-grams, and assigned clusters.
    """
    # Validate input
    if 'Keywords' not in df.columns:
        st.error("Error: The dataframe must contain a column named 'Keywords'.")
        return None

    # Prepare the list of words to exclude (seed keyword and its components)
    seed_words = []
    if seed_keyword:
        seed_words = seed_keyword.lower().split()

    # Function to extract keyphrases (n-grams) from text
    def extract_keyphrases(text):
        """
        Extracts keyphrases (unigrams, bigrams, trigrams) from text using KeyBERT.

        Parameters:
        - text: string to extract keyphrases from.

        Returns:
        - dict with n-grams as keys and extracted phrases as values.
        """
        keyphrases = {}
        for n in range(1, 4):
            keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(n, n), stop_words='english')
            keyphrases[f'{n}-gram'] = keywords[0][0] if keywords else ''
        return keyphrases

    # Apply keyphrase extraction to each keyword with progress
    progress_bar = st.progress(0)
    total = len(df)
    df['Keyphrases'] = ''
    for idx, row in df.iterrows():
        df.at[idx, 'Keyphrases'] = extract_keyphrases(row['Keywords'])
        progress_bar.progress((idx + 1) / total)
    progress_bar.empty()

    # Function to clean keyphrases by removing the seed keyword and its component words
    def clean_phrase(phrase):
        """
        Removes the seed keyword and its component words from the phrase if present.

        Parameters:
        - phrase: string to clean.

        Returns:
        - cleaned phrase.
        """
        cleaned = phrase
        if seed_keyword:
            # Remove the seed keyword phrase
            pattern = rf'\b{re.escape(seed_keyword)}\b'
            cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)
        if seed_words:
            # Remove individual seed words
            for word in seed_words:
                pattern = rf'\b{re.escape(word)}\b'
                cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)
        cleaned = cleaned.strip()
        return cleaned if len(cleaned.split()) > 0 else phrase  # Retain original phrase if empty after removal

    # Clean keyphrases in the DataFrame
    df['Cleaned Keyphrases'] = df['Keyphrases'].apply(
        lambda kp_dict: {k: clean_phrase(v) for k, v in kp_dict.items()}
    )

    # Combine all cleaned keyphrases to find common terms
    all_phrases = []
    for kp_dict in df['Cleaned Keyphrases']:
        all_phrases.extend(kp_dict.values())

    # Count term frequencies
    word_counts = Counter()
    for phrase in all_phrases:
        words = re.findall(r'\w+', phrase.lower())
        # Exclude seed words from counts
        words = [word for word in words if word not in seed_words]
        word_counts.update(words)

    # Get common terms that appear more than once
    common_terms = [term for term, freq in word_counts.items() if freq > 1]

    # If no common terms, assign 'Other' cluster
    if not common_terms:
        df['Cluster'] = 'Other'
    else:
        # Assign clusters based on common terms
        def assign_cluster(row):
            for term in common_terms:
                for phrase in row['Cleaned Keyphrases'].values():
                    if re.search(rf'\b{term}\b', phrase, re.IGNORECASE):
                        return term.capitalize()
            return 'Other'

        # Apply with progress
        progress_bar = st.progress(0)
        df['Cluster'] = ''
        for idx, row in df.iterrows():
            df.at[idx, 'Cluster'] = assign_cluster(row)
            progress_bar.progress((idx + 1) / total)
        progress_bar.empty()

    # Include the n-grams in the output
    df['Core (1-gram)'] = df['Keyphrases'].apply(lambda x: x['1-gram'])
    df['Core (2-gram)'] = df['Keyphrases'].apply(lambda x: x['2-gram'])
    df['Core (3-gram)'] = df['Keyphrases'].apply(lambda x: x['3-gram'])

    # Reorder columns for clarity
    output_columns = ['Keywords', 'Core (1-gram)', 'Core (2-gram)', 'Core (3-gram)', 'Cluster']

    return df[output_columns]

if uploaded_file:
    # Load the uploaded file into a DataFrame
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    with st.spinner("Clustering keywords..."):
        df_with_clusters = cluster_keywords(df, seed_keyword)

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
