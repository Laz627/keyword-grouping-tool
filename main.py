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

# Function to apply KeyBERT and extract unigrams, bigrams, and trigrams with progress updates
def apply_keybert(df):
    if 'Keywords' not in df.columns:
        st.error("Error: The dataframe must contain a column named 'Keywords'.")
        return None

    # Progress bar for KeyBERT extraction
    progress_bar = st.progress(0)
    total_keywords = len(df)

    # Extract n-grams using KeyBERT
    def extract_ngram(text, ngram_range):
        keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=ngram_range, stop_words='english')
        return keywords[0][0] if keywords else ""  # Return the keyword or an empty string if none found

    # Apply KeyBERT for unigrams, bigrams, and trigrams with progress updates
    df['Core (1-gram)'] = ""
    df['Core (2-gram)'] = ""
    df['Core (3-gram)'] = ""
    for idx, row in df.iterrows():
        df.at[idx, 'Core (1-gram)'] = extract_ngram(row['Keywords'], (1, 1))
        df.at[idx, 'Core (2-gram)'] = extract_ngram(row['Keywords'], (2, 2))
        df.at[idx, 'Core (3-gram)'] = extract_ngram(row['Keywords'], (3, 3))
        progress_bar.progress((idx + 1) / total_keywords)  # Update progress bar

    return df

# Function to detect themes with conditional removal of the seed keyword
def detect_themes(df, seed_keyword=""):
    # Remove the seed keyword conditionally from bi-grams and tri-grams
    def clean_phrase(phrase):
        if seed_keyword:
            # Remove the seed keyword only if there are enough remaining words
            pattern = rf'\b{re.escape(seed_keyword)}\b'
            cleaned = re.sub(pattern, '', phrase, flags=re.IGNORECASE).strip()
            return cleaned if len(cleaned.split()) > 1 else phrase  # Retain phrase if too short after removal
        return phrase

    df['Cleaned (2-gram)'] = df['Core (2-gram)'].apply(clean_phrase)
    df['Cleaned (3-gram)'] = df['Core (3-gram)'].apply(clean_phrase)

    # Combine cleaned bi-grams and tri-grams into a single list to detect frequent terms
    all_phrases = df['Cleaned (2-gram)'].tolist() + df['Cleaned (3-gram)'].tolist()
    word_counts = Counter([word for phrase in all_phrases for word in re.findall(r'\w+', phrase.lower())])

    # Define themes based on the most common non-seed terms
    common_terms = [term for term, freq in word_counts.items() if freq > 1]

    # Function to categorize each row based on detected themes, with selective seed keyword removal
    def categorize_theme(row):
        for term in common_terms:
            if re.search(rf'\b{term}\b', row['Cleaned (3-gram)'], re.IGNORECASE):
                return term.capitalize()  # Use the frequent term as the theme
        return "Other"  # Default if no common term is found

    # Add a secondary progress bar for theme detection
    theme_progress_bar = st.progress(0)
    for idx in range(len(df)):
        df.at[idx, 'Theme'] = categorize_theme(df.iloc[idx])
        theme_progress_bar.progress((idx + 1) / len(df))

    return df

if uploaded_file:
    # Load the uploaded file into a DataFrame
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    
    # Apply KeyBERT to extract themes with progress
    with st.spinner("Extracting themes with KeyBERT..."):
        df_with_keybert = apply_keybert(df)
    
    if df_with_keybert is not None:
        # Automatically detect themes with conditional seed keyword removal
        with st.spinner("Categorizing themes..."):
            df_with_themes = detect_themes(df_with_keybert, seed_keyword)

        st.write("Extracted Themes for Keywords:")
        st.dataframe(df_with_themes[['Keywords', 'Core (1-gram)', 'Core (2-gram)', 'Core (3-gram)', 'Theme']])

        # Option to download the modified DataFrame
        csv = df_with_themes.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Extracted Themes CSV",
            data=csv,
            file_name="keywords_with_themes.csv",
            mime="text/csv"
        )
