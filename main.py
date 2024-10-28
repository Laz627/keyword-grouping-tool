import streamlit as st
import pandas as pd
from keybert import KeyBERT
from collections import Counter
import re

# Initialize the KeyBERT model
kw_model = KeyBERT()

st.title("Keyword Theme Extraction with Common Phrase Identification")
st.markdown("Upload a CSV file with a 'Keywords' column and specify a seed keyword to refine theme extraction.")

# Optional input for seed keyword
seed_keyword = st.text_input("Enter Seed Keyword for Context (Optional)", value="")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv", "xls", "xlsx"])

def extract_and_cluster_keywords(df, seed_keyword=''):
    """
    Extracts keyphrases from keywords, identifies common phrases excluding seed words,
    and assigns themes based on these phrases.

    Parameters:
    - df: pandas DataFrame with a 'Keywords' column.
    - seed_keyword: optional string to provide context for theme extraction.

    Returns:
    - pandas DataFrame with original keywords, extracted n-grams, and assigned themes.
    - List of most common phrases used as themes.
    """
    # Validate input
    if 'Keywords' not in df.columns:
        st.error("Error: The dataframe must contain a column named 'Keywords'.")
        return None, None

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
            keyphrase = keywords[0][0] if keywords else ''
            keyphrases[f'{n}-gram'] = keyphrase
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
        return cleaned if len(cleaned) > 0 else phrase  # Retain original phrase if empty after removal

    # Clean keyphrases in the DataFrame
    df['Cleaned Keyphrases'] = df['Keyphrases'].apply(
        lambda kp_dict: {k: clean_phrase(v) for k, v in kp_dict.items()}
    )

    # Combine all cleaned keyphrases to find common phrases
    all_phrases = []
    for kp_dict in df['Cleaned Keyphrases']:
        all_phrases.extend([phrase.lower() for phrase in kp_dict.values() if phrase])

    # Count phrase frequencies
    phrase_counts = Counter(all_phrases)

    # Remove seed keyword and seed words from phrases
    phrases_to_exclude = [seed_keyword.lower()] + seed_words
    for phrase in phrases_to_exclude:
        if phrase in phrase_counts:
            del phrase_counts[phrase]

    # Get common phrases that appear more than once
    common_phrases = [phrase for phrase, freq in phrase_counts.items() if freq > 1]

    # If no common phrases, assign 'Other' theme
    if not common_phrases:
        df['Theme'] = 'Other'
    else:
        # Assign themes based on common phrases
        def assign_theme(row):
            for phrase in common_phrases:
                for keyphrase in row['Cleaned Keyphrases'].values():
                    if keyphrase.lower() == phrase:
                        return phrase.capitalize()
            return 'Other'

        # Apply with progress
        progress_bar = st.progress(0)
        df['Theme'] = ''
        for idx, row in df.iterrows():
            df.at[idx, 'Theme'] = assign_theme(row)
            progress_bar.progress((idx + 1) / total)
        progress_bar.empty()

    # Include the n-grams in the output
    df['Core (1-gram)'] = df['Keyphrases'].apply(lambda x: x['1-gram'])
    df['Core (2-gram)'] = df['Keyphrases'].apply(lambda x: x['2-gram'])
    df['Core (3-gram)'] = df['Keyphrases'].apply(lambda x: x['3-gram'])

    # Reorder columns for clarity
    output_columns = ['Keywords', 'Core (1-gram)', 'Core (2-gram)', 'Core (3-gram)', 'Theme']

    # Return the DataFrame and the list of common phrases
    return df[output_columns], common_phrases

if uploaded_file:
    # Load the uploaded file into a DataFrame
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    with st.spinner("Extracting keyphrases and identifying themes..."):
        df_with_themes, common_phrases = extract_and_cluster_keywords(df, seed_keyword)

    if df_with_themes is not None:
        st.write("Most Common Phrases Used as Themes (excluding seed words):")
        st.write(common_phrases)

        st.write("Keywords with Assigned Themes:")
        st.dataframe(df_with_themes)

        # Option to download the modified DataFrame
        csv = df_with_themes.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Keywords with Themes CSV",
            data=csv,
            file_name="keywords_with_themes.csv",
            mime="text/csv"
        )
