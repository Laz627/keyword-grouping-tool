import streamlit as st
import pandas as pd
import re
from collections import Counter

# ---------------------------
# Imports for KeyBERT & Embeddings
# ---------------------------
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer

# Create a dedicated SentenceTransformer instance for encoding.
embedding_model_encode = SentenceTransformer('all-MiniLM-L6-v2')
# Initialize KeyBERT with that model.
kw_model = KeyBERT(model=embedding_model_encode)

# ---------------------------
# Imports for Adjective Extraction and Normalization
# ---------------------------
import nltk
from nltk import pos_tag, word_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()

def normalize_phrase(phrase):
    """
    Normalizes a phrase by lowercasing and lemmatizing each word (as a noun).
    This helps cluster singular and plural forms together.
    """
    tokens = word_tokenize(phrase.lower())
    norm_tokens = [lemmatizer.lemmatize(token, pos='n') for token in tokens]
    return " ".join(norm_tokens)

# ---------------------------
# Streamlit UI
# ---------------------------
st.title("Enhanced Programmatic Keyword Tagging")
st.markdown(
    """
    Upload a CSV or Excel file with a **Keywords** column.  
    This tool will extract key phrases and assign tags along multiple dimensions:
    
    - **A‑tag:** From the 1‑gram keyphrase.
    - **B‑tag:** From adjectives in the keyword.
    - **C‑tag:** From the 2‑gram keyphrase.
    - **D‑tag:** From the 3‑gram keyphrase.
    
    **Additional Options:**
    - Provide a seed keyword (or phrase) to remove from the extracted phrases.
    - Specify a comma‑separated list of phrases to omit (e.g. “Pella”) so they won’t be used for tagging.
    """
)
seed_keyword = st.text_input("Enter Seed Keyword for Context (Optional)", value="")
omit_input = st.text_input("Omit Phrases (comma-separated)", value="Pella")
uploaded_file = st.file_uploader("Upload your CSV/Excel file", type=["csv", "xls", "xlsx"])

# ---------------------------
# Helper Functions
# ---------------------------
def omit_phrases(phrase, omitted_list):
    """
    Removes any occurrence (case-insensitive) of omitted phrases from the given phrase.
    """
    result = phrase
    for omit in omitted_list:
        pattern = rf'\b{re.escape(omit.lower())}\b'
        result = re.sub(pattern, '', result.lower())
    return result.strip()

def extract_keyphrases(text):
    """
    Uses KeyBERT to extract keyphrases for 1‑gram, 2‑gram, and 3‑gram.
    Returns a dict with keys "1-gram", "2-gram", and "3-gram".
    """
    keyphrases = {}
    for n in range(1, 4):
        keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(n, n), stop_words='english')
        keyphrase = keywords[0][0] if keywords else ""
        keyphrases[f"{n}-gram"] = keyphrase
    return keyphrases

def clean_phrase(phrase, seed_keyword, seed_words, omitted_list):
    """
    Removes the seed keyword and its component words from the phrase.
    Also removes any omitted phrases.
    """
    cleaned = phrase
    if seed_keyword:
        pattern = rf'\b{re.escape(seed_keyword)}\b'
        cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)
    if seed_words:
        for word in seed_words:
            pattern = rf'\b{re.escape(word)}\b'
            cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)
    # Remove any omitted phrases.
    cleaned = omit_phrases(cleaned, omitted_list)
    return cleaned.strip() if cleaned.strip() else phrase

def extract_adjectives_from_text(text):
    """
    Extracts adjectives from the text using NLTK.
    """
    tokens = word_tokenize(text)
    tagged = pos_tag(tokens)
    adjectives = [word for word, tag in tagged if tag in ["JJ", "JJR", "JJS"]]
    return adjectives

def process_keywords(df, seed_keyword, omitted_list):
    """
    Processes each row of the DataFrame:
      - Extracts keyphrases (1‑gram, 2‑gram, 3‑gram).
      - Cleans them using the seed keyword and omitted phrases.
      - Normalizes them so that singular/plural variants are grouped.
      - Extracts adjectives (and normalizes the first one for B‑tag).
      
    Returns the updated DataFrame with new columns:
      "A-tag", "B-tag", "C-tag", "D-tag", "Adjectives",
      plus the raw core extractions.
      
    Also returns an aggregated summary (counts per tag).
    """
    if "Keywords" not in df.columns:
        st.error("The DataFrame must contain a 'Keywords' column.")
        return None, None
    
    seed_words = seed_keyword.lower().split() if seed_keyword else []
    
    # Prepare lists to hold per-row results.
    a_tags = []
    b_tags = []
    c_tags = []
    d_tags = []
    adjectives_list = []
    core_1 = []
    core_2 = []
    core_3 = []
    
    for idx, row in df.iterrows():
        text = row["Keywords"]
        keyphrases = extract_keyphrases(text)
        # Clean keyphrases with seed keyword and omitted phrases.
        keyphrases_cleaned = {
            k: clean_phrase(v, seed_keyword, seed_words, omitted_list)
            for k, v in keyphrases.items()
        }
        # Normalize the cleaned phrases for clustering (singular vs. plural)
        norm_1 = normalize_phrase(keyphrases_cleaned["1-gram"]) if keyphrases_cleaned["1-gram"] else ""
        norm_2 = normalize_phrase(keyphrases_cleaned["2-gram"]) if keyphrases_cleaned["2-gram"] else ""
        norm_3 = normalize_phrase(keyphrases_cleaned["3-gram"]) if keyphrases_cleaned["3-gram"] else ""
        
        # Extract adjectives from the original text and normalize the first one.
        adjs = extract_adjectives_from_text(text)
        norm_adj = normalize_phrase(adjs[0]) if adjs else ""
        adjectives_list.append(", ".join(adjs))
        
        # Assign tags using the normalized phrases.
        a_tag = "A: " + norm_1.capitalize() if norm_1 else ""
        b_tag = "B: " + norm_adj.capitalize() if norm_adj else ""
        c_tag = "C: " + norm_2.capitalize() if norm_2 else ""
        d_tag = "D: " + norm_3.capitalize() if norm_3 else ""
        
        a_tags.append(a_tag)
        b_tags.append(b_tag)
        c_tags.append(c_tag)
        d_tags.append(d_tag)
        core_1.append(keyphrases_cleaned["1-gram"])
        core_2.append(keyphrases_cleaned["2-gram"])
        core_3.append(keyphrases_cleaned["3-gram"])
    
    # Add new columns to the DataFrame.
    df["A-tag"] = a_tags
    df["B-tag"] = b_tags
    df["C-tag"] = c_tags
    df["D-tag"] = d_tags
    df["Adjectives"] = adjectives_list
    df["Core (1-gram)"] = core_1
    df["Core (2-gram)"] = core_2
    df["Core (3-gram)"] = core_3
    
    # Create an aggregated summary for each tag column.
    summary = {}
    for col in ["A-tag", "B-tag", "C-tag", "D-tag"]:
        counts = Counter(df[col])
        if "" in counts:
            del counts[""]
        summary[col] = dict(counts)
    
    return df, summary

# ---------------------------
# Process Uploaded File
# ---------------------------
if uploaded_file:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    
    # Parse omitted phrases from the user input.
    omitted_list = [phrase.strip().lower() for phrase in omit_input.split(",") if phrase.strip()]
    
    with st.spinner("Processing keywords and assigning tags..."):
        result_df, tag_summary = process_keywords(df, seed_keyword, omitted_list)
    
    if result_df is not None:
        st.write("### Detailed Keyword Tagging Output")
        st.dataframe(result_df[["Keywords", "A-tag", "B-tag", "C-tag", "D-tag", "Adjectives"]])
        
        st.write("### Aggregated Tag Summary")
        for tag_col, counts in tag_summary.items():
            st.write(f"**{tag_col}**")
            st.write(counts)
        
        csv_data = result_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Tagged Keywords CSV",
            data=csv_data,
            file_name="tagged_keywords.csv",
            mime="text/csv"
        )
