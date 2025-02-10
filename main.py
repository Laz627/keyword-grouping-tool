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
# Pass it to KeyBERT.
kw_model = KeyBERT(model=embedding_model_encode)

# ---------------------------
# Imports for Adjective Extraction
# ---------------------------
import nltk
from nltk import pos_tag, word_tokenize
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# ---------------------------
# Streamlit UI
# ---------------------------
st.title("Enhanced Programmatic Keyword Tagging")
st.markdown(
    """
    Upload a CSV or Excel file with a **Keywords** column.  
    This tool will automatically extract key phrases and assign tags across multiple dimensions:
    
    - **A‑tag:** Derived from the 1‑gram keyphrase.
    - **B‑tag:** Derived from adjectives in the keyword.
    - **C‑tag:** Derived from the 2‑gram keyphrase.
    - **D‑tag:** Derived from the 3‑gram keyphrase.
    
    Optionally, provide a seed keyword (or phrase) to remove from the extracted phrases.
    """
)

seed_keyword = st.text_input("Enter Seed Keyword for Context (Optional)", value="")
uploaded_file = st.file_uploader("Upload your CSV/Excel file", type=["csv", "xls", "xlsx"])

# ---------------------------
# Helper Functions
# ---------------------------
def extract_keyphrases(text):
    """
    Uses KeyBERT to extract keyphrases at n‑gram levels 1, 2, and 3.
    Returns a dictionary with keys '1-gram', '2-gram', and '3-gram'.
    """
    keyphrases = {}
    for n in range(1, 4):
        keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(n, n), stop_words='english')
        keyphrase = keywords[0][0] if keywords else ""
        keyphrases[f"{n}-gram"] = keyphrase
    return keyphrases

def clean_phrase(phrase, seed_keyword, seed_words):
    """
    If a seed keyword is provided, remove it and its component words from the phrase.
    """
    cleaned = phrase
    if seed_keyword:
        pattern = rf'\b{re.escape(seed_keyword)}\b'
        cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)
    if seed_words:
        for word in seed_words:
            pattern = rf'\b{re.escape(word)}\b'
            cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)
    return cleaned.strip() if cleaned.strip() else phrase

def extract_adjectives_from_text(text):
    """
    Uses NLTK to extract adjectives from text.
    """
    tokens = word_tokenize(text)
    tagged = pos_tag(tokens)
    adjectives = [word for word, tag in tagged if tag in ["JJ", "JJR", "JJS"]]
    return adjectives

def process_keywords(df, seed_keyword):
    """
    Processes the DataFrame row-by-row:
      - Extracts keyphrases (1-gram, 2-gram, 3-gram) from each keyword.
      - Cleans them using the optional seed keyword.
      - Extracts adjectives.
      - Assigns programmatic tags for each dimension:
           A‑tag: From 1-gram (prefixed with "A: ")
           B‑tag: First extracted adjective (prefixed with "B: ")
           C‑tag: From 2-gram (prefixed with "C: ")
           D‑tag: From 3-gram (prefixed with "D: ")
    
    Also aggregates tag counts for each dimension.
    
    Returns:
      - A new DataFrame with additional columns.
      - A summary dictionary with tag counts.
    """
    if "Keywords" not in df.columns:
        st.error("The DataFrame must contain a 'Keywords' column.")
        return None, None
    
    seed_words = seed_keyword.lower().split() if seed_keyword else []
    
    # Lists to store per-row results.
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
        # Clean keyphrases using the provided seed keyword (if any)
        keyphrases_cleaned = {k: clean_phrase(v, seed_keyword, seed_words) for k, v in keyphrases.items()}
        
        # Extract adjectives from the original text.
        adjs = extract_adjectives_from_text(text)
        adjectives_list.append(", ".join(adjs))
        
        # Define tag dimensions.
        a_tag = "A: " + keyphrases_cleaned["1-gram"].capitalize() if keyphrases_cleaned["1-gram"] else ""
        c_tag = "C: " + keyphrases_cleaned["2-gram"].capitalize() if keyphrases_cleaned["2-gram"] else ""
        d_tag = "D: " + keyphrases_cleaned["3-gram"].capitalize() if keyphrases_cleaned["3-gram"] else ""
        # For B‑tag, use the first adjective (if any).
        b_tag = "B: " + adjs[0].capitalize() if adjs else ""
        
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
    # Read file based on extension.
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    
    with st.spinner("Processing keywords and assigning tags..."):
        result_df, tag_summary = process_keywords(df, seed_keyword)
    
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
