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
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
kw_model = KeyBERT(model=embedding_model)

# ---------------------------
# Imports for NLP Normalization and POS Tagging
# ---------------------------
import nltk
from nltk import word_tokenize, pos_tag
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()

# ---------------------------
# Helper Functions
# ---------------------------
def extract_keyphrases(text):
    """
    Uses KeyBERT to extract keyphrases at 1-gram, 2-gram, and 3-gram levels.
    Returns a dictionary with keys "1-gram", "2-gram", and "3-gram".
    """
    keyphrases = {}
    for n in range(1, 4):
        keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(n, n), stop_words='english')
        phrase = keywords[0][0] if keywords else ""
        keyphrases[f"{n}-gram"] = phrase
    return keyphrases

def omit_phrases(text, omitted_list):
    """
    Lowercases the text and removes any occurrence of any word in omitted_list.
    """
    result = text.lower()
    for omit in omitted_list:
        pattern = rf'\b{re.escape(omit.lower())}\b'
        result = re.sub(pattern, '', result)
    return result.strip()

def clean_phrase(phrase, seed_keyword, omitted_list):
    """
    Removes the seed keyword and any omitted phrases from the given phrase.
    """
    cleaned = phrase
    if seed_keyword:
        pattern = rf'\b{re.escape(seed_keyword)}\b'
        cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)
    cleaned = omit_phrases(cleaned, omitted_list)
    return cleaned.strip()

def normalize_phrase(phrase):
    """
    Tokenizes, lowercases, and lemmatizes the phrase (as nouns) so that similar words are unified.
    """
    tokens = word_tokenize(phrase.lower())
    norm_tokens = [lemmatizer.lemmatize(t, pos='n') for t in tokens]
    return " ".join(norm_tokens)

def extract_adjective(text):
    """
    Uses NLTK to extract the first adjective from the text.
    Returns an empty string if none is found.
    """
    tokens = word_tokenize(text)
    tags = pos_tag(tokens)
    for word, tag in tags:
        if tag.startswith('JJ'):  # adjective
            return word.lower()
    return ""

def process_keyword(keyword, seed_keyword, omitted_list):
    """
    Process a single keyword string:
      - Extracts keyphrases using KeyBERT.
      - Cleans each keyphrase by removing the seed keyword and omitted phrases.
      - Normalizes the cleaned keyphrases.
      - Uses:
          * Normalized 1-gram as the Category.
          * The first extracted adjective (if any) as the Adjective.
          * Normalized 2-gram (with the Category removed if it is a prefix) as the C-tag.
          * Normalized 3-gram (with occurrences of Category or Adjective removed) as the D-tag.
    Returns a tuple: (Category, Adjective, C-tag, D-tag).
    """
    # Extract keyphrases from the keyword text.
    keyphrases = extract_keyphrases(keyword)
    core1 = clean_phrase(keyphrases.get("1-gram", ""), seed_keyword, omitted_list)
    core2 = clean_phrase(keyphrases.get("2-gram", ""), seed_keyword, omitted_list)
    core3 = clean_phrase(keyphrases.get("3-gram", ""), seed_keyword, omitted_list)
    # Normalize each extracted phrase.
    norm1 = normalize_phrase(core1) if core1 else ""
    norm2 = normalize_phrase(core2) if core2 else ""
    norm3 = normalize_phrase(core3) if core3 else ""
    # Category is taken as the normalized 1-gram.
    category = norm1
    # Extract an adjective from the original keyword text.
    adjective = extract_adjective(keyword)
    # For C-tag, use the normalized 2-gram.
    c_tag = norm2
    if category and c_tag.startswith(category):
        c_tag = c_tag[len(category):].strip()
    # For D-tag, use the normalized 3-gram and remove occurrences of the Category and adjective.
    d_tag = norm3
    if category:
        d_tag = d_tag.replace(category, "").strip()
    if adjective:
        d_tag = d_tag.replace(adjective, "").strip()
    return category, adjective, c_tag, d_tag

def process_keywords(df, seed_keyword, omitted_list):
    """
    Processes each keyword in the DataFrame (expects a "Keywords" column).
    Returns the modified DataFrame with new columns:
      "Category", "Adjective", "C-tag", "D-tag"
    and an aggregated summary of tag counts.
    """
    categories = []
    adjectives = []
    c_tags = []
    d_tags = []
    
    for idx, row in df.iterrows():
        keyword = row["Keywords"]
        cat, adj, ctag, dtag = process_keyword(keyword, seed_keyword, omitted_list)
        categories.append(cat)
        adjectives.append(adj)
        c_tags.append(ctag)
        d_tags.append(dtag)
    
    df["Category"] = categories
    df["Adjective"] = adjectives
    df["C-tag"] = c_tags
    df["D-tag"] = d_tags
    
    # Build an aggregated summary for each tag column.
    summary = {}
    for col in ["Category", "Adjective", "C-tag", "D-tag"]:
        counts = Counter(df[col])
        if "" in counts:
            del counts[""]
        summary[col] = dict(counts)
    return df, summary

# ---------------------------
# Streamlit Interface
# ---------------------------
st.title("Dynamic Keyword Tagging")
st.markdown(
    """
    Upload a CSV or Excel file with a **Keywords** column.
    
    The app will dynamically extract and assign tags based on the content of each keyword.
    
    **Options:**
    - **Seed Keyword:** Any text you wish to remove from the keywords.
    - **Omit Phrases:** A comma-separated list of words (for example, "Pella") that will be omitted from the tag derivation.
    """
)
seed_keyword = st.text_input("Seed Keyword (Optional)", value="")
omit_input = st.text_input("Omit Phrases (comma-separated)", value="")
uploaded_file = st.file_uploader("Upload file", type=["csv", "xls", "xlsx"])

if uploaded_file:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    omitted_list = [s.strip() for s in omit_input.split(",") if s.strip()]
    
    with st.spinner("Processing keywords..."):
        result_df, tag_summary = process_keywords(df, seed_keyword, omitted_list)
    
    st.write("### Keyword Tagging Output")
    st.dataframe(result_df[["Keywords", "Category", "Adjective", "C-tag", "D-tag"]])
    
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
