import streamlit as st
import pandas as pd
import re
from collections import Counter

# ---------------------------
# Imports for NLP Normalization
# ---------------------------
import nltk
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()

# ---------------------------
# Streamlit UI
# ---------------------------
st.title("Programmatic Keyword Tagging")
st.markdown(
    """
    Upload a CSV or Excel file with a **Keywords** column.  
    The app will process each keyword (after omitting any specified phrases and seed text) 
    and output five columns:
    
    - **Keywords** (original)
    - **Category** – e.g. “door”, “window”, “price-general”, “comparison”
    - **Adjective** – e.g. “price”, “installation”, “cost”, “expensive”, “replacement”
    - **C-tag** – additional qualifier (often modifiers preceding the category)
    - **D-tag** – additional modifier (often a series indicator such as “250-series”)
    
    **Options:**
    - *Seed Keyword*: any text to be removed from each keyword.
    - *Omit Phrases*: a comma‑separated list of words (e.g. “Pella”) that will be removed.
    """
)

seed_keyword = st.text_input("Enter Seed Keyword for Context (Optional)", value="")
omit_input = st.text_input("Omit Phrases (comma‑separated)", value="Pella")
uploaded_file = st.file_uploader("Upload your CSV/Excel file", type=["csv", "xls", "xlsx"])

# ---------------------------
# Helper Functions
# ---------------------------
def omit_phrases(text, omitted_list):
    """
    Lowercase the text and remove any occurrence of omitted phrases.
    """
    result = text.lower()
    for omit in omitted_list:
        pattern = rf'\b{re.escape(omit.lower())}\b'
        result = re.sub(pattern, '', result)
    return result.strip()

def normalize_tokens(tokens):
    """
    Lowercase and lemmatize each token (as noun) so that plural/singular forms cluster together.
    """
    return [lemmatizer.lemmatize(t.lower(), pos='n') for t in tokens]

def get_category(tokens):
    """
    Heuristic for Category:
      - If any token is "door" (or "doors"), return "door"
      - Else if any token is "window" (or "windows"), return "window"
      - Else if token "vs" is present, return "comparison"
      - Else if any token in {"cost", "price", "quote", "pric", "quot"} is present, return "price-general"
      - Otherwise, return an empty string.
    """
    if any(t in ["door", "doors"] for t in tokens):
        return "door"
    elif any(t in ["window", "windows"] for t in tokens):
        return "window"
    elif "vs" in tokens:
        return "comparison"
    elif any(t in ["cost", "price", "quote", "pric", "quot"] for t in tokens):
        return "price-general"
    else:
        return ""

def get_adjective(tokens):
    """
    Return the first token that appears to indicate a pricing or installation attribute.
    If none is found, default to "cost".
    """
    candidates = ["cost", "price", "installation", "expensive", "replacement", "quote",
                  "pric", "expens", "install", "replac"]
    for t in tokens:
        for cand in candidates:
            if cand in t:
                return t
    return "cost"

def get_d_tag(tokens):
    """
    Look at the very beginning of the token list for a series indicator.
    For example, if the first token is numeric and the second token is "series",
    return "X-series" (e.g. "250-series").
    Otherwise, if the first token is in a set of qualifiers that appear in our rules,
    return that token.
    """
    d_tag = ""
    if len(tokens) >= 2 and tokens[0].isdigit() and tokens[1] == "series":
        d_tag = tokens[0] + "-series"
        # Remove these tokens for further processing (handled in process_keyword)
    elif tokens and tokens[0] in {"architect", "defender", "lifestyle", "150", "250", "350", 
                                   "bow", "casement", "front", "impervia", "double-hung", "lowes", "bay"}:
        d_tag = tokens[0]
    return d_tag

def get_c_tag(tokens, category, adjective):
    """
    Determine C-tag as follows:
      - Find the first occurrence of the category token.
      - Then take tokens from the beginning (after removing common filler words)
        up to the category token.
      - Also, if the first token equals the adjective, skip it.
    """
    fillers = {"of", "the", "and", "a", "an"}
    try:
        idx_cat = tokens.index(category)  # first occurrence
    except ValueError:
        idx_cat = len(tokens)
    c_tokens = tokens[:idx_cat]
    if c_tokens and c_tokens[0] == adjective:
        c_tokens = c_tokens[1:]
    c_tokens = [t for t in c_tokens if t not in fillers]
    return " ".join(c_tokens)

def process_keyword(keyword, seed_keyword, omitted_list):
    """
    Process a single keyword string and return a tuple:
      (Category, Adjective, C-tag, D-tag)
    using simple token-based heuristics.
    """
    # Lowercase the keyword.
    text = keyword.lower()
    # Remove the seed keyword (if provided) and any omitted phrases.
    if seed_keyword:
        pattern = rf'\b{re.escape(seed_keyword.lower())}\b'
        text = re.sub(pattern, '', text)
    text = omit_phrases(text, omitted_list)
    # Tokenize and normalize.
    tokens = word_tokenize(text)
    tokens = normalize_tokens(tokens)
    if not tokens:
        return ("", "", "", "")
    
    # Determine D-tag from the beginning tokens.
    d_tag = get_d_tag(tokens)
    # If D-tag was found as a numeric series, remove the first two tokens.
    if d_tag and len(tokens) >= 2 and tokens[0].isdigit() and tokens[1] == "series":
        tokens = tokens[2:]
    # Determine Category.
    category = get_category(tokens)
    # Determine Adjective.
    adjective = get_adjective(tokens)
    # Determine C-tag.
    c_tag = get_c_tag(tokens, category, adjective)
    
    return (category, adjective, c_tag, d_tag)

def process_keywords(df, seed_keyword, omitted_list):
    """
    For each row in the DataFrame (with column "Keywords"), process the keyword text 
    and add new columns:
      "Category", "Adjective", "C-tag", "D-tag".
    Returns the modified DataFrame and an aggregated tag summary.
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
    
    # Build an aggregated summary (counts per new tag column).
    summary = {}
    for col in ["Category", "Adjective", "C-tag", "D-tag"]:
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
    
    # Parse omitted phrases (comma-separated)
    omitted_list = [phrase.strip() for phrase in omit_input.split(",") if phrase.strip()]
    
    with st.spinner("Processing keywords and assigning tags..."):
        result_df, tag_summary = process_keywords(df, seed_keyword, omitted_list)
    
    if result_df is not None:
        st.write("### Programmatic Keyword Tagging Output")
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
