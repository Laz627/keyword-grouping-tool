import streamlit as st
import pandas as pd
import re
from nltk import word_tokenize
import nltk
from nltk.stem import WordNetLemmatizer

# Download required NLTK data
nltk.download('punkt')
nltk.download('wordnet')

# Initialize NLTK lemmatizer
lemmatizer = WordNetLemmatizer()

# Define a small set of stopwords for our heuristic.
STOPWORDS = {"the", "a", "an", "are", "is", "of", "and", "or", "how", "much",
             "do", "does", "be", "in", "to", "for", "on", "at"}

def omit_phrases(text, omitted_list):
    """
    Lowercases the text and removes any occurrence of any word in omitted_list.
    """
    result = text.lower()
    for omit in omitted_list:
        pattern = rf'\b{re.escape(omit.lower())}\b'
        result = re.sub(pattern, '', result)
    return result.strip()

def normalize_token(token):
    """
    Lowercases and lemmatizes a token (using noun mode) so that singular/plural forms match.
    """
    return lemmatizer.lemmatize(token.lower(), pos='n')

def process_keyword_order(keyword, seed_keyword, omitted_list, user_a_tags):
    """
    Processes a single keyword based on word order and user-provided parameters.
    
    Steps:
      1. Lowercase the keyword.
      2. Remove the seed keyword (if provided) and any omitted phrases.
      3. Tokenize and normalize (keep only alphanumeric tokens).
      4. Search for the first occurrence of any token in the normalized user_a_tags set.
         - If found, that token becomes the Category.
         - If not found, the Category is set to "other-general" and the entire token list is used.
      5. If an A‑tag is found:
         • Let tokens_before be all normalized tokens before the found A‑tag (ignoring stopwords and tokens equal to the Category).
         • Let tokens_after be all normalized tokens after the found A‑tag (ignoring stopwords and tokens equal to the Category).
         • If tokens_after exist, assign B‑Tag as the last token from tokens_after and D‑Tag as the remaining tokens from tokens_after.
         • Otherwise, if no tokens_after exist, use the last token of tokens_before as B‑Tag.
         • C‑Tag is formed from tokens_before (joined together).
      6. If no A‑tag is found (i.e. Category is "other-general"):
         • Let tokens_all be all normalized tokens (ignoring stopwords).
         • If tokens_all exist, set B‑Tag as the last token and C‑Tag as all tokens except the last (joined together). D‑Tag remains empty.
      7. Remove duplicates (e.g. if B‑Tag equals Category, then B‑Tag is blank).
    
    Returns a tuple: (Category, B_tag, C_tag, D_tag)
    """
    # Step 1: Lowercase the keyword.
    text = keyword.lower()
    # Step 2: Remove seed keyword and omitted phrases.
    if seed_keyword:
        pattern = rf'\b{re.escape(seed_keyword.lower())}\b'
        text = re.sub(pattern, '', text)
    text = omit_phrases(text, omitted_list)
    
    # Step 3: Tokenize and normalize.
    tokens = word_tokenize(text)
    norm_tokens = [normalize_token(t) for t in tokens if t.isalnum()]
    
    # Step 4: Search for the first occurrence of any A:Tag.
    idx_A = None
    category = ""
    for i, token in enumerate(norm_tokens):
        if token in user_a_tags:
            idx_A = i
            category = token
            break
    # Step 5: If A:Tag was found, split tokens at that point.
    if idx_A is not None:
        raw_before = norm_tokens[:idx_A]
        tokens_before = [t for t in raw_before if t not in STOPWORDS and t != category]
        raw_after = norm_tokens[idx_A+1:]
        tokens_after = [t for t in raw_after if t not in STOPWORDS and t != category]
        
        # B-Tag: if tokens_after exist, use the last token; else, fallback to last token of tokens_before.
        if tokens_after:
            b_tag = tokens_after[-1]
            d_tag = " ".join(tokens_after[:-1]) if len(tokens_after) > 1 else ""
        elif tokens_before:
            b_tag = tokens_before[-1]
            d_tag = ""
        else:
            b_tag = ""
            d_tag = ""
        c_tag = " ".join(tokens_before) if tokens_before else ""
    else:
        # Step 6: No A:Tag found. Use all tokens.
        category = "other-general"
        tokens_all = [t for t in norm_tokens if t not in STOPWORDS]
        if tokens_all:
            b_tag = tokens_all[-1]
            c_tag = " ".join(tokens_all[:-1]) if len(tokens_all) > 1 else ""
        else:
            b_tag = ""
            c_tag = ""
        d_tag = ""
    
    # Step 7: Remove duplicates across buckets.
    if b_tag == category:
        b_tag = ""
    if c_tag == category:
        c_tag = ""
    if d_tag == category or d_tag == b_tag:
        d_tag = ""
    
    return (category, b_tag, c_tag, d_tag)

# ---------------------------
# Streamlit Interface
# ---------------------------
st.title("Dynamic Keyword Tagging by Word Order")
st.markdown(
    """
    **Instructions:**
    
    1. Upload a CSV/Excel file with a **Keywords** column.
    2. (Optional) Enter a **Seed Keyword** to remove from each keyword.
    3. Enter **Omit Phrases** (comma‑separated) that should be removed (e.g. "pella, andersen").
    4. Enter one or more **A:Tags** (comma‑separated) that represent the classification keywords
       (e.g. "window, door, price"). These are normalized so that "window" and "windows" match.
       
    For each keyword, the app:
      - Removes the seed/omitted words.
      - Tokenizes and normalizes the text.
      - Searches for the first occurrence of any A:Tag.
         • If found, that token becomes the **Category**.
         • Tokens after the A:Tag yield:
              – **B:Tag:** the last token (if available).
              – **D:Tag:** the remaining tokens (if any) before that last token.
         • Tokens before the A:Tag yield **C:Tag**.
      - If no A:Tag is found, the Category is set to **other-general**, and the entire token list (after filtering) is used
        to determine B:Tag (last token) and C:Tag (the remaining tokens).
      - Duplicate tags across columns are suppressed.
    """
)

seed_keyword = st.text_input("Seed Keyword (Optional)", value="")
omit_input = st.text_input("Omit Phrases (comma‑separated)", value="")
a_tags_input = st.text_input("A:Tags (comma‑separated)", value="window, door, price")
uploaded_file = st.file_uploader("Upload CSV/Excel File", type=["csv", "xls", "xlsx"])

if uploaded_file:
    # Load file.
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    
    # Process user inputs.
    omitted_list = [s.strip() for s in omit_input.split(",") if s.strip()]
    # Normalize and build a set of A:Tags.
    user_a_tags = set(normalize_token(s.strip()) for s in a_tags_input.split(",") if s.strip())
    
    categories = []
    b_tags = []
    c_tags = []
    d_tags = []
    
    # Process each keyword.
    for idx, row in df.iterrows():
        keyword = row["Keywords"]
        cat, b_tag, c_tag, d_tag = process_keyword_order(keyword, seed_keyword, omitted_list, user_a_tags)
        categories.append(cat)
        b_tags.append(b_tag)
        c_tags.append(c_tag)
        d_tags.append(d_tag)
    
    df["Category"] = categories
    df["B-tag"] = b_tags
    df["C-tag"] = c_tags
    df["D-tag"] = d_tags
    
    st.write("### Keyword Tagging Output")
    st.dataframe(df[["Keywords", "Category", "B-tag", "C-tag", "D-tag"]])
    
    csv_data = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Tagged Keywords CSV",
        data=csv_data,
        file_name="tagged_keywords.csv",
        mime="text/csv"
    )
