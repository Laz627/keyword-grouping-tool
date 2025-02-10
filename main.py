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
         - If not found, return ("other-general", "", "", "").
      5. Let tokens_before be all normalized tokens before the found A:Tag (ignoring stopwords and tokens equal to the Category).
         – If tokens_before starts with a digit and "series" (e.g. ["250", "series", ...]), then assign the first two tokens as C:Tag and the remaining as extra_before.
         – Otherwise, let C:Tag be all tokens_before.
      6. Let tokens_after be all normalized tokens after the A:Tag (ignoring stopwords and tokens equal to the Category).
         - If tokens_after exists, let B:Tag be the *last* token from tokens_after and D:Tag be the remaining tokens (joined in order).
         - If tokens_after is empty and extra_before exists (from the special series rule), then let B:Tag be the last token of extra_before.
      7. Finally, remove any duplicate tag (if B‑Tag equals Category, or if D‑Tag equals Category or B‑Tag, clear them).
    
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
    
    # Step 4: Find the first occurrence of any user-specified A:Tag.
    idx_A = None
    category = ""
    for i, token in enumerate(norm_tokens):
        if token in user_a_tags:
            idx_A = i
            category = token
            break
    if idx_A is None:
        return ("other-general", "", "", "")
    
    # Step 5: Tokens before the A:Tag.
    raw_before = norm_tokens[:idx_A]
    tokens_before = [t for t in raw_before if t not in STOPWORDS and t != category]
    
    # Special handling: if tokens_before starts with a number and "series"
    extra_before = []
    if len(tokens_before) >= 2 and tokens_before[0].isdigit() and tokens_before[1] == "series":
        c_tag = " ".join(tokens_before[:2])
        extra_before = tokens_before[2:]
    else:
        c_tag = " ".join(tokens_before)
    
    # Step 6: Tokens after the A:Tag.
    raw_after = norm_tokens[idx_A+1:]
    tokens_after = [t for t in raw_after if t not in STOPWORDS and t != category]
    if tokens_after:
        # Use the last token as B:Tag and the rest as D:Tag.
        b_tag = tokens_after[-1]
        d_tag = " ".join(tokens_after[:-1]) if len(tokens_after) > 1 else ""
    else:
        if extra_before:
            b_tag = extra_before[-1]
            extra_before = extra_before[:-1]
        else:
            b_tag = ""
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
    2. (Optional) Enter a **Seed Keyword** to be removed from each keyword.
    3. Enter **Omit Phrases** (comma‑separated) that should be removed (for example, "pella, andersen").
    4. Enter one or more **A:Tags** (comma‑separated) that represent the classification keywords.
       (For example, "window, door, price"). These will be normalized so that "window" and "windows" match.
    
    For each keyword, the app:
      - Removes the seed/omitted words.
      - Tokenizes and normalizes the text.
      - Searches for the first occurrence of any A:Tag.
         • If found, that token becomes the **Category**.
         • Tokens after the A:Tag (ignoring stopwords) yield:
              – **B:Tag:** the last token in that group.
              – **D:Tag:** the remaining tokens (if any) before that last token.
         • Tokens before the A:Tag yield **C:Tag** (with a special rule if the beginning tokens indicate a series, e.g. "250 series").
      - If no A:Tag is found, the Category is set to **other-general**.
      - Duplicate tags across buckets are suppressed.
    """
)

seed_keyword = st.text_input("Seed Keyword (Optional)", value="")
omit_input = st.text_input("Omit Phrases (comma‑separated)", value="")
a_tags_input = st.text_input("A:Tags (comma‑separated)", value="window, door, price")
uploaded_file = st.file_uploader("Upload CSV/Excel File", type=["csv", "xls", "xlsx"])

if uploaded_file:
    # Load file into a DataFrame.
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    
    # Process user inputs.
    omitted_list = [s.strip() for s in omit_input.split(",") if s.strip()]
    # Normalize and build a set of A:Tags.
    user_a_tags = set(normalize_token(s.strip()) for s in a_tags_input.split(",") if s.strip())
    
    # Process each keyword.
    categories = []
    b_tags = []
    c_tags = []
    d_tags = []
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
