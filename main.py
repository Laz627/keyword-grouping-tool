import streamlit as st
import pandas as pd
import re
from nltk import word_tokenize
import nltk

# Download required NLTK data
nltk.download('punkt')

# Define a small set of stopwords for our heuristic.
STOPWORDS = {"the", "a", "an", "are", "is", "of", "and", "or", "how", "much", "do", "does", "be", "in", "to", "more"}

def omit_phrases(text, omitted_list):
    """
    Lowercases the text and removes any occurrence of words in omitted_list.
    """
    result = text.lower()
    for omit in omitted_list:
        pattern = rf'\b{re.escape(omit.lower())}\b'
        result = re.sub(pattern, '', result)
    return result.strip()

def process_keyword_by_order(keyword, omitted_list, user_a_tag, seed_keyword=""):
    """
    Processes a single keyword based on word order.
    
    Steps:
      1. Lowercase the keyword.
      2. Remove the seed keyword (if provided) and any omitted phrases.
      3. Tokenize the cleaned text.
      4. Find the index of the user‑specified A:Tag (user_a_tag) in the tokens.
      5. Let tokens_before be all tokens preceding the A:Tag and tokens_after be those following it.
      6. Remove common stopwords from both lists.
      7. For B:Tag, if tokens_after exists, choose its last token; otherwise use the last token of tokens_before.
      8. For C:Tag, if tokens_before (after stopword removal) contains more than one token, join them.
      9. For D:Tag, if there are more than one token in tokens_after (after stopword removal), join all but the last token.
      
    Returns a tuple: (Category, B:Tag, C:Tag, D:Tag)
    where Category is the user‑specified A:Tag.
    """
    # Step 1: Lowercase the keyword.
    text = keyword.lower()
    
    # Step 2: Remove the seed keyword (if provided) and omitted phrases.
    if seed_keyword:
        pattern = rf'\b{re.escape(seed_keyword.lower())}\b'
        text = re.sub(pattern, '', text)
    text = omit_phrases(text, omitted_list)
    
    # Step 3: Tokenize.
    tokens = word_tokenize(text)
    
    # Step 4: Find the index of the user‑specified A:Tag.
    try:
        idx = tokens.index(user_a_tag.lower())
    except ValueError:
        # If the specified A:Tag is not found, return empty tags.
        return ("", "", "", "")
    
    # Step 5: Split tokens.
    tokens_before = tokens[:idx]
    tokens_after = tokens[idx+1:]
    
    # Step 6: Remove stopwords.
    tokens_before_filtered = [t for t in tokens_before if t not in STOPWORDS]
    tokens_after_filtered = [t for t in tokens_after if t not in STOPWORDS]
    
    # Step 7: Determine B:Tag.
    if tokens_after_filtered:
        b_tag = tokens_after_filtered[-1]
    elif tokens_before_filtered:
        b_tag = tokens_before_filtered[-1]
    else:
        b_tag = ""
    
    # Step 8: Determine C:Tag.
    c_tag = " ".join(tokens_before_filtered) if len(tokens_before_filtered) > 1 else ""
    
    # Step 9: Determine D:Tag.
    d_tag = " ".join(tokens_after_filtered[:-1]) if len(tokens_after_filtered) > 1 else ""
    
    # Category is the user-specified A:Tag.
    return (user_a_tag.lower(), b_tag, c_tag, d_tag)

# ---------------------------
# Streamlit Interface
# ---------------------------
st.title("User-Specified Keyword Tagging")
st.markdown(
    """
    **Instructions:**
    
    1. Upload a CSV or Excel file that contains a column named **Keywords**.
    2. Enter an optional **Seed Keyword** to be removed from each keyword.
    3. Enter a comma‑separated list of **Omit Phrases** (for example, "pella, andersen") that should be removed.
    4. Enter the desired **A:Tag** (the classification keyword) that should appear in each keyword.
    
    The app will then process each keyword as follows:
      - It finds the specified A:Tag in the keyword (after cleaning).
      - It uses word order to assign:
          • **B:Tag:** the word (or token) immediately following (or, if none, preceding) the A:Tag.
          • **C:Tag:** additional modifier from words preceding the A:Tag (if available).
          • **D:Tag:** additional modifier from words following the A:Tag (if more than one word exists).
    
    **Examples:**
    
    - *Keyword:* "how much are pella replacement windows"  
      *Omit:* "pella"  
      *A:Tag:* "windows"  
      → **Category:** windows, **B:Tag:** replacement, **C:Tag:** (empty)
    
    - *Keyword:* "are pella or andersen windows more expensive"  
      *Omit:* "pella, andersen"  
      *A:Tag:* "windows"  
      → **Category:** windows, **B:Tag:** expensive, **C:Tag:** (empty)
    
    - *Keyword:* "pella hurricane shield windows cost"  
      *Omit:* "pella"  
      *A:Tag:* "windows"  
      → **Category:** windows, **B:Tag:** cost, **C:Tag:** hurricane shield
    """
)

seed_keyword = st.text_input("Seed Keyword (Optional)", value="")
omit_input = st.text_input("Omit Phrases (comma‑separated)", value="")
a_tag_input = st.text_input("A:Tag (Classification Keyword)", value="windows")
uploaded_file = st.file_uploader("Upload CSV/Excel file", type=["csv", "xls", "xlsx"])

if uploaded_file:
    # Load file
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    
    # Parse omitted phrases into a list.
    omitted_list = [s.strip() for s in omit_input.split(",") if s.strip()]
    
    categories = []
    b_tags = []
    c_tags = []
    d_tags = []
    
    # Process each keyword in the DataFrame.
    for idx, row in df.iterrows():
        keyword = row["Keywords"]
        cat, b, c, d = process_keyword_by_order(keyword, omitted_list, a_tag_input, seed_keyword)
        categories.append(cat)
        b_tags.append(b)
        c_tags.append(c)
        d_tags.append(d)
    
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
