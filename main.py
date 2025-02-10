import streamlit as st
import pandas as pd
import re
from nltk import word_tokenize
import nltk
from nltk.stem import WordNetLemmatizer

# Download required NLTK data (only if not already cached)
nltk.download('punkt')
nltk.download('wordnet')

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Define a small set of stopwords (if desired)
STOPWORDS = {"the", "a", "an", "are", "is", "of", "and", "or", "how", "much",
             "do", "does", "be", "in", "to", "for", "on", "at"}

# Low-priority A-tags (which we do not want to use for classification if a better candidate exists)
LOW_PRIORITY = {"price", "cost", "comparison", "calculator"}

def normalize_token(token):
    """
    Lowercase and lemmatize a token (noun mode).
    Also, if token is 'vs', convert it to 'v'.
    """
    token = token.lower()
    if token == "vs":
        token = "v"
    return lemmatizer.lemmatize(token, pos='n')

def create_token_lists(text, omitted_set):
    """
    Given raw text, produce two lists:
      - raw_tokens: all normalized tokens (keeping omitted words)
      - filtered_tokens: raw_tokens with any token in omitted_set removed.
    Only alphanumeric tokens are kept.
    """
    tokens = word_tokenize(text.lower())
    raw_tokens = [normalize_token(t) for t in tokens if t.isalnum()]
    filtered_tokens = [t for t in raw_tokens if t not in omitted_set]
    return raw_tokens, filtered_tokens

def process_keyword_order(keyword, seed_keyword, omitted_list, user_a_tags):
    """
    Processes a single keyword based on word order and user-provided parameters.
    
    Parameters:
      - keyword: the original keyword string.
      - seed_keyword: (optional) a string to remove from the keyword (for classification purposes).
      - omitted_list: list of words to omit for classification (but not necessarily from output).
      - user_a_tags: a set of normalized A-tags (from user input).
    
    Processing steps:
      1. Lowercase the keyword and, if provided, remove the seed keyword (by word-boundary).
      2. Create two token lists from the cleaned text:
         • raw_tokens: all normalized tokens.
         • filtered_tokens: raw_tokens with any token in the omitted set removed.
      3. Search filtered_tokens for the first occurrence of any token that is in user_a_tags
         and is not in LOW_PRIORITY. If found, let that token (from raw_tokens) be the Category.
         Otherwise, set Category to "other-general".
      4. If a Category (from an A-tag) was found, let idx be the first index (in raw_tokens)
         where the normalized token (that is not omitted) equals that candidate.
         Then define:
             - tokens_before = raw_tokens[0:idx]
             - tokens_after = raw_tokens[idx+1:]
         Else (Category is "other-general"):
             - tokens_all = raw_tokens.
      5. For a found A-tag:
         - B-tag: if tokens_after is non-empty, use the last token of tokens_after; else, if tokens_before non-empty, use its last token.
         - C-tag: join all tokens_before (space‑separated).
         - D-tag: if tokens_after has more than one token, join tokens_after except the last; else, empty.
      6. For "other-general":
         - B-tag: last token of tokens_all (if any).
         - C-tag: join all tokens_all except the last.
         - D-tag: empty.
      7. Finally, if any tag equals the Category, clear it from that bucket.
    
    Returns a tuple: (Category, B-tag, C-tag, D-tag)
    """
    # Step 1: Clean the text.
    text = keyword.lower()
    if seed_keyword:
        pattern = rf'\b{re.escape(seed_keyword.lower())}\b'
        text = re.sub(pattern, '', text)
    # Do not remove omitted words from the raw output!
    omitted_set = set(s.lower() for s in omitted_list)
    
    # Step 2: Create token lists.
    raw_tokens, filtered_tokens = create_token_lists(text, omitted_set)
    
    # Step 3: Search for first occurrence of a candidate A-tag.
    category = ""
    idx_in_raw = None
    for token in filtered_tokens:
        if token in user_a_tags and token not in LOW_PRIORITY:
            # Find its index in raw_tokens (first occurrence that is not omitted)
            for i, rt in enumerate(raw_tokens):
                if rt == token and rt not in omitted_set:
                    idx_in_raw = i
                    category = rt
                    break
            if idx_in_raw is not None:
                break
    if idx_in_raw is None:
        category = "other-general"
    
    # Step 4: Split tokens and assign buckets.
    if category != "other-general":
        tokens_before = raw_tokens[:idx_in_raw]
        tokens_after = raw_tokens[idx_in_raw+1:]
        # B-tag: prefer last token in tokens_after if exists; otherwise, use last token of tokens_before.
        if tokens_after:
            b_tag = tokens_after[-1]
            d_tokens = tokens_after[:-1]
            d_tag = " ".join(d_tokens) if d_tokens else ""
        elif tokens_before:
            b_tag = tokens_before[-1]
            d_tag = ""
        else:
            b_tag = ""
            d_tag = ""
        c_tag = " ".join(tokens_before) if tokens_before else ""
    else:
        # Use all tokens.
        tokens_all = [t for t in raw_tokens if t]  # all non-empty tokens
        if tokens_all:
            b_tag = tokens_all[-1]
            c_tokens = tokens_all[:-1]
            c_tag = " ".join(c_tokens) if c_tokens else ""
        else:
            b_tag = ""
            c_tag = ""
        d_tag = ""
    
    # Step 5: Suppress duplicate values.
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
    2. (Optional) Enter a **Seed Keyword** to be removed from each keyword for classification.
    3. Enter **Omit Phrases** (comma‑separated) that should be skipped for A‑tag detection
       (e.g. "pella, andersen"). They will still appear in the output.
    4. Enter one or more **A:Tags** (comma‑separated) that represent classification keywords
       (e.g. "window, door, price"). These are normalized so that "window" and "windows" match.
       Note: Tokens in LOW_PRIORITY (currently "price", "cost", "comparison", "calculator")
       will be ignored for classification.
       
    For each keyword, the app:
      - Cleans and tokenizes the text.
      - Searches (in a filtered token list) for the first occurrence of any A‑Tag.
         • If found, that token becomes the **Category**.
         • Otherwise, the Category is set to **other-general**.
      - It then splits the raw tokens into those before and after the found A‑Tag and assigns:
          • **B‑Tag:** the last token after the A‑Tag (or from before if none follow).
          • **C‑Tag:** the tokens before the A‑Tag (joined together).
          • **D‑Tag:** the remaining tokens after the A‑Tag (if more than one token exists).
      - Duplicate tokens (if equal to the Category or repeating) are suppressed.
      
    **Examples:**
      - "pella rolscreen window cost" with Omit: "pella" and A‑Tags: "window, door, price"
        → Category: window, B‑Tag: cost, C‑Tag: "pella rolscreen"
      - "pella skylights prices" with Omit: "pella" and A‑Tags: "window, door, price"
        → Category: other-general, B‑Tag: price, C‑Tag: "pella skylight"
      - "pella window cost calculator" with Omit: "pella" and A‑Tags: "window, door, price"
        → Category: window, B‑Tag: calculator, C‑Tag: "pella", D‑Tag: cost
      - "pella vs marvin cost" with Omit: "pella, andersen" and A‑Tags: "window, door, price"
        → Category: other-general, B‑Tag: cost, C‑Tag: "pella v marvin"
    """
)

seed_keyword = st.text_input("Seed Keyword (Optional)", value="")
omit_input = st.text_input("Omit Phrases (comma‑separated)", value="pella, andersen")
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
