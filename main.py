import streamlit as st
import pandas as pd
import re
from nltk import word_tokenize
import nltk
from nltk.stem import WordNetLemmatizer

# Download required NLTK data
nltk.download('punkt')
nltk.download('wordnet')

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Define a small set of stopwords for our heuristic.
STOPWORDS = {"the", "a", "an", "are", "is", "of", "and", "or", "how", "much", "do", "does", "be", "in", "to", "for", "on", "at"}

def omit_phrases(text, omitted_list):
    """
    Lowercases the text and removes any occurrence of words in omitted_list.
    """
    result = text.lower()
    for omit in omitted_list:
        pattern = rf'\b{re.escape(omit.lower())}\b'
        result = re.sub(pattern, '', result)
    return result.strip()

def normalize_token(token):
    """
    Lowercase and lemmatize a token (using noun mode).
    """
    return lemmatizer.lemmatize(token.lower(), pos='n')

def process_keyword_order(keyword, seed_keyword, omitted_list, user_a_tags):
    """
    Processes a single keyword string based on word order.
    
    Steps:
      1. Lowercase the keyword.
      2. Remove the seed keyword (if provided) and any omitted phrases.
      3. Tokenize and normalize each token.
      4. Search for the first token that appears in the user-specified A:Tag set.
         • If found, that normalized token becomes the Category.
         • If not found, Category is "other-general" and other tags are empty.
      5. From the remaining tokens (ignoring stopwords and tokens equal to the Category),
         determine:
         - B:Tag: if tokens after the A:tag exist, choose the last token; else, if tokens before exist, choose the last.
         - C:Tag: join all tokens before the A:tag.
         - D:Tag: join all tokens after the A:tag except for the token chosen for B:Tag.
      6. Ensure that the same token does not appear in more than one bucket.
    
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
    norm_tokens = [normalize_token(t) for t in tokens if t.isalnum()]  # keep only alphanumeric tokens

    # Step 4: Find the first occurrence of any user-specified A:Tag.
    idx_A = None
    category = ""
    for i, token in enumerate(norm_tokens):
        if token in user_a_tags:
            idx_A = i
            category = token  # use the normalized token as category
            break
    if idx_A is None:
        # No A:Tag found.
        return ("other-general", "", "", "")
    
    # Step 5: Remove tokens equal to the category from further processing.
    tokens_before = [t for t in norm_tokens[:idx_A] if t not in STOPWORDS and t != category]
    tokens_after = [t for t in norm_tokens[idx_A+1:] if t not in STOPWORDS and t != category]
    
    # Step 6: Determine B:Tag.
    if tokens_after:
        b_tag = tokens_after[-1]
    elif tokens_before:
        b_tag = tokens_before[-1]
    else:
        b_tag = ""
    # Ensure B:Tag is not the same as category.
    if b_tag == category:
        b_tag = ""
    
    # Step 7: Determine C:Tag as all tokens before the A:tag.
    c_tag = " ".join(tokens_before) if tokens_before else ""
    
    # Step 8: Determine D:Tag as tokens after the A:tag excluding the B:Tag.
    if tokens_after:
        d_tokens = tokens_after[:-1]  # remove the last token used for B:Tag
        d_tag = " ".join(d_tokens) if d_tokens else ""
    else:
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
    3. Enter **Omit Phrases** (comma‑separated) that should be removed (for example, "pella, andersen").
    4. Enter one or more **A:Tags** (comma‑separated) that represent the classification keywords.
       (For example, "window, door, price").
    
    The app will process each keyword by:
      - Removing the seed and omitted words.
      - Tokenizing and normalizing (so singular and plural forms match).
      - Searching for the first occurrence of any A:Tag.
         • If found, that token becomes the **Category**.
         • The token immediately following (if available) becomes the **B:Tag**.
         • All tokens before the A:Tag (if any) become the **C:Tag**.
         • All tokens after the A:Tag except the one chosen for B:Tag become the **D:Tag**.
      - If no A:Tag is found, the Category is set to **other-general**.
      
    The same tag will not be repeated across multiple columns.
    """
)

seed_keyword = st.text_input("Seed Keyword (Optional)", value="")
omit_input = st.text_input("Omit Phrases (comma‑separated)", value="")
a_tags_input = st.text_input("A:Tags (comma‑separated)", value="window, door, price")
uploaded_file = st.file_uploader("Upload CSV/Excel File", type=["csv", "xls", "xlsx"])

if uploaded_file:
    # Load the file.
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    
    # Process user inputs.
    omitted_list = [s.strip() for s in omit_input.split(",") if s.strip()]
    # Normalize each provided A:Tag.
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
