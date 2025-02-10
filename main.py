import streamlit as st
import pandas as pd
import re
from collections import Counter
import numpy as np

# --- KeyBERT & Sentence Embeddings ---
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

# --- NLTK for Tokenization, POS tagging, and Lemmatization ---
import nltk
from nltk import word_tokenize, pos_tag
from nltk.stem import WordNetLemmatizer

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

lemmatizer = WordNetLemmatizer()

# Initialize KeyBERT
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
kw_model = KeyBERT(model=embedding_model)

###
### Helper Functions
###

def normalize_token(token):
    """Lowercase & lemmatize (noun mode); also convert 'vs' -> 'v'."""
    token = token.lower()
    if token == "vs":
        token = "v"
    return lemmatizer.lemmatize(token, pos='n')

def normalize_phrase(phrase):
    """
    Lowercase, tokenize, keep alphanum, and lemmatize in noun mode.
    For example, 'pella windows cost' becomes 'pella window cost'.
    """
    tokens = word_tokenize(phrase.lower())
    return " ".join(normalize_token(t) for t in tokens if t.isalnum())

def canonicalize_phrase(phrase):
    """
    Remove unwanted tokens (e.g. "series") while preserving original order.
    Also replace underscores with spaces so that tokens like "sash_replacement"
    become "sash replacement". For example, 'pella 350 series' becomes 'pella 350'.
    """
    tokens = word_tokenize(phrase.lower())
    norm = [normalize_token(t) for t in tokens if t.isalnum() and normalize_token(t) != "series"]
    return " ".join(norm).replace("_", " ")

def pick_tags_pos_based(tokens, user_a_tags):
    """
    Given a list of candidate tokens (in original order), assign one-word tags for A, B, and C as follows:
    
    1. A:Tag  
       - Flatten the tokens (splitting on whitespace) and scan for one that “contains” an allowed A:tag
         (or vice-versa). If found, remove that token and set A:Tag to the allowed tag.
       - If none is found, force A:Tag to "general-other" (without removing any token).
    
    2. B:Tag and C:Tag  
       - From the remaining tokens, filter out any stopwords.
       - If at least two tokens remain, assign B:Tag = first token and C:Tag = second token.
         Then, use POS tagging on these two tokens: if the first token is not an adjective/gerund
         but the second is, swap them.
       - If only one token remains, assign it to B:Tag and leave C:Tag blank.
    """
    # Flatten tokens in case any token contains embedded whitespace.
    flat_tokens = []
    for token in tokens:
        flat_tokens.extend(token.split())
    tokens_copy = flat_tokens[:]  # work on a copy

    a_tag = None
    a_index = None
    for i, token in enumerate(tokens_copy):
        for allowed in user_a_tags:
            if allowed in token or token in allowed:
                a_tag = allowed
                a_index = i
                break
        if a_tag is not None:
            break

    if a_tag is not None:
        tokens_copy.pop(a_index)
    else:
        a_tag = "general-other"

    # Filter out stopwords from the remaining tokens.
    filtered = [t for t in tokens_copy if t.lower() not in stop_words and t.strip() != ""]

    if len(filtered) >= 2:
        b_tag, c_tag = filtered[0], filtered[1]
        pos_tags = pos_tag([b_tag, c_tag])
        # If the first token is not an adjective/gerund but the second is, swap them.
        if not (pos_tags[0][1].startswith("JJ") or pos_tags[0][1] == "VBG") and \
           (pos_tags[1][1].startswith("JJ") or pos_tags[1][1] == "VBG"):
            b_tag, c_tag = c_tag, b_tag
    elif len(filtered) == 1:
        b_tag = filtered[0]
        c_tag = ""
    else:
        b_tag = ""
        c_tag = ""

    return a_tag, b_tag, c_tag

def classify_keyword_three(keyword, seed, omitted_list, user_a_tags):
    """
    Process a keyword string as follows:
      1) Remove the seed (if provided) and any omitted phrases.
      2) Use KeyBERT to extract the top candidate keyphrase from the remaining text,
         using an n-gram range of (1,4) (up to four words).
      3) Normalize and canonicalize the candidate while preserving word order.
      4) Split the candidate into individual tokens and use pick_tags_pos_based() to assign A, B, and C tags.
         If no candidate is found, return ("general-other", "", "").
    """
    text = keyword.lower()
    if seed:
        pat = rf'\b{re.escape(seed.lower())}\b'
        text = re.sub(pat, '', text)
    for omit in omitted_list:
        pat = rf'\b{re.escape(omit)}\b'
        text = re.sub(pat, '', text)
    text = text.strip()

    keyphrases = kw_model.extract_keywords(text, keyphrase_ngram_range=(1,4), stop_words='english', top_n=1)
    if not keyphrases:
        return ("general-other", "", "")
    candidate = keyphrases[0][0].lower()
    norm_candidate = normalize_phrase(candidate)
    canon = canonicalize_phrase(norm_candidate)
    if not canon:
        return ("general-other", "", "")
    tokens = [t for t in canon.split() if t.strip() != ""]
    return pick_tags_pos_based(tokens, user_a_tags)

def extract_candidate_themes(keywords_list, top_n):
    """
    Gather up to top_n candidate keyphrases from each keyword in keywords_list,
    using an n-gram range of (1,4). Each candidate phrase is later split into tokens.
    """
    all_phrases = []
    for kw in keywords_list:
        kps = kw_model.extract_keywords(kw, keyphrase_ngram_range=(1,4), stop_words='english', top_n=top_n)
        for kp in kps:
            if kp[0]:
                all_phrases.append(kp[0].lower())
    return all_phrases

def group_candidate_themes(all_phrases, min_freq):
    """
    Group candidate phrases by their canonical form. For each group that meets the frequency
    threshold (min_freq), select the most common normalized form as the representative.
    Returns a dictionary mapping the representative phrase to its frequency.
    """
    grouped = {}
    for phr in all_phrases:
        norm = normalize_phrase(phr)
        canon = canonicalize_phrase(norm)
        if canon:
            grouped.setdefault(canon, []).append(norm)
    candidate_map = {}
    for canon, arr in grouped.items():
        freq = len(arr)
        if freq >= min_freq:
            rep = Counter(arr).most_common(1)[0][0]
            candidate_map[rep] = freq
    return candidate_map

def realign_tags_based_on_frequency(df, col_name="B:Tag", other_col="C:Tag"):
    """
    Post-processing re-alignment:
      1) Gather token frequencies in the specified columns.
      2) For each row, reassign tokens based on overall frequency.
      3) Ensure that each cell gets only one token (by picking the first token after re-assignment).
    """
    freq_in_col = Counter()
    freq_in_other = Counter()

    for i, row in df.iterrows():
        bval = row[col_name]
        oval = row[other_col]
        if bval:
            for token in bval.split():
                freq_in_col[token] += 1
        if oval:
            for token in oval.split():
                freq_in_other[token] += 1

    unify_map = {}
    all_tokens = set(freq_in_col.keys()) | set(freq_in_other.keys())
    for tok in all_tokens:
        c_freq = freq_in_col[tok]
        o_freq = freq_in_other[tok]
        if o_freq > c_freq:
            unify_map[tok] = other_col
        else:
            unify_map[tok] = col_name

    new_b_col, new_o_col = [], []
    for i, row in df.iterrows():
        b_tokens = row[col_name].split() if row[col_name] else []
        o_tokens = row[other_col].split() if row[other_col] else []
        combined = [(t, "b") for t in b_tokens] + [(t, "o") for t in o_tokens]
        new_b_list = []
        new_o_list = []
        for (t, orig) in combined:
            if unify_map.get(t, col_name) == col_name:
                new_b_list.append(t)
            else:
                new_o_list.append(t)
        new_b_col.append(new_b_list[0] if new_b_list else "")
        new_o_col.append(new_o_list[0] if new_o_list else "")
    df[col_name] = new_b_col
    df[other_col] = new_o_col
    return df

###
### Streamlit UI with Enhanced Instructions
###

# Sidebar instructions
st.sidebar.title("Keyword Tagging Tool")
st.sidebar.markdown("""
This tool allows you to extract candidate themes and perform full tagging on keywords using a three-tag system.

**Modes:**
- **Candidate Theme Extraction:** Extract candidate themes from your keywords and see a preview of the A, B, and C tags.
- **Full Tagging:** Process the entire file and assign A:Tag (e.g. door/window), B:Tag, and C:Tag to each keyword.

**Instructions:**
1. Upload a CSV/Excel file that contains a column named **Keywords**.
2. Use the settings to limit processing or adjust n-gram extraction.
3. Specify allowed A:Tags (e.g. "door, window") to force categorization.
4. Optionally, enable post-processing re-alignment for consistent one-word tagging.
""")

# Radio buttons to choose mode
mode = st.sidebar.radio("Select Mode", ("Candidate Theme Extraction", "Full Tagging"))

if mode == "Candidate Theme Extraction":
    st.title("Candidate Theme Extraction (Preview of A, B, C Tags)")
    st.markdown("""
    **How It Works:**
    - The tool extracts candidate keyphrases from each keyword using KeyBERT with an n-gram range of (1,4).
    - It then normalizes and canonicalizes these phrases.
    - The tagging algorithm:
       1. Searches for an allowed A:Tag (e.g. "door" or "window") and removes it.
       2. Assigns the first two remaining tokens as B:Tag and C:Tag (with a possible swap based on part-of-speech).
    - You will see a table of candidate themes with their frequency and the resulting tags.
    """)
    file = st.file_uploader("Upload CSV/Excel file (must contain a 'Keywords' column)", type=["csv", "xls", "xlsx"])
    nm = st.number_input("Process first N keywords (0 for all)", min_value=0, value=0)
    topn = st.number_input("Keyphrases per keyword", min_value=1, value=3)
    mfreq = st.number_input("Minimum frequency for a candidate theme", min_value=1, value=2)
    clust = st.number_input("Number of clusters (0 to skip clustering)", min_value=0, value=0)
    
    # Allowed A:Tags
    user_atags_str = st.text_input("Specify allowed A:Tags (comma-separated)", "door, window")
    user_a_tags = set(normalize_token(x.strip()) for x in user_atags_str.split(",") if x.strip())
    
    if file:
        try:
            if file.name.endswith(".csv"):
                df = pd.read_csv(file)
            else:
                df = pd.read_excel(file)
        except Exception as e:
            st.error(f"Error reading file: {e}")
            st.stop()
    
        if "Keywords" not in df.columns:
            st.error("The file must have a 'Keywords' column.")
        else:
            kw_list = df["Keywords"].tolist()
            if nm > 0:
                kw_list = kw_list[:nm]
    
            all_phrases = extract_candidate_themes(kw_list, topn)
            c_map = group_candidate_themes(all_phrases, mfreq)
    
            if c_map:
                cdf = pd.DataFrame(list(c_map.items()), columns=["Candidate Theme", "Frequency"])
                cdf = cdf.sort_values(by="Frequency", ascending=False)
                splitted = []
                for theme in cdf["Candidate Theme"]:
                    norm = normalize_phrase(theme)
                    canon = canonicalize_phrase(norm)
                    tokens = [t for t in canon.split() if t.strip() != ""]
                    a, b, c = pick_tags_pos_based(tokens, user_a_tags)
                    splitted.append((a, b, c))
                cdf["A:Tag"], cdf["B:Tag"], cdf["C:Tag"] = zip(*splitted)
    
                st.dataframe(cdf)
    
                if clust > 0 and len(c_map) >= clust:
                    st.markdown("### Clustering of Candidate Themes")
                    reps = list(c_map.keys())
                    emb = embedding_model.encode(reps)
                    km = KMeans(n_clusters=clust, random_state=42)
                    labs = km.fit_predict(emb)
                    cluster_dict = {}
                    for lab, rep in zip(labs, reps):
                        cluster_dict.setdefault(lab, []).append(rep)
                    for lab, arr in cluster_dict.items():
                        st.write(f"**Cluster {lab}:** {', '.join(arr)}")
    
                csvd = cdf.to_csv(index=False).encode('utf-8')
                st.download_button("Download Candidate CSV", csvd, "candidate_themes.csv", "text/csv")
            else:
                st.write("No candidate themes meet the frequency threshold.")
                
elif mode == "Full Tagging":
    st.title("Full Tagging (Assigning A, B, C Tags to Keywords)")
    st.markdown("""
    **How It Works:**
    - The tool processes each keyword by first removing any specified seed or omitted phrases.
    - It then extracts a candidate keyphrase using KeyBERT (n-gram range: 1-4) and normalizes it.
    - The tagging algorithm then:
       1. Searches for an allowed A:Tag (e.g. "door" or "window") and removes it.
       2. Uses the first two remaining non-stopword tokens for B:Tag and C:Tag.
       3. If necessary, swaps B and C based on part-of-speech (e.g., adjectives should be in B).
    - A post-processing re-alignment step (optional) ensures each tag cell contains only one word.
    """)
    seed = st.text_input("Seed Keyword (optional)", "")
    omit_str = st.text_input("Omit Phrases (comma-separated)", "")
    user_atags_str = st.text_input("Specify allowed A:Tags (comma-separated)", "door, window")
    do_realign = st.checkbox("Enable post-processing re-alignment for B/C tags?", value=True)
    
    file = st.file_uploader("Upload CSV/Excel file (must contain a 'Keywords' column)", type=["csv", "xls", "xlsx"])
    
    if file:
        try:
            if file.name.endswith(".csv"):
                df = pd.read_csv(file)
            else:
                df = pd.read_excel(file)
        except Exception as e:
            st.error(f"Error reading file: {e}")
            st.stop()
    
        if "Keywords" not in df.columns:
            st.error("The file must have a 'Keywords' column.")
        else:
            omitted_list = [x.strip().lower() for x in omit_str.split(",") if x.strip()]
            user_a_tags = set(normalize_token(x.strip()) for x in user_atags_str.split(",") if x.strip())
    
            A_list, B_list, C_list = [], [], []
            for kw in df["Keywords"]:
                a, b, c = classify_keyword_three(kw, seed, omitted_list, user_a_tags)
                A_list.append(a)
                B_list.append(b)
                C_list.append(c)
    
            df["A:Tag"] = A_list
            df["B:Tag"] = B_list
            df["C:Tag"] = C_list
    
            if do_realign:
                st.markdown("### Post-Processing Re-Alignment")
                st.write("This step reassigns tokens based on overall frequency while ensuring that each tag cell contains only one word.")
                df = realign_tags_based_on_frequency(df, col_name="B:Tag", other_col="C:Tag")
    
            st.dataframe(df[["Keywords", "A:Tag", "B:Tag", "C:Tag"]])
            csvres = df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Full Tagging CSV", csvres, "full_tagged.csv", "text/csv")
