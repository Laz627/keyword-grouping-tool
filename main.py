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

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()

# Initialize KeyBERT
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
kw_model = KeyBERT(model=embedding_model)

###
### Helper Functions
###

def normalize_token(token):
    """Lowercase & lemmatize (noun mode), also convert 'vs' -> 'v'."""
    token = token.lower()
    if token == "vs":
        token = "v"
    return lemmatizer.lemmatize(token, pos='n')

def normalize_phrase(phrase):
    """
    Lowercase, tokenize, keep alphanum, lemmatize in noun mode.
    Preserves order so that e.g. 'pella windows cost' => 'pella window cost'
    """
    tokens = word_tokenize(phrase.lower())
    return " ".join(normalize_token(t) for t in tokens if t.isalnum())

def canonicalize_phrase(phrase):
    """
    Remove 'series' tokens, then sort.
    e.g. 'pella 350 series' => '350 pella'
    """
    tokens = word_tokenize(phrase.lower())
    norm = [normalize_token(t) for t in tokens if t.isalnum() and normalize_token(t) != "series"]
    return " ".join(sorted(norm))

def pick_tags_pos_based(tokens, user_a_tags):
    """
    Splits a list of tokens into three tags (A, B, C) with the following logic:
    
      1) A:Tag – Look for the first token that is in the allowed set (user_a_tags).  
         If found, remove it from the candidate token list and use it as A.  
         If not found, set A to "general-other".
         
      2) B:Tag – From the remaining tokens, select the first token whose POS tag starts with
         "JJ" (adjective) or is "VBG" (gerund). If none, take the first remaining token.
         Remove the selected token from the list.
         
      3) C:Tag – The rest of the tokens (joined by a space) or empty if none.
    """
    leftover = tokens[:]  # make a copy
    # --- A:Tag ---
    a_tag = ""
    found = False
    for token in tokens:
        if token in user_a_tags:
            a_tag = token
            leftover.remove(token)
            found = True
            break
    if not found:
        a_tag = "general-other"
        
    # --- B:Tag ---
    b_tag = ""
    if leftover:
        # Use NLTK's POS tagger on the leftover tokens
        pos_list = pos_tag(leftover)
        for i, (w, p) in enumerate(pos_list):
            if p.startswith("JJ") or p == "VBG":
                b_tag = w
                leftover.remove(w)
                break
        if not b_tag and leftover:
            b_tag = leftover.pop(0)
            
    # --- C:Tag ---
    c_tag = " ".join(leftover) if leftover else ""
    
    return (a_tag, b_tag, c_tag)

def classify_keyword_three(keyword, seed, omitted_list, user_a_tags):
    """
    Processes a keyword string as follows:
      1) Removes the seed and any omitted phrases.
      2) Uses KeyBERT to extract the top candidate keyphrase from the remaining text,
         limiting extraction to single-word (1-gram) candidates.
      3) Normalizes and canonicalizes the candidate.
      4) Splits the candidate into tokens and uses pick_tags_pos_based to produce A, B, and C tags.
         If no candidate is found, returns ("general-other", "", "").
    """
    text = keyword.lower()
    if seed:
        pat = rf'\b{re.escape(seed.lower())}\b'
        text = re.sub(pat, '', text)
    for omit in omitted_list:
        pat = rf'\b{re.escape(omit)}\b'
        text = re.sub(pat, '', text)
    text = text.strip()

    # Extract a candidate keyphrase as a 1-gram
    keyphrases = kw_model.extract_keywords(text, keyphrase_ngram_range=(1,1), stop_words='english', top_n=1)
    if not keyphrases:
        return ("general-other", "", "")
    candidate = keyphrases[0][0].lower()
    norm_candidate = normalize_phrase(candidate)
    canon = canonicalize_phrase(norm_candidate)
    if not canon:
        return ("general-other", "", "")
    tokens = canon.split()
    return pick_tags_pos_based(tokens, user_a_tags)

def extract_candidate_themes(keywords_list, top_n):
    """
    Gathers up to top_n candidate keyphrases from each keyword in keywords_list,
    using only single-word (1-gram) extraction.
    """
    all_phrases = []
    for kw in keywords_list:
        kps = kw_model.extract_keywords(kw, keyphrase_ngram_range=(1,1), stop_words='english', top_n=top_n)
        for kp in kps:
            if kp[0]:
                all_phrases.append(kp[0].lower())
    return all_phrases

def group_candidate_themes(all_phrases, min_freq):
    """
    Groups candidate phrases by their canonical form. For each canonical group, if the frequency
    is at least min_freq, selects the most common normalized form as the representative.
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

###
### Post-Processing Realignment (unchanged)
###

def realign_tags_based_on_frequency(df, col_name="B:Tag", other_col="C:Tag"):
    """
    Example approach: 
    1) Gather frequencies of each token in col_name vs. other_col.
    2) If a token appears more often in the other column, reassign it there.
    (This function is unchanged from your original.)
    """
    freq_in_col = Counter()
    freq_in_other = Counter()

    # Parse tokens in each col
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
            if unify_map[t] == col_name:
                new_b_list.append(t)
            else:
                new_o_list.append(t)
        new_b_col.append(" ".join(new_b_list) if new_b_list else "")
        new_o_col.append(" ".join(new_o_list) if new_o_list else "")

    df[col_name] = new_b_col
    df[other_col] = new_o_col

    return df

###
### Streamlit UI
###

st.sidebar.title("Select Mode")
mode = st.sidebar.radio("Choose mode:", ("Candidate Theme Extraction", "Full Tagging"))

if mode == "Candidate Theme Extraction":
    st.title("Candidate Theme Extraction (A, B, C Tags)")
    
    file = st.file_uploader("Upload CSV/Excel with a 'Keywords' column", type=["csv", "xls", "xlsx"])
    nm = st.number_input("Process first N keywords (0 = all)", min_value=0, value=0)
    topn = st.number_input("Keyphrases per keyword", min_value=1, value=3)
    mfreq = st.number_input("Minimum frequency for a candidate theme", min_value=1, value=2)
    clust = st.number_input("Number of clusters (0 = skip)", min_value=0, value=0)
    
    # (Optional) Let the user specify allowed A:Tags
    user_atags_str = st.text_input("User A:Tags (comma-separated)", "door, window")
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
            st.error("File must have a 'Keywords' column.")
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
                    tokens = canon.split()
                    # Use our updated function – note that if no allowed token is found, A:Tag becomes "general-other"
                    a, b, c = pick_tags_pos_based(tokens, user_a_tags)
                    splitted.append((a, b, c))
                cdf["A:Tag"], cdf["B:Tag"], cdf["C:Tag"] = zip(*splitted)
    
                st.dataframe(cdf)
    
                if clust > 0 and len(c_map) >= clust:
                    st.write("### Clusters:")
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
                st.write("No candidate meets the frequency threshold.")

elif mode == "Full Tagging":
    st.title("Full Tagging with A, B, C Tags")
    
    seed = st.text_input("Seed Keyword (optional)")
    omit_str = st.text_input("Omit Phrases (comma-separated)")
    user_atags_str = st.text_input("User A:Tags (comma-separated)", "door, window")
    do_realign = st.checkbox("Enable post-processing realignment of B/C tokens?")
    
    file = st.file_uploader("Upload CSV/Excel with a 'Keywords' column", type=["csv", "xls", "xlsx"])
    
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
            st.error("File must have a 'Keywords' column.")
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
                st.write("Performing post-processing realignment of tokens between B:Tag & C:Tag")
                df = realign_tags_based_on_frequency(df, col_name="B:Tag", other_col="C:Tag")
    
            st.dataframe(df[["Keywords", "A:Tag", "B:Tag", "C:Tag"]])
            csvres = df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Full Tagging CSV", csvres, "full_tagged.csv", "text/csv")
