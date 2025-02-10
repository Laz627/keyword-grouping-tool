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
    'pella 350 series' => '350 pella'
    """
    tokens = word_tokenize(phrase.lower())
    norm = [normalize_token(t) for t in tokens if t.isalnum() and normalize_token(t) != "series"]
    return " ".join(sorted(norm))

def pick_single_tag(tokens, user_a_tags):
    """
    Given a list of tokens (already normalized), return the first token
    that appears in user_a_tags. If none, return "general-other".
    """
    for token in tokens:
        if token in user_a_tags:
            return token
    return "general-other"

def classify_keyword_single(keyword, seed, omitted_list, user_a_tags):
    """
    Process a keyword:
      1) Remove seed and omitted phrases.
      2) Use KeyBERT to extract a top candidate using only 1‑grams.
      3) Normalize the candidate token.
      4) If it is in user_a_tags, return it; else return "general-other".
    """
    text = keyword.lower()
    if seed:
        pat = rf'\b{re.escape(seed.lower())}\b'
        text = re.sub(pat, '', text)
    for omit in omitted_list:
        pat = rf'\b{re.escape(omit)}\b'
        text = re.sub(pat, '', text)
    text = text.strip()
    
    # Extract candidate with only single-word keyphrases
    keyphrases = kw_model.extract_keywords(text, keyphrase_ngram_range=(1,1), stop_words='english', top_n=1)
    if not keyphrases:
        return "general-other"
    candidate = keyphrases[0][0].lower().strip()
    candidate = normalize_token(candidate)
    if candidate in user_a_tags:
        return candidate
    else:
        return "general-other"

def extract_candidate_themes(keywords_list, top_n):
    """
    Gather up to top_n keyphrases from each keyword using only 1‑grams.
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
    Group by canonical form => pick frequency => return {rep: freq}.
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
### (Optional) Post-Processing Realignment – no longer used with a single tag
###
# (The following function is no longer needed since we only produce a single tag.)
# def realign_tags_based_on_frequency(df, col_name="B:Tag", other_col="C:Tag"):
#     ...

###
### Streamlit UI
###

st.sidebar.title("Select Mode")
mode = st.sidebar.radio("Choose mode:", ("Candidate Theme Extraction", "Full Tagging"))

if mode == "Candidate Theme Extraction":
    st.title("Candidate Theme Extraction (Single-Tag)")
    
    file = st.file_uploader("Upload CSV/Excel with 'Keywords' column", type=["csv", "xls", "xlsx"])
    nm = st.number_input("Process first N keywords (0=all)", min_value=0, value=0)
    topn = st.number_input("Keyphrases per keyword", min_value=1, value=3)
    mfreq = st.number_input("Minimum frequency for a candidate theme", min_value=1, value=2)
    clust = st.number_input("Number of clusters (0=skip)", min_value=0, value=0)
    
    # Add a text input for allowed tags (User A:Tags)
    user_atags_str = st.text_input("User A:Tags (comma-separated)", "door, window")
    user_a_tags = set(normalize_token(x.strip()) for x in user_atags_str.split(",") if x.strip())
    
    if file:
        # load file
        try:
            if file.name.endswith(".csv"):
                df = pd.read_csv(file)
            else:
                df = pd.read_excel(file)
        except Exception as e:
            st.error(f"Error reading file: {e}")
            st.stop()
    
        if "Keywords" not in df.columns:
            st.error("File must have 'Keywords' column.")
        else:
            kw_list = df["Keywords"].tolist()
            if nm > 0:
                kw_list = kw_list[:nm]
    
            all_phrases = extract_candidate_themes(kw_list, topn)
            c_map = group_candidate_themes(all_phrases, mfreq)
    
            if c_map:
                cdf = pd.DataFrame(list(c_map.items()), columns=["Candidate Theme", "Frequency"])
                cdf = cdf.sort_values(by="Frequency", ascending=False)
                
                # Instead of splitting into A/B/C, simply pick a single tag
                tags = []
                for theme in cdf["Candidate Theme"]:
                    norm = normalize_phrase(theme)
                    canon = canonicalize_phrase(norm)
                    tokens = canon.split()
                    tag = pick_single_tag(tokens, user_a_tags)
                    tags.append(tag)
                cdf["Tag"] = tags
                
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
                st.write("No candidate meets frequency threshold.")

elif mode == "Full Tagging":
    st.title("Full Tagging (Single Tag per Keyword)")
    
    seed = st.text_input("Seed Keyword (optional)")
    omit_str = st.text_input("Omit Phrases (comma-separated)")
    user_atags_str = st.text_input("User A:Tags (comma-separated)", "door, window")
    # With only one tag, the realignment option is no longer applicable.
    # do_realign = st.checkbox("Enable post-processing realignment of tokens? (Not used)")
    
    file = st.file_uploader("Upload CSV/Excel with 'Keywords'", type=["csv", "xls", "xlsx"])
    
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
            st.error("File must have 'Keywords' column.")
        else:
            omitted_list = [x.strip().lower() for x in omit_str.split(",") if x.strip()]
            user_a_tags = set(normalize_token(x.strip()) for x in user_atags_str.split(",") if x.strip())
    
            # Process each keyword to assign a single tag.
            tag_list = []
            for kw in df["Keywords"]:
                tag = classify_keyword_single(kw, seed, omitted_list, user_a_tags)
                tag_list.append(tag)
    
            df["Tag"] = tag_list
    
            st.dataframe(df[["Keywords", "Tag"]])
            csvres = df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Full Tagging CSV", csvres, "full_tagged.csv", "text/csv")
