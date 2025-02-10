import streamlit as st
import pandas as pd
import re
from collections import Counter
import numpy as np

# --- KeyBERT & Sentence Embeddings ---
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

# --- NLTK for Tokenization and Lemmatization ---
import nltk
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()

# Initialize KeyBERT
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
kw_model = KeyBERT(model=embedding_model)

# ------------------------
# HELPER FUNCTIONS
# ------------------------

def normalize_token(token):
    """
    Lowercase and lemmatize a token in noun mode.
    Also, 'vs' -> 'v'.
    """
    token = token.lower()
    if token == "vs":
        token = "v"
    return lemmatizer.lemmatize(token, pos='n')

def normalize_phrase(phrase):
    """
    Tokenize and normalize the phrase (lowercase, lemmatize).
    Preserves token order but lumps singular/plural forms.
    """
    tokens = word_tokenize(phrase.lower())
    return " ".join(normalize_token(t) for t in tokens if t.isalnum())

def canonicalize_phrase(phrase):
    """
    Canonical form for grouping:
    - Remove the token "series" (normalized),
    - Sort the remaining tokens.
    E.g. "pella 350 series" -> "350 pella"
    """
    tokens = word_tokenize(phrase.lower())
    norm = [normalize_token(t) for t in tokens if t.isalnum() and normalize_token(t) != "series"]
    return " ".join(sorted(norm))

def three_tag_split(canon_tokens, user_a_tags):
    """
    Splits tokens into (A_tag, B_tag, C_tag):
      1. If any user A‑tag is found, pick the first that appears; remove it from tokens => A:Tag.
      2. If no forced A:Tag was found, fallback: A:Tag = last token (remove it).
      3. B:Tag = first leftover token (remove it) if any.
      4. C:Tag = everything else joined by space.
    """
    a_tag, b_tag, c_tag = "", "", ""

    # Step 1: Force user A‑tag if present
    for tag in user_a_tags:
        if tag in canon_tokens:
            a_tag = tag
            canon_tokens.remove(tag)
            break

    # Fallback if no forced A‑tag:
    if not a_tag and canon_tokens:
        a_tag = canon_tokens[-1]
        canon_tokens = canon_tokens[:-1]

    # Step 2: B‑tag => first leftover if any
    if canon_tokens:
        b_tag = canon_tokens[0]
        canon_tokens = canon_tokens[1:]

    # Step 3: rest => c_tag
    if canon_tokens:
        c_tag = " ".join(canon_tokens)

    return (a_tag, b_tag, c_tag)

# ------------------------
# FULL TAGGING HELPER
# ------------------------

def classify_keyword_three(keyword, seed_keyword, omitted_list, user_a_tags):
    """
    For a single keyword:
      1) remove seed & omitted phrases,
      2) KeyBERT => top candidate phrase,
      3) normalize & canonicalize,
      4) split tokens => A,B,C.
    If no candidate => ("other-general","","").
    """
    text = keyword.lower()
    # Remove seed
    if seed_keyword:
        pattern = rf'\b{re.escape(seed_keyword.lower())}\b'
        text = re.sub(pattern, '', text)
    # Remove omitted
    for omit in omitted_list:
        pat = rf'\b{re.escape(omit)}\b'
        text = re.sub(pat, '', text)
    text = text.strip()

    # Extract top candidate
    keyphrases = kw_model.extract_keywords(text, keyphrase_ngram_range=(1,3), stop_words='english', top_n=1)
    if not keyphrases:
        return ("other-general","","")

    candidate = keyphrases[0][0].lower()
    norm_candidate = normalize_phrase(candidate)
    canon = canonicalize_phrase(norm_candidate)
    if not canon:
        return ("other-general","","")

    # turn canonical string into list of tokens
    tokens = canon.split()
    return three_tag_split(tokens, user_a_tags)

# ------------------------
# CANDIDATE THEME EXTRACTION
# ------------------------

def extract_candidate_themes(keywords_list, top_n):
    """
    For each keyword, extract up to top_n keyphrases (1..3-grams).
    Return all extracted phrases as a list (lowercased).
    """
    all_phrases = []
    for kw in keywords_list:
        keyphrases = kw_model.extract_keywords(kw, keyphrase_ngram_range=(1,3), stop_words='english', top_n=top_n)
        extracted = [kp[0].lower() for kp in keyphrases if kp[0]]
        all_phrases.extend(extracted)
    return all_phrases

def group_candidate_themes(all_phrases, min_freq):
    """
    Group by canonical form. Return {rep: freq}.
    rep = most common normalized string in that canonical group.
    """
    grouped = {}
    for phrase in all_phrases:
        norm = normalize_phrase(phrase)
        canon = canonicalize_phrase(norm)
        if canon:
            grouped.setdefault(canon, []).append(norm)

    candidate_themes = {}
    for canon, sublist in grouped.items():
        freq = len(sublist)
        if freq >= min_freq:
            rep = Counter(sublist).most_common(1)[0][0]
            candidate_themes[rep] = freq
    return candidate_themes

# ------------------------
# STREAMLIT UI
# ------------------------

st.sidebar.title("Select Mode")
mode = st.sidebar.radio("Choose an option:", ("Candidate Theme Extraction", "Full Tagging"))

if mode == "Candidate Theme Extraction":
    st.title("Generic Candidate Theme Extraction with KeyBERT")
    st.write("""
    **This mode** extracts all candidate keyphrases from your keyword list, 
    normalizes & canonicalizes them to group similar phrases, 
    and shows their frequency. Then it recommends a default 3‑tag split 
    (A:Tag, B:Tag, C:Tag) with no domain-specific forcing.
    """)

    file = st.file_uploader("Upload CSV/Excel with a 'Keywords' column", type=["csv","xls","xlsx"])
    num_keywords = st.number_input("Process first N keywords (0=all)", min_value=0, value=0)
    top_n = st.number_input("Keyphrases per keyword", min_value=1, value=3)
    min_freq = st.number_input("Min frequency for candidate theme", min_value=1, value=2)
    cluster_count = st.number_input("Number of clusters (0=skip)", min_value=0, value=0)

    if file:
        try:
            if file.name.endswith('.csv'):
                df = pd.read_csv(file)
            else:
                df = pd.read_excel(file)
        except Exception as e:
            st.error(f"Error reading file: {e}")
            st.stop()

        if "Keywords" not in df.columns:
            st.error("File must have a 'Keywords' column.")
        else:
            keywords_list = df["Keywords"].tolist()
            if num_keywords>0:
                keywords_list = keywords_list[:num_keywords]

            st.write("Extracting candidate keyphrases using KeyBERT...")
            all_phrases = extract_candidate_themes(keywords_list, top_n)
            candidate_map = group_candidate_themes(all_phrases, min_freq)

            if candidate_map:
                cdf = pd.DataFrame(list(candidate_map.items()), columns=["Representative Theme","Frequency"])
                cdf = cdf.sort_values(by="Frequency", ascending=False)

                # For demonstration, we do a default split with user_a_tags = {} 
                splitted = []
                for theme in cdf["Representative Theme"]:
                    canon = canonicalize_phrase(normalize_phrase(theme))
                    tokens = canon.split()
                    a,b,c = three_tag_split(tokens, user_a_tags=set())
                    splitted.append((a,b,c))

                cdf["A:Tag (Recommended)"] = [s[0] for s in splitted]
                cdf["B:Tag (Recommended)"] = [s[1] for s in splitted]
                cdf["C:Tag (Recommended)"] = [s[2] for s in splitted]

                st.dataframe(cdf)

                # optional clustering
                if cluster_count>0 and len(candidate_map)>=cluster_count:
                    st.write("### Clusters:")
                    reps = list(candidate_map.keys())
                    embeddings = embedding_model.encode(reps)
                    km = KMeans(n_clusters=cluster_count, random_state=42)
                    labs = km.fit_predict(embeddings)
                    clusters = {}
                    for lab,rep in zip(labs,reps):
                        clusters.setdefault(lab,[]).append(rep)
                    for lab,group in clusters.items():
                        st.write(f"**Cluster {lab}:** {', '.join(group)}")
                
                csvdata = cdf.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "Download Candidate Theme CSV",
                    data=csvdata,
                    file_name="candidate_theme_recommendations.csv",
                    mime="text/csv"
                )
            else:
                st.write("No candidate themes met the frequency threshold.")

elif mode=="Full Tagging":
    st.title("Generic Full Tagging (3-Tag) w/ User A-Tags")
    st.write("""
    For each keyword:
    1) Remove seed + omitted text
    2) Extract top candidate via KeyBERT
    3) Normalize + canonicalize
    4) If user A-tag is found, that becomes A:Tag
    5) Else fallback to last token as A:Tag
    6) B:Tag = first leftover
    7) C:Tag = rest
    """)

    seed = st.text_input("Seed Keyword (optional)")
    omit_input = st.text_input("Omit Phrases (comma-separated)")
    user_atags_str = st.text_input("User A-Tags (comma-separated)", "door, window")

    file = st.file_uploader("Upload CSV/Excel with 'Keywords' column", type=["csv","xls","xlsx"])

    if file:
        try:
            if file.name.endswith('.csv'):
                df = pd.read_csv(file)
            else:
                df = pd.read_excel(file)
        except Exception as e:
            st.error(f"Error reading file: {e}")
            st.stop()

        if "Keywords" not in df.columns:
            st.error("File must have a 'Keywords' column.")
        else:
            omitted_list = [o.strip().lower() for o in omit_input.split(",") if o.strip()]
            user_a_tags = set(normalize_token(a.strip()) for a in user_atags_str.split(",") if a.strip())

            a_tags, b_tags, c_tags = [], [], []

            for kw in df["Keywords"]:
                a,b,c = classify_keyword_three(kw, seed, omitted_list, user_a_tags)
                a_tags.append(a)
                b_tags.append(b)
                c_tags.append(c)

            df["A:Tag"] = a_tags
            df["B:Tag"] = b_tags
            df["C:Tag"] = c_tags

            st.dataframe(df[["Keywords","A:Tag","B:Tag","C:Tag"]])

            csvdata = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "Download Full Tagging CSV",
                data=csvdata,
                file_name="full_tagging_output.csv",
                mime="text/csv"
            )
