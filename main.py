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

# ------------------------
# HELPER FUNCTIONS
# ------------------------

def normalize_token(token):
    """
    Lowercase and lemmatize a token (noun mode). Also converts 'vs' -> 'v'.
    """
    token = token.lower()
    if token == "vs":
        token = "v"
    return lemmatizer.lemmatize(token, pos='n')

def normalize_phrase(phrase):
    """
    Tokenize and normalize the phrase (lowercasing, lemmatizing). Preserves word order,
    which groups singular/plural forms but doesn't reorder tokens.
    """
    tokens = word_tokenize(phrase.lower())
    return " ".join(normalize_token(t) for t in tokens if t.isalnum())

def canonicalize_phrase(phrase):
    """
    Produces a canonical form for grouping. Removes token 'series' and sorts the rest.
    So 'pella 350 series' -> '350 pella', unifying variants.
    """
    tokens = word_tokenize(phrase.lower())
    norm = [normalize_token(t) for t in tokens if t.isalnum() and normalize_token(t) != "series"]
    return " ".join(sorted(norm))

def split_tokens_pos_based(tokens, user_a_tags):
    """
    Splits the token list into A, B, and C with a POS-based approach:
      1) If any user-supplied A‑tag is found, pick that as A:Tag and remove it;
         else fallback to the last token.
      2) From leftover, pick the *first adjective or gerund* (POS = JJ/JJR/JJS or VBG) as B:Tag.
         If none, pick the first leftover token as B:Tag.
      3) The rest => C:Tag (joined by space).
    """
    # We'll keep a copy for POS tagging
    leftover = tokens[:]

    # Step 1: Force user A:Tag if present
    a_tag = ""
    for tag in user_a_tags:
        if tag in leftover:
            a_tag = tag
            leftover.remove(tag)
            break
    # If no forced A:Tag, fallback to the last leftover token
    if not a_tag and leftover:
        a_tag = leftover[-1]
        leftover = leftover[:-1]

    # Step 2: B:Tag => the first token with POS in [JJ, JJR, JJS, VBG], else the first leftover
    b_tag = ""
    if leftover:
        # We'll POS-tag these leftover tokens
        # Reconstruct them as text for tagging
        leftover_text = " ".join(leftover)
        pos_result = pos_tag(leftover_text.split())  # simple usage
        # pos_result is list of (word, tag)
        # We'll find the first with tag starting with 'JJ' or 'VBG'
        # but also watch the leftover index carefully
        token_indices_by_pos = {}
        for i, (word, tag) in enumerate(pos_result):
            # naive matching
            if tag.startswith("JJ") or tag == "VBG":
                # found our B:Tag
                b_tag = word
                # remove from leftover
                # we find the actual index in leftover that matches 'word'
                # (in case there are duplicates)
                if word in leftover:
                    leftover.remove(word)
                break
        # If still no B:Tag => pick the first leftover
        if not b_tag and leftover:
            b_tag = leftover[0]
            leftover = leftover[1:]

    # Step 3: The remainder => c
    c_tag = " ".join(leftover) if leftover else ""
    return (a_tag, b_tag, c_tag)


def classify_keyword_three(keyword, seed_keyword, omitted_list, user_a_tags):
    """
    For a single keyword:
     1) Remove seed & omit phrases
     2) Extract top candidate phrase w/ KeyBERT
     3) Normalize & canonicalize
     4) Split tokens using the POS-based approach
    If no candidate, => ("other-general","","").
    """
    text = keyword.lower()
    # remove seed
    if seed_keyword:
        pattern = rf'\b{re.escape(seed_keyword.lower())}\b'
        text = re.sub(pattern, '', text)
    # remove omitted
    for omit in omitted_list:
        pattern = rf'\b{re.escape(omit.lower())}\b'
        text = re.sub(pattern, '', text)
    text = text.strip()

    # top_n=1
    keyphrases = kw_model.extract_keywords(text, keyphrase_ngram_range=(1,3), stop_words='english', top_n=1)
    if not keyphrases:
        return ("other-general","","")

    candidate = keyphrases[0][0].lower()
    norm_candidate = normalize_phrase(candidate)
    canon = canonicalize_phrase(norm_candidate)
    if not canon:
        return ("other-general","","")

    tokens = canon.split()
    return split_tokens_pos_based(tokens, user_a_tags)

# -------------------------------------------
# Candidate theme extraction for all keyphrases
# -------------------------------------------

def extract_candidate_themes(keywords_list, top_n):
    """
    Use KeyBERT to get up to top_n keyphrases for each keyword (1..3-gram).
    Return a combined list of extracted phrases.
    """
    all_phrases = []
    for kw in keywords_list:
        keyphrases = kw_model.extract_keywords(kw, keyphrase_ngram_range=(1,3), stop_words='english', top_n=top_n)
        extracted = [kp[0].lower() for kp in keyphrases if kp[0]]
        all_phrases.extend(extracted)
    return all_phrases

def group_candidate_themes(all_phrases, min_freq):
    """
    Groups candidate phrases by canonical form => picks a representative
    and counts frequency. Return {representative_string: freq}.
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
            # pick the most common normalized phrase
            rep = Counter(sublist).most_common(1)[0][0]
            candidate_themes[rep] = freq
    return candidate_themes

# -------------------------------------------
# Streamlit UI
# -------------------------------------------

st.sidebar.title("Select Mode")
mode = st.sidebar.radio("Choose an option:", ("Candidate Theme Extraction", "Full Tagging"))

if mode == "Candidate Theme Extraction":
    st.title("Candidate Theme Extraction (POS-based 3-Tag Example)")
    file = st.file_uploader("Upload CSV/Excel with 'Keywords' column", type=["csv","xls","xlsx"])
    num_keywords = st.number_input("Process first N keywords (0=all)", min_value=0, value=0)
    top_n = st.number_input("Keyphrases per keyword", min_value=1, value=3)
    min_freq = st.number_input("Min frequency for candidate theme", min_value=1, value=2)
    cluster_count = st.number_input("Number of clusters (0=skip)", min_value=0, value=0)

    if file:
        # load the DF
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
            keywords_list = df["Keywords"].tolist()
            if num_keywords>0:
                keywords_list = keywords_list[:num_keywords]

            all_phrases = extract_candidate_themes(keywords_list, top_n)
            candidate_map = group_candidate_themes(all_phrases, min_freq)

            if candidate_map:
                cdf = pd.DataFrame(list(candidate_map.items()), columns=["Candidate Theme","Frequency"])
                cdf = cdf.sort_values(by="Frequency", ascending=False)

                # For demonstration, let's do a POS-based 3-tag split with no user a tags
                splitted = []
                for theme in cdf["Candidate Theme"]:
                    canon = canonicalize_phrase(normalize_phrase(theme))
                    tokens = canon.split()
                    # no forced user A tags => empty set
                    a,b,c = split_tokens_pos_based(tokens, set())
                    splitted.append((a,b,c))

                cdf["A:Tag (Rec)"] = [x[0] for x in splitted]
                cdf["B:Tag (Rec)"] = [x[1] for x in splitted]
                cdf["C:Tag (Rec)"] = [x[2] for x in splitted]

                st.dataframe(cdf)

                # optional cluster
                if cluster_count>0 and len(candidate_map)>=cluster_count:
                    reps = list(candidate_map.keys())
                    embeddings = embedding_model.encode(reps)
                    km = KMeans(n_clusters=cluster_count, random_state=42)
                    labs = km.fit_predict(embeddings)
                    cluster_dict = {}
                    for lab,rep in zip(labs,reps):
                        cluster_dict.setdefault(lab,[]).append(rep)
                    st.write("### Clusters:")
                    for lab,group in cluster_dict.items():
                        st.write(f"**Cluster {lab}** => {', '.join(group)}")

                csvdata = cdf.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "Download Candidate Themes CSV",
                    csvdata,
                    "candidate_themes_pos_tagging.csv",
                    "text/csv"
                )
            else:
                st.write("No candidate themes met frequency threshold.")

elif mode=="Full Tagging":
    st.title("Full Keyword Tagging (3-Tag, POS-based for leftover tokens)")
    st.write("""
    For each keyword:
     1) Remove seed + omitted text,
     2) KeyBERT => top candidate,
     3) Normalize & canonicalize,
     4) A:Tag => user A-tag if found, else last token,
     5) B:Tag => first leftover token with POS=JJ/JJR/JJS/VBG, else first leftover,
     6) C:Tag => remainder.
    """)

    seed = st.text_input("Seed Keyword (Optional)")
    omit_str = st.text_input("Omit Phrases (comma-separated)")
    user_atags_str = st.text_input("User A:Tags (comma-separated)", value="door, window")
    file = st.file_uploader("Upload CSV/Excel with 'Keywords' column", type=["csv","xls","xlsx"])

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
            user_a_tags = set(normalize_token(a.strip()) for a in user_atags_str.split(",") if a.strip())

            A_tags, B_tags, C_tags = [], [], []
            for kw in df["Keywords"]:
                a,b,c = classify_keyword_three(kw, seed, omitted_list, user_a_tags)
                A_tags.append(a)
                B_tags.append(b)
                C_tags.append(c)

            df["A:Tag"] = A_tags
            df["B:Tag"] = B_tags
            df["C:Tag"] = C_tags

            st.dataframe(df[["Keywords","A:Tag","B:Tag","C:Tag"]])

            csvdata = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "Download Full Tagging CSV",
                data=csvdata,
                file_name="full_tagging_pos_based.csv",
                mime="text/csv"
            )

# ----------------------
# The POS-based leftover splitting:
# ----------------------

def split_tokens_pos_based(tokens, user_a_tags):
    """
    1) Force A:Tag => user A tag if found, else last token.
    2) leftover => B => first 'JJ*/VBG' or else first leftover
    3) leftover => C => remainder
    """
    # We'll do a copy for leftover
    leftover = tokens[:]

    # Step 1: Force user A tag
    a_tag = ""
    for tag in user_a_tags:
        if tag in leftover:
            a_tag = tag
            leftover.remove(tag)
            break
    if not a_tag and leftover:
        a_tag = leftover[-1]
        leftover = leftover[:-1]

    # Step 2: B tag => first leftover token that is an adjective or VBG, else first leftover
    b_tag = ""
    if leftover:
        # We'll do a quick POS tagging of leftover
        # Reconstruct them as text
        leftover_text = " ".join(leftover)
        pairs = pos_tag(leftover_text.split())  # list of (word, pos)
        # find the first that has pos in [JJ, JJR, JJS, VBG]
        found_idx = None
        for i,(word, posx) in enumerate(pairs):
            if posx.startswith("JJ") or posx=="VBG":
                b_tag = word
                found_idx = i
                break
        if b_tag:
            # remove it from leftover
            if b_tag in leftover:
                leftover.remove(b_tag)
        else:
            # pick the first leftover
            b_tag = leftover[0]
            leftover = leftover[1:]
    # Step 3: rest => c
    c_tag = " ".join(leftover) if leftover else ""

    return (a_tag, b_tag, c_tag)
