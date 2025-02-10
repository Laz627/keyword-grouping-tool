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

# --- Initialize KeyBERT ---
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
kw_model = KeyBERT(model=embedding_model)

# --- Helper Functions for Normalization & Candidate Tag Recommendation ---

def normalize_token(token):
    """Lowercase and lemmatize a token (noun mode). Also convert 'vs' to 'v'."""
    token = token.lower()
    if token == "vs":
        token = "v"
    return lemmatizer.lemmatize(token, pos='n')

def normalize_phrase(phrase):
    """Tokenize and normalize (preserving word order) the phrase."""
    tokens = word_tokenize(phrase.lower())
    norm_tokens = [normalize_token(t) for t in tokens if t.isalnum()]
    return " ".join(norm_tokens)

def canonicalize_phrase(phrase):
    """
    Returns a canonical form of the phrase by sorting the normalized tokens.
    This groups variants like "pella window", "pella windows", and "window pella" together.
    """
    tokens = word_tokenize(phrase.lower())
    norm_tokens = [normalize_token(t) for t in tokens if t.isalnum()]
    tokens_sorted = sorted(norm_tokens)
    return " ".join(tokens_sorted)

def extract_candidate_tags(candidate):
    """
    Given a candidate theme (a string), use POS-tagging to recommend:
      - A:Tag (Category): chosen as the rightmost noun.
      - B:Tag: the first adjective (if any).
      - C:Tag: all tokens before the head noun.
      - D:Tag: all tokens after the head noun.
    If no noun is found, fall back to the entire phrase.
    Returns a tuple: (A_tag, B_tag, C_tag, D_tag).
    """
    tokens = word_tokenize(candidate)
    tagged = pos_tag(tokens)
    norm_tokens = [normalize_token(t) for t in tokens if t.isalnum()]
    norm_tagged = list(zip(norm_tokens, [tag for word, tag in tagged if word.isalnum()]))
    
    # Extract adjectives (JJ*) and nouns (NN*)
    adjectives = [word for word, tag in norm_tagged if tag.startswith("JJ")]
    nouns = [word for word, tag in norm_tagged if tag.startswith("NN")]
    
    if nouns:
        a_tag = nouns[-1]
        idx = len(norm_tokens) - 1 - norm_tokens[::-1].index(a_tag)
    else:
        a_tag = candidate  # fallback
        idx = len(norm_tokens) - 1
    
    b_tag = adjectives[0] if adjectives else ""
    c_tag = " ".join(norm_tokens[:idx]) if idx > 0 else ""
    d_tag = " ".join(norm_tokens[idx+1:]) if idx+1 < len(norm_tokens) else ""
    
    # Remove duplicates across buckets.
    if b_tag == a_tag:
        b_tag = ""
    if c_tag == a_tag:
        c_tag = ""
    if d_tag == a_tag or d_tag == b_tag:
        d_tag = ""
    return a_tag, b_tag, c_tag, d_tag

# --- Helper Functions for Full Tagging ---

# For full tagging, we use a word-order–based approach.
STOPWORDS = {"the", "a", "an", "are", "is", "of", "and", "or", "how", "much",
             "do", "does", "be", "in", "to", "for", "on", "at"}

def process_keyword_order(keyword, seed_keyword, omitted_list, user_a_tags):
    """
    Processes a single keyword based on word order and user-provided parameters.
    
    Steps:
      1. Lowercase the keyword and remove seed keyword (if provided) and omitted phrases.
      2. Tokenize and normalize (keep only alphanumeric tokens).
      3. Search for the first occurrence (in the filtered tokens) of any token in user_a_tags.
         If found, that token becomes the Category; otherwise, Category is "other-general".
      4. For a found A-tag:
         - tokens_before = raw tokens before the found A-tag (ignoring stopwords and tokens equal to the Category).
         - tokens_after = raw tokens after the A-tag (ignoring stopwords and tokens equal to the Category).
         - B-tag: last token of tokens_after if available; else, last token of tokens_before.
         - C-tag: join tokens_before.
         - D-tag: join tokens_after (excluding the token used for B-tag).
      5. For "other-general", use all filtered tokens:
         - B-tag: last token.
         - C-tag: join all tokens except the last.
         - D-tag: empty.
      6. Remove duplicates (if any tag equals the Category or repeats).
    
    Returns (Category, B-tag, C-tag, D-tag).
    """
    text = keyword.lower()
    if seed_keyword:
        pattern = rf'\b{re.escape(seed_keyword.lower())}\b'
        text = re.sub(pattern, '', text)
    def omit(text, omitted_list):
        result = text
        for word in omitted_list:
            result = re.sub(rf'\b{re.escape(word.lower())}\b', '', result)
        return result.strip()
    text = omit(text, omitted_list)
    
    tokens = word_tokenize(text)
    raw_tokens = [normalize_token(t) for t in tokens if t.isalnum()]
    filtered_tokens = [t for t in raw_tokens if t not in set(s.lower() for s in omitted_list)]
    
    idx_A = None
    category = ""
    for i, token in enumerate(filtered_tokens):
        if token in user_a_tags:
            idx_A = i
            category = token
            break
    if idx_A is None:
        category = "other-general"
    
    if category != "other-general":
        tokens_before = raw_tokens[:idx_A]
        tokens_after = raw_tokens[idx_A+1:]
        tokens_before = [t for t in tokens_before if t not in STOPWORDS and t != category]
        tokens_after = [t for t in tokens_after if t not in STOPWORDS and t != category]
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
        tokens_all = [t for t in raw_tokens if t not in STOPWORDS]
        if tokens_all:
            b_tag = tokens_all[-1]
            c_tokens = tokens_all[:-1]
            c_tag = " ".join(c_tokens) if c_tokens else ""
        else:
            b_tag = ""
            c_tag = ""
        d_tag = ""
    
    if b_tag == category:
        b_tag = ""
    if c_tag == category:
        c_tag = ""
    if d_tag == category or d_tag == b_tag:
        d_tag = ""
    
    return (category, b_tag, c_tag, d_tag)

# --- Streamlit App Interface ---

st.sidebar.title("Select Mode")
mode = st.sidebar.radio("Choose an option:", ("Candidate Theme Extraction", "Full Tagging"))

if mode == "Candidate Theme Extraction":
    st.title("KeyBERT Candidate Theme Extraction and Tag Recommendation")
    st.markdown(
        """
        This mode uses KeyBERT to extract candidate keyphrases from your keyword list,
        normalizes them to group singular/plural variants and different word orders,
        and then provides recommended tag components.
        """
    )
    uploaded_file = st.file_uploader("Upload CSV/Excel File", type=["csv", "xls", "xlsx"], key="cand")
    num_keywords = st.number_input("Process first N keywords (0 = all)", min_value=0, value=0, step=1, key="cand_num")
    top_n = st.number_input("Keyphrases per keyword", min_value=1, value=3, step=1, key="cand_top")
    min_freq = st.number_input("Minimum frequency for candidate theme", min_value=1, value=2, step=1, key="cand_freq")
    num_clusters = st.number_input("Number of clusters (0 to skip clustering)", min_value=0, value=0, step=1, key="cand_cluster")
    
    if uploaded_file:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
        except Exception as e:
            st.error(f"Error loading file: {e}")
            st.stop()
        if "Keywords" not in df.columns:
            st.error("The file must contain a column named 'Keywords'.")
        else:
            keywords_list = df["Keywords"].tolist()
            if num_keywords > 0:
                keywords_list = keywords_list[:num_keywords]
            
            st.write("Extracting keyphrases from keywords using KeyBERT...")
            all_phrases = []
            progress_bar = st.progress(0)
            for idx, kw in enumerate(keywords_list):
                keyphrases = kw_model.extract_keywords(kw, keyphrase_ngram_range=(1, 3), stop_words='english', top_n=top_n)
                extracted = [kp[0].lower() for kp in keyphrases if kp[0]]
                all_phrases.extend(extracted)
                progress_bar.progress((idx + 1) / len(keywords_list))
            progress_bar.empty()
            
            # Group candidate themes by canonical form (sorted normalized tokens).
            grouped = {}
            for phrase in all_phrases:
                norm_phrase = normalize_phrase(phrase)
                canon = canonicalize_phrase(norm_phrase)
                if canon:
                    grouped[canon] = grouped.get(canon, []) + [norm_phrase]
            
            # Sum frequencies and choose a representative phrase for each group.
            candidate_themes = {}
            for canon, phrases in grouped.items():
                freq = len(phrases)
                if freq >= min_freq:
                    # Choose the most common candidate from the group.
                    rep = Counter(phrases).most_common(1)[0][0]
                    candidate_themes[rep] = freq
            
            st.write("### Candidate Themes and Frequencies (Grouped)")
            if candidate_themes:
                candidate_df = pd.DataFrame(list(candidate_themes.items()), columns=["Theme", "Frequency"])
                candidate_df = candidate_df.sort_values(by="Frequency", ascending=False)
                st.dataframe(candidate_df)
            else:
                st.write("No candidate themes met the minimum frequency threshold.")
            
            # Provide recommended tag structure for each candidate theme.
            recommendations = []
            for theme, freq in candidate_themes.items():
                a_tag, b_tag, c_tag, d_tag = extract_candidate_tags(theme)
                recommendations.append({
                    "Theme": theme,
                    "Frequency": freq,
                    "A:Tag (Category)": a_tag,
                    "B:Tag": b_tag,
                    "C:Tag": c_tag,
                    "D:Tag": d_tag
                })
            rec_df = pd.DataFrame(recommendations)
            st.write("### Recommended Tag Structure for Each Candidate Theme")
            st.dataframe(rec_df)
            
            # Optionally, cluster the candidate themes.
            if num_clusters > 0 and len(candidate_themes) >= num_clusters:
                st.write("### Candidate Theme Clusters")
                themes = list(candidate_themes.keys())
                embeddings = embedding_model.encode(themes)
                kmeans = KMeans(n_clusters=num_clusters, random_state=42)
                cluster_labels = kmeans.fit_predict(embeddings)
                clusters = {}
                for label, theme in zip(cluster_labels, themes):
                    clusters.setdefault(label, []).append(theme)
                for label, group in clusters.items():
                    st.write(f"**Cluster {label}:** {', '.join(group)}")
            
            st.markdown(
                """
                **Next Steps:**  
                Use the candidate themes and tag recommendations above as a basis for designing your final
                rule-based tagging system.
                """
            )
            
            # Optionally, allow the candidate theme recommendations to be downloaded.
            csv_data = rec_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download Candidate Theme Recommendations CSV",
                data=csv_data,
                file_name="candidate_theme_recommendations.csv",
                mime="text/csv"
            )

elif mode == "Full Tagging":
    st.title("Full Keyword Tagging")
    st.markdown(
        """
        In this mode, you provide parameters (a Seed Keyword, Omit Phrases, and desired A:Tags),
        and each keyword in your input file is processed to assign:
          - **Category (A:Tag)** – based on the first occurrence of any of your A:Tags,
          - **B:Tag, C:Tag, D:Tag** – derived from the remaining word order.
        If no A:Tag is found, the Category is set to "other-general" and the remainder of the tokens
        form the other tags.
        """
    )
    seed_keyword = st.text_input("Seed Keyword (Optional)", value="", key="full_seed")
    omit_input = st.text_input("Omit Phrases (comma‑separated)", value="", key="full_omit")
    a_tags_input = st.text_input("A:Tags (comma‑separated)", value="window, door, price", key="full_a_tags")
    uploaded_file = st.file_uploader("Upload CSV/Excel File", type=["csv", "xls", "xlsx"], key="full_file")
    
    if uploaded_file:
        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
        except Exception as e:
            st.error(f"Error loading file: {e}")
            st.stop()
        if "Keywords" not in df.columns:
            st.error("The file must contain a column named 'Keywords'.")
        else:
            omitted_list = [s.strip() for s in omit_input.split(",") if s.strip()]
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
            
            st.write("### Full Keyword Tagging Output")
            st.dataframe(df[["Keywords", "Category", "B-tag", "C-tag", "D-tag"]])
            
            csv_data = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download Tagged Keywords CSV",
                data=csv_data,
                file_name="tagged_keywords.csv",
                mime="text/csv"
            )
