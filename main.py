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

# --- Initialize KeyBERT ---
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
kw_model = KeyBERT(model=embedding_model)

# --- Helper Functions ---

def normalize_token(token):
    """Lowercase and lemmatize a token (noun mode). Also convert 'vs' to 'v'."""
    token = token.lower()
    if token == "vs":
        token = "v"
    return lemmatizer.lemmatize(token, pos='n')

def normalize_phrase(phrase):
    """
    Tokenize and normalize a phrase—preserving the original order.
    This groups singular/plural variants together.
    """
    tokens = word_tokenize(phrase.lower())
    norm_tokens = [normalize_token(t) for t in tokens if t.isalnum()]
    return " ".join(norm_tokens)

def canonicalize_phrase(phrase):
    """
    Returns a canonical form of the phrase.
    It tokenizes and normalizes the phrase, removes the word "series",
    and then sorts the tokens alphabetically. This ensures that variants like
    "pella 350", "pella 350 series", and "350 pella" become identical.
    """
    tokens = word_tokenize(phrase.lower())
    norm_tokens = [normalize_token(t) for t in tokens if t.isalnum() and normalize_token(t) != "series"]
    tokens_sorted = sorted(norm_tokens)
    return " ".join(tokens_sorted)

# --- Candidate Theme Extraction Functions ---

def extract_candidate_themes(keywords_list, top_n):
    """
    Uses KeyBERT to extract keyphrases (of lengths 1 to 3) from each keyword.
    Returns a list of extracted candidate phrases (all lowercased).
    """
    all_phrases = []
    for kw in keywords_list:
        keyphrases = kw_model.extract_keywords(kw, keyphrase_ngram_range=(1, 3), stop_words='english', top_n=top_n)
        extracted = [kp[0].lower() for kp in keyphrases if kp[0]]
        all_phrases.extend(extracted)
    return all_phrases

def group_candidate_themes(all_phrases, min_freq):
    """
    Groups candidate themes by their canonical form.
    Returns a dictionary mapping a representative candidate theme to its total frequency.
    """
    grouped = {}
    for phrase in all_phrases:
        norm_phrase = normalize_phrase(phrase)
        canon = canonicalize_phrase(norm_phrase)
        if canon:
            grouped[canon] = grouped.get(canon, []) + [norm_phrase]
    candidate_themes = {}
    for canon, phrases in grouped.items():
        freq = len(phrases)
        if freq >= min_freq:
            # Choose the most common candidate (in original normalized form)
            rep = Counter(phrases).most_common(1)[0][0]
            candidate_themes[rep] = freq
    return candidate_themes

# --- Full Tagging Function ---
def classify_keyword(keyword, seed_keyword, omitted_list):
    """
    Processes a single keyword by:
      1. Removing the seed keyword and omitted phrases.
      2. Extracting the top candidate keyphrase using KeyBERT.
      3. Canonicalizing the candidate.
    Returns the canonical candidate as the classification. If no candidate is extracted,
    returns "other-general".
    """
    text = keyword.lower()
    if seed_keyword:
        pattern = rf'\b{re.escape(seed_keyword.lower())}\b'
        text = re.sub(pattern, '', text)
    # Remove omitted phrases.
    for omit in omitted_list:
        text = re.sub(rf'\b{re.escape(omit.lower())}\b', '', text)
    text = text.strip()
    keyphrases = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 3), stop_words='english', top_n=1)
    if keyphrases:
        candidate = keyphrases[0][0].lower()
        norm_candidate = normalize_phrase(candidate)
        canon = canonicalize_phrase(norm_candidate)
        return canon if canon else "other-general"
    else:
        return "other-general"

# --- Streamlit App Interface ---

st.sidebar.title("Select Mode")
mode = st.sidebar.radio("Choose an option:", ("Candidate Theme Extraction", "Full Tagging"))

if mode == "Candidate Theme Extraction":
    st.title("Candidate Theme Extraction with KeyBERT")
    st.markdown(
        """
        Upload a CSV/Excel file with a **Keywords** column.  
        The app will extract candidate keyphrases using KeyBERT, canonicalize them (grouping
        singular/plural variants and different word orders together), and aggregate their frequencies.
        These candidate themes (and their recommended classification tag, which is simply the canonical form)
        are displayed below.
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
            
            st.write("Extracting candidate keyphrases using KeyBERT...")
            all_phrases = extract_candidate_themes(keywords_list, top_n)
            
            candidate_themes = group_candidate_themes(all_phrases, min_freq)
            
            st.write("### Candidate Themes (Grouped by Canonical Form)")
            if candidate_themes:
                candidate_df = pd.DataFrame(list(candidate_themes.items()), columns=["Candidate Theme", "Frequency"])
                candidate_df = candidate_df.sort_values(by="Frequency", ascending=False)
                # The recommended classification tag is the canonical form.
                candidate_df["Recommended Tag"] = candidate_df["Candidate Theme"]
                st.dataframe(candidate_df)
            else:
                st.write("No candidate themes met the minimum frequency threshold.")
            
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
                Use the candidate themes and their recommended tags as a starting point for your final tagging rules.
                """
            )
            csv_data = candidate_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download Candidate Theme Recommendations CSV",
                data=csv_data,
                file_name="candidate_theme_recommendations.csv",
                mime="text/csv"
            )

elif mode == "Full Tagging":
    st.title("Full Keyword Tagging (Using Canonical Classification)")
    st.markdown(
        """
        In this mode, you provide a Seed Keyword (optional) and Omit Phrases.  
        Each keyword is processed by extracting its top candidate phrase via KeyBERT,
        canonicalizing it, and using that as its classification.
        This ensures that similar phrases (e.g. "pella 350" and "pella 350 series")
        receive the same tag.
        """
    )
    seed_keyword = st.text_input("Seed Keyword (Optional)", value="", key="full_seed")
    omit_input = st.text_input("Omit Phrases (comma‑separated)", value="", key="full_omit")
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
            
            classifications = []
            for idx, row in df.iterrows():
                keyword = row["Keywords"]
                tag = classify_keyword(keyword, seed_keyword, omitted_list)
                classifications.append(tag)
            
            df["Classification"] = classifications
            
            st.write("### Full Keyword Tagging Output")
            st.dataframe(df[["Keywords", "Classification"]])
            
            csv_data = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download Tagged Keywords CSV",
                data=csv_data,
                file_name="tagged_keywords.csv",
                mime="text/csv"
            )
