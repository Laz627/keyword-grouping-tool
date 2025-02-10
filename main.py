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

# --- Helper Functions for Normalization and Canonicalization ---

def normalize_token(token):
    """
    Lowercase and lemmatize a token (noun mode).
    Also converts "vs" to "v".
    """
    token = token.lower()
    if token == "vs":
        token = "v"
    return lemmatizer.lemmatize(token, pos='n')

def normalize_phrase(phrase):
    """
    Tokenize and normalize a phrase (preserving word order).
    This groups singular/plural variants together.
    """
    tokens = word_tokenize(phrase.lower())
    norm_tokens = [normalize_token(t) for t in tokens if t.isalnum()]
    return " ".join(norm_tokens)

def canonicalize_phrase(phrase):
    """
    Returns a canonical form for grouping candidate themes.
    It tokenizes and normalizes the phrase, removes any occurrence of "series",
    and then sorts the tokens alphabetically.
    For example, both "pella 350" and "pella 350 series" become "350 pella".
    """
    tokens = word_tokenize(phrase.lower())
    # Remove non-alphanumeric tokens and remove tokens equal to "series"
    norm_tokens = [normalize_token(t) for t in tokens if t.isalnum() and normalize_token(t) != "series"]
    tokens_sorted = sorted(norm_tokens)
    return " ".join(tokens_sorted)

# --- Recommended Tag Split Function ---
def recommend_tags(candidate):
    """
    Given a candidate theme (string), this function:
      1. Normalizes the candidate (using normalize_phrase).
      2. Removes any occurrence of "series" (since that is considered extraneous for grouping).
      3. Splits the resulting phrase into tokens.
      4. If the candidate has multiple tokens:
            - A:Tag is set to the last token,
            - B:Tag is set to the first token,
            - C:Tag is set to the join of the remaining tokens (if any).
         If only one token exists, that token becomes A:Tag and B and C are empty.
    Returns a tuple: (A_tag, B_tag, C_tag)
    """
    norm = normalize_phrase(candidate)
    tokens = norm.split()
    # Remove "series" if present
    tokens = [t for t in tokens if t != "series"]
    if not tokens:
        return ("other-general", "", "")
    if len(tokens) == 1:
        return (tokens[0], "", "")
    else:
        a_tag = tokens[-1]      # last token as the main category
        b_tag = tokens[0]       # first token as a modifier
        c_tag = " ".join(tokens[1:-1])  # the middle tokens
        # Avoid duplicates: if b_tag equals a_tag, clear b_tag.
        if b_tag == a_tag:
            b_tag = ""
        return (a_tag, b_tag, c_tag)

# --- Candidate Theme Extraction Functions ---
def extract_candidate_themes(keywords_list, top_n):
    """
    Uses KeyBERT to extract keyphrases (lengths 1 to 3) from each keyword.
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
    Returns a dictionary mapping a representative candidate (preserving natural order)
    to its total frequency.
    """
    grouped = {}
    for phrase in all_phrases:
        norm_phrase = normalize_phrase(phrase)
        canon = canonicalize_phrase(norm_phrase)
        if canon:
            grouped.setdefault(canon, []).append(norm_phrase)
    candidate_themes = {}
    for canon, phrases in grouped.items():
        freq = len(phrases)
        if freq >= min_freq:
            # Choose the most common candidate (by natural order) from the group.
            rep = Counter(phrases).most_common(1)[0][0]
            candidate_themes[rep] = freq
    return candidate_themes

# --- Full Tagging Function (Three-Tag Version) ---
def classify_keyword_three(keyword, seed_keyword, omitted_list):
    """
    Processes a single keyword as follows:
      1. Lowercase the keyword.
      2. Remove the seed keyword (if provided) and any omitted phrases.
      3. Use KeyBERT (top candidate) to extract a candidate keyphrase.
      4. Normalize the candidate and then canonicalize it (to group variants).
      5. Use recommend_tags on the candidate to get a three-tag structure.
         If no candidate is extracted, returns ("other-general", "", "").
    """
    text = keyword.lower()
    if seed_keyword:
        pattern = rf'\b{re.escape(seed_keyword.lower())}\b'
        text = re.sub(pattern, '', text)
    for omit in omitted_list:
        text = re.sub(rf'\b{re.escape(omit.lower())}\b', '', text)
    text = text.strip()
    keyphrases = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 3), stop_words='english', top_n=1)
    if keyphrases:
        candidate = keyphrases[0][0].lower()
        norm_candidate = normalize_phrase(candidate)
        canon = canonicalize_phrase(norm_candidate)
        if canon:
            return recommend_tags(canon)
        else:
            return ("other-general", "", "")
    else:
        return ("other-general", "", "")

# --- Streamlit App Interface ---

st.sidebar.title("Select Mode")
mode = st.sidebar.radio("Choose an option:", ("Candidate Theme Extraction", "Full Tagging"))

if mode == "Candidate Theme Extraction":
    st.title("Candidate Theme Extraction with KeyBERT")
    st.markdown(
        """
        Upload a CSV/Excel file with a **Keywords** column.
        
        The app will extract candidate keyphrases from your keywords using KeyBERT,
        canonicalize them (so that variants like "pella 350" and "pella 350 series" are grouped together),
        and aggregate their frequencies. For each candidate theme, a recommended three-tag structure is provided,
        where:
          - **A:Tag** (Category) is the last token of the candidate,
          - **B:Tag** is the first token, and
          - **C:Tag** is the remaining tokens (if any).
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
                # Apply our recommended three-tag split.
                rec_tags = candidate_df["Candidate Theme"].apply(recommend_tags)
                candidate_df["A:Tag (Recommended)"] = rec_tags.apply(lambda x: x[0])
                candidate_df["B:Tag (Recommended)"] = rec_tags.apply(lambda x: x[1])
                candidate_df["C:Tag (Recommended)"] = rec_tags.apply(lambda x: x[2])
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
                Use these candidate themes and their recommended tag splits as a starting point
                for designing your final rule-based tagging system.
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
    st.title("Full Keyword Tagging (Three-Tag Version)")
    st.markdown(
        """
        In this mode, you provide a Seed Keyword (optional) and Omit Phrases.
        Each keyword in your file is processed as follows:
          1. The seed keyword and omitted phrases are removed.
          2. KeyBERT (top candidate) extracts a keyphrase.
          3. That candidate is normalized and canonicalized.
          4. The recommended three-tag structure is computed using the following simple rule:
             - **A:Tag:** the last token of the canonical candidate (usually the head noun),
             - **B:Tag:** the first token,
             - **C:Tag:** the remaining tokens in between.
          If no candidate is extracted, the classification is "other-general".
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
            a_tags_out = []
            b_tags_out = []
            c_tags_out = []
            for idx, row in df.iterrows():
                keyword = row["Keywords"]
                a_tag, b_tag, c_tag = classify_keyword_three(keyword, seed_keyword, omitted_list)
                classifications.append((a_tag, b_tag, c_tag))
                a_tags_out.append(a_tag)
                b_tags_out.append(b_tag)
                c_tags_out.append(c_tag)
            
            df["A:Tag"] = a_tags_out
            df["B:Tag"] = b_tags_out
            df["C:Tag"] = c_tags_out
            
            st.write("### Full Keyword Tagging Output")
            st.dataframe(df[["Keywords", "A:Tag", "B:Tag", "C:Tag"]])
            
            csv_data = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download Tagged Keywords CSV",
                data=csv_data,
                file_name="tagged_keywords.csv",
                mime="text/csv"
            )

# --- Full Tagging Helper using Three-Tag Recommendation ---
def classify_keyword_three(keyword, seed_keyword, omitted_list):
    """
    Processes a single keyword by:
      1. Removing the seed keyword and omitted phrases.
      2. Extracting the top candidate keyphrase using KeyBERT (top_n=1).
      3. Normalizing the candidate and canonicalizing it.
      4. Applying recommend_tags to obtain a three-tag structure.
         If no candidate is extracted, returns ("other-general", "", "").
    """
    text = keyword.lower()
    if seed_keyword:
        pattern = rf'\b{re.escape(seed_keyword.lower())}\b'
        text = re.sub(pattern, '', text)
    for omit in omitted_list:
        text = re.sub(rf'\b{re.escape(omit.lower())}\b', '', text)
    text = text.strip()
    keyphrases = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 3), stop_words='english', top_n=1)
    if keyphrases:
        candidate = keyphrases[0][0].lower()
        norm_candidate = normalize_phrase(candidate)
        canon = canonicalize_phrase(norm_candidate)
        if canon:
            return recommend_tags(canon)
        else:
            return ("other-general", "", "")
    else:
        return ("other-general", "", "")
