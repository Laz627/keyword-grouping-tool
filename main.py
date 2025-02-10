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

# --- Helper Functions for Normalization and Canonicalization ---

def normalize_token(token):
    """Lowercase and lemmatize a token (noun mode); convert 'vs' to 'v'."""
    token = token.lower()
    if token == "vs":
        token = "v"
    return lemmatizer.lemmatize(token, pos='n')

def normalize_phrase(phrase):
    """
    Tokenize and normalize a phrase—preserving word order.
    This groups singular/plural variants together.
    """
    tokens = word_tokenize(phrase.lower())
    norm_tokens = [normalize_token(t) for t in tokens if t.isalnum()]
    return " ".join(norm_tokens)

def canonicalize_phrase(phrase):
    """
    Returns a canonical form of the phrase.
    It tokenizes and normalizes the phrase, removes any occurrence of the word "series",
    and then sorts the tokens alphabetically.
    This ensures that variants like "pella 350", "pella 350 series", and "350 pella" become identical.
    """
    tokens = word_tokenize(phrase.lower())
    norm_tokens = [normalize_token(t) for t in tokens if t.isalnum() and normalize_token(t) != "series"]
    tokens_sorted = sorted(norm_tokens)
    return " ".join(tokens_sorted)

# --- Candidate Tag Recommendation (Three-Tag Version) ---

def extract_candidate_tags_three_v2(candidate):
    """
    Given a candidate theme (string), recommend a three-tag structure:
      - A:Tag (Category):
          • If a numeric token is present and "series" appears in the candidate,
            use the last numeric token combined with "-series".
          • Otherwise, fall back to choosing the rightmost noun (if any).
      - B:Tag and C:Tag: For simplicity in this version, we return the canonical candidate as the recommended A:Tag.
        (You can later refine this function to split the canonical candidate into additional modifiers.)
    Here we simply recommend:
          A:Tag = canonical candidate,
          B:Tag = canonical candidate,
          C:Tag = "".
    (You can modify this as needed.)
    """
    # For our purposes, we canonicalize the candidate.
    canon = canonicalize_phrase(normalize_phrase(candidate))
    if not canon:
        return ("other-general", "", "")
    # In this simple recommendation, we set A and B to be the canonical phrase.
    # (A more advanced version might use POS-tagging to split out a head noun.)
    return (canon, canon, "")

# --- Full Tagging Function (Three-Tag Version) ---
def classify_keyword_three(keyword, seed_keyword, omitted_list):
    """
    Processes a single keyword as follows:
      1. Lowercase the keyword.
      2. Remove the seed keyword (if provided) and any omitted phrases.
      3. Use KeyBERT (top_n=1) to extract a candidate keyphrase.
      4. Normalize and canonicalize the candidate.
      5. Use extract_candidate_tags_three_v2 to obtain a three-tag structure.
         If no candidate is found, return ("other-general", "", "").
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
            return extract_candidate_tags_three_v2(canon)
        else:
            return ("other-general", "", "")
    else:
        return ("other-general", "", "")

# --- Streamlit App Modes ---

st.sidebar.title("Select Mode")
mode = st.sidebar.radio("Choose an option:", ("Candidate Theme Extraction", "Full Tagging"))

if mode == "Candidate Theme Extraction":
    st.title("Candidate Theme Extraction with KeyBERT")
    st.markdown(
        """
        Upload a CSV/Excel file with a **Keywords** column.  
        This mode extracts candidate keyphrases from your keywords using KeyBERT,
        canonicalizes them (so that variants like "pella 350" and "pella 350 series" group together),
        and aggregates their frequencies.
        
        The recommended tag for each candidate is the canonical form (used as both A:Tag and B:Tag),
        leaving C:Tag empty. Use these recommendations as a starting point for your final tagging rules.
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
            all_phrases = []
            progress_bar = st.progress(0)
            for idx, kw in enumerate(keywords_list):
                keyphrases = kw_model.extract_keywords(kw, keyphrase_ngram_range=(1, 3), stop_words='english', top_n=top_n)
                extracted = [kp[0].lower() for kp in keyphrases if kp[0]]
                all_phrases.extend(extracted)
                progress_bar.progress((idx + 1) / len(keywords_list))
            progress_bar.empty()
            
            # Group candidate themes by canonical form.
            grouped = {}
            for phrase in all_phrases:
                norm_phrase = normalize_phrase(phrase)
                canon = canonicalize_phrase(norm_phrase)
                if canon:
                    grouped[canon] = grouped.get(canon, []) + [norm_phrase]
            
            # Aggregate frequencies and choose a representative candidate.
            candidate_themes = {}
            for canon, phrases in grouped.items():
                freq = len(phrases)
                if freq >= min_freq:
                    rep = Counter(phrases).most_common(1)[0][0]
                    candidate_themes[rep] = freq
            
            st.write("### Candidate Themes (Grouped)")
            if candidate_themes:
                candidate_df = pd.DataFrame(list(candidate_themes.items()), columns=["Candidate Theme", "Frequency"])
                candidate_df = candidate_df.sort_values(by="Frequency", ascending=False)
                # Recommended tag is the canonical form.
                candidate_df["Recommended A:Tag"] = candidate_df["Candidate Theme"].apply(lambda x: canonicalize_phrase(normalize_phrase(x)))
                candidate_df["Recommended B:Tag"] = candidate_df["Recommended A:Tag"]
                candidate_df["Recommended C:Tag"] = ""
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
                Review these candidate themes and the recommended tag structure (A:Tag, B:Tag, C:Tag).
                Use these recommendations as a starting point for designing your final rule‑based tagging system.
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
          4. The canonical candidate is then processed to generate a three-tag structure:
             - **A:Tag (Category)**
             - **B:Tag**
             - **C:Tag**
        If no candidate is extracted, the keyword is classified as "other-general" (with empty B and C tags).
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
      3. Normalizing and canonicalizing the candidate.
      4. Using extract_candidate_tags_three_v2 to get a three-tag structure.
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
            return extract_candidate_tags_three_v2(canon)
        else:
            return ("other-general", "", "")
    else:
        return ("other-general", "", "")
