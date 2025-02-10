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

# --- Helper Functions for Normalization & Canonicalization ---

def normalize_token(token):
    """Lowercase and lemmatize a token (noun mode); also convert 'vs' to 'v'."""
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
    It tokenizes and normalizes the phrase, removes any token equal to "series",
    and then sorts the tokens alphabetically.
    This way, phrases such as "pella 350", "pella 350 series", and "350 pella" become identical.
    """
    tokens = word_tokenize(phrase.lower())
    norm_tokens = [normalize_token(t) for t in tokens if t.isalnum() and normalize_token(t) != "series"]
    tokens_sorted = sorted(norm_tokens)
    return " ".join(tokens_sorted)

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
    Returns a dictionary mapping a representative candidate (in natural normalized form)
    to its aggregated frequency.
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
            # Choose the most common candidate (by natural order)
            rep = Counter(phrases).most_common(1)[0][0]
            candidate_themes[rep] = freq
    return candidate_themes

def default_recommend_tags(candidate):
    """
    Given a candidate theme string, compute a default three-tag split by:
      - Splitting the normalized candidate into tokens.
      - Setting A:Tag to the last token,
      - B:Tag to the first token,
      - C:Tag to the middle tokens (joined by space), or empty if only two tokens.
    If only one token exists, that token is A:Tag and B and C are empty.
    """
    norm = normalize_phrase(candidate)
    tokens = norm.split()
    if not tokens:
        return ("other-general", "", "")
    if len(tokens) == 1:
        return (tokens[0], "", "")
    else:
        a_tag = tokens[-1]
        b_tag = tokens[0]
        c_tag = " ".join(tokens[1:-1]) if len(tokens) > 2 else ""
        # Avoid duplicates: if b_tag equals a_tag, clear b_tag.
        if b_tag == a_tag:
            b_tag = ""
        return (a_tag, b_tag, c_tag)

# --- Full Tagging Functions ---

def classify_keyword(keyword, seed_keyword, omitted_list, target_a_tags):
    """
    Processes a single keyword as follows:
      1. Lowercase the keyword and remove the seed keyword and any omitted phrases.
      2. Use KeyBERT (top candidate) to extract a candidate keyphrase.
      3. Normalize and canonicalize the candidate.
      4. If any token from target_a_tags (provided by the user) appears in the candidate,
         force the classification (A:Tag) to that token.
      5. Otherwise, use the default recommendation split (default_recommend_tags).
      6. If no candidate is extracted, return ("other-general", "", "").
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
        if not canon:
            return ("other-general", "", "")
        # Check if any target A:tag appears in the canonical candidate.
        tokens = canon.split()
        forced = None
        for tag in target_a_tags:
            if tag in tokens:
                forced = tag
                break
        if forced:
            # Remove the forced tag from tokens and then use default_recommend_tags on the remainder.
            remaining = [t for t in tokens if t != forced]
            if remaining:
                default_part = " ".join(remaining)
                a_tag, b_tag, c_tag = default_recommend_tags(default_part)
                # Then set A:Tag to the forced target.
                a_tag = forced
            else:
                a_tag, b_tag, c_tag = forced, "", ""
        else:
            a_tag, b_tag, c_tag = default_recommend_tags(canon)
        return (a_tag, b_tag, c_tag)
    else:
        return ("other-general", "", "")

# --- Streamlit App Interface ---

st.sidebar.title("Select Mode")
mode = st.sidebar.radio("Choose an option:", ("Candidate Theme Extraction", "Full Tagging"))

if mode == "Candidate Theme Extraction":
    st.title("Candidate Theme Extraction with KeyBERT")
    st.markdown(
        """
        **Step 1: Candidate Extraction & Grouping**

        Upload a CSV/Excel file with a **Keywords** column.  
        The app extracts candidate keyphrases using KeyBERT, then normalizes and canonicalizes them so that variants like
        "pella 350" and "pella 350 series" are grouped together.  
        
        The output is a table of candidate themes (by their representative canonical form) along with frequencies.
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
                # For recommendation, we use the default split on the candidate theme.
                rec_tags = candidate_df["Candidate Theme"].apply(default_recommend_tags)
                candidate_df["A:Tag (Rec)"] = rec_tags.apply(lambda x: x[0])
                candidate_df["B:Tag (Rec)"] = rec_tags.apply(lambda x: x[1])
                candidate_df["C:Tag (Rec)"] = rec_tags.apply(lambda x: x[2])
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
                Review the candidate themes and their recommended tag splits.  
                You may choose to refine these manually and then use them in your final tagging rules.
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
        **Step 2: Full Tagging Using Candidate Themes**

        In this mode you provide:
         - A Seed Keyword (optional) to remove extraneous text.
         - Omit Phrases (comma‑separated) to be removed (e.g. brand names).
         - **Target A:Tags:** a comma‑separated list of keywords (e.g. "door, window") that you expect to be
           the main classification. If any of these appear in the extracted candidate, they will be used as the A:Tag.
        
        For each keyword, the top candidate keyphrase is extracted using KeyBERT, normalized and canonicalized,
        and then split into three tags using our default recommendation. If any target A:tag is found in the canonical
        candidate, that target is forced as the A:Tag.
        
        The output table shows columns: Keywords, A:Tag, B:Tag, and C:Tag.
        """
    )
    seed_keyword = st.text_input("Seed Keyword (Optional)", value="", key="full_seed")
    omit_input = st.text_input("Omit Phrases (comma‑separated)", value="", key="full_omit")
    target_a_tags_input = st.text_input("Target A:Tags (comma‑separated)", value="door, window", key="full_a_tags")
    mapping_file = st.file_uploader("Optional: Upload Mapping CSV", type=["csv"], key="mapping")
    # Mapping CSV should have columns: Canonical, A_Tag, B_Tag, C_Tag
    uploaded_file = st.file_uploader("Upload Keywords CSV/Excel File", type=["csv", "xls", "xlsx"], key="full_file")
    
    # Build mapping dictionary if provided.
    mapping_dict = {}
    if mapping_file:
        try:
            mapping_df = pd.read_csv(mapping_file)
            # Expect columns: Canonical, A_Tag, B_Tag, C_Tag
            for _, row in mapping_df.iterrows():
                key = row["Canonical"].strip().lower()
                mapping_dict[key] = (row["A_Tag"].strip(), row["B_Tag"].strip(), row["C_Tag"].strip())
        except Exception as e:
            st.error(f"Error loading mapping file: {e}")
    
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
            target_a_tags = set(normalize_token(s.strip()) for s in target_a_tags_input.split(",") if s.strip())
            
            a_tags_out = []
            b_tags_out = []
            c_tags_out = []
            # For each keyword, extract candidate and then either use the mapping (if available)
            # or the default recommendation.
            for idx, row in df.iterrows():
                keyword = row["Keywords"]
                # Extract candidate three-tag structure.
                a_tag, b_tag, c_tag = classify_keyword_three(keyword, seed_keyword, omitted_list)
                # Canonicalize the candidate for lookup.
                # (We re-run the extraction on the candidate keyphrase.)
                keyphrases = kw_model.extract_keywords(keyword.lower(), keyphrase_ngram_range=(1, 3), stop_words='english', top_n=1)
                if keyphrases:
                    candidate = keyphrases[0][0].lower()
                    norm_candidate = normalize_phrase(candidate)
                    canon = canonicalize_phrase(norm_candidate)
                else:
                    canon = "other-general"
                # If any target A:tag is present in the canonical candidate, force it.
                for tag in target_a_tags:
                    if tag in canon.split():
                        a_tag = tag
                        break
                # If a mapping file was provided and the canonical candidate is in the mapping, use it.
                if mapping_dict and canon in mapping_dict:
                    a_tag, b_tag, c_tag = mapping_dict[canon]
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
