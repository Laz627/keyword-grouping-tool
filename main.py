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

# --- Helper Functions ---

def normalize_token(token):
    """Lowercase and lemmatize a token (using noun mode). Also convert 'vs' to 'v'."""
    token = token.lower()
    if token == "vs":
        token = "v"
    return lemmatizer.lemmatize(token, pos='n')

def normalize_phrase(phrase):
    """Tokenize, normalize (using the function above), and reassemble a phrase.
       This groups singular and plural forms together.
    """
    tokens = word_tokenize(phrase.lower())
    norm_tokens = [normalize_token(t) for t in tokens if t.isalnum()]
    return " ".join(norm_tokens)

def extract_candidate_tags(candidate):
    """
    Given a candidate theme (a string), use POS-tagging to recommend:
      - A:Tag (Category): chosen as the rightmost noun.
      - B:Tag: the first adjective (if any).
      - C:Tag: all tokens before the head noun.
      - D:Tag: all tokens after the head noun.
    If no noun is found, fallback by using the candidate phrase as is.
    Duplicate tokens (if equal to the category) are suppressed.
    Returns a tuple: (A_tag, B_tag, C_tag, D_tag).
    """
    tokens = word_tokenize(candidate)
    tagged = pos_tag(tokens)
    
    # Collect normalized tokens for consistency.
    norm_tokens = [normalize_token(t) for t in tokens if t.isalnum()]
    norm_tagged = list(zip(norm_tokens, [tag for word, tag in tagged if word.isalnum()]))
    
    # Find adjectives (tags starting with JJ) and nouns (tags starting with NN)
    adjectives = [word for word, tag in norm_tagged if tag.startswith("JJ")]
    nouns = [word for word, tag in norm_tagged if tag.startswith("NN")]
    
    # For A:Tag, choose the rightmost noun if available.
    if nouns:
        a_tag = nouns[-1]
        # Find index of the rightmost occurrence of that noun in norm_tokens.
        idx = len(norm_tokens) - 1 - norm_tokens[::-1].index(a_tag)
    else:
        a_tag = candidate  # fallback
        idx = len(norm_tokens) - 1

    # B:Tag: choose the first adjective (if any)
    b_tag = adjectives[0] if adjectives else ""
    
    # C:Tag: tokens before the chosen noun.
    c_tag = " ".join(norm_tokens[:idx]) if idx > 0 else ""
    # D:Tag: tokens after the chosen noun.
    d_tag = " ".join(norm_tokens[idx+1:]) if idx+1 < len(norm_tokens) else ""
    
    # Suppress duplicate occurrences.
    if b_tag == a_tag:
        b_tag = ""
    if c_tag == a_tag:
        c_tag = ""
    if d_tag == a_tag or d_tag == b_tag:
        d_tag = ""
    return a_tag, b_tag, c_tag, d_tag

# --- Initialize KeyBERT ---
# Make sure these lines are included in your script.
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
kw_model = KeyBERT(model=embedding_model)

# --- Streamlit App for Theme Extraction and Tag Recommendation ---

st.title("KeyBERT Theme Extraction and Tag Recommendation")
st.markdown(
    """
    This tool uses KeyBERT to extract candidate themes from your keyword list,
    groups singular and plural forms together, and then provides recommended
    tag components for each candidate theme (A:Tag, B:Tag, C:Tag, D:Tag).
    
    **Workflow:**
    1. Upload a CSV/Excel file with a **Keywords** column.
    2. (Optionally) Limit processing to the first N keywords.
    3. Specify how many keyphrases per keyword to extract.
    4. Set a minimum frequency threshold (themes occurring less frequently will be ignored).
    5. (Optionally) Specify a number of clusters to group similar candidate themes.
    
    Review the candidate themes (and their frequencies) along with the recommended tag structure.
    Use these recommendations to inform your final programmatic tagging rules.
    """
)

# User inputs
uploaded_file = st.file_uploader("Upload CSV/Excel File", type=["csv", "xls", "xlsx"])
num_keywords = st.number_input("Process first N keywords (0 = all)", min_value=0, value=0, step=1)
top_n = st.number_input("Keyphrases per keyword", min_value=1, value=3, step=1)
min_freq = st.number_input("Minimum frequency for candidate theme", min_value=1, value=2, step=1)
num_clusters = st.number_input("Number of clusters (0 to skip clustering)", min_value=0, value=0, step=1)

if uploaded_file:
    # Load file into a DataFrame
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Error loading file: {e}")
        st.stop()
    
    # Check that the file has a "Keywords" column.
    if "Keywords" not in df.columns:
        st.error("The file must contain a column named 'Keywords'.")
    else:
        keywords_list = df["Keywords"].tolist()
        # If num_keywords is specified (non-zero), restrict to that many.
        if num_keywords > 0:
            keywords_list = keywords_list[:num_keywords]
        
        st.write("Extracting keyphrases from keywords...")
        all_phrases = []
        progress_bar = st.progress(0)
        
        # Process each keyword in the list.
        for idx, kw in enumerate(keywords_list):
            # Extract keyphrases for the current keyword.
            # We extract keyphrases of lengths 1 to 3 and take the top_n candidates.
            keyphrases = kw_model.extract_keywords(kw, keyphrase_ngram_range=(1, 3), stop_words='english', top_n=top_n)
            # keyphrases is a list of tuples (phrase, score)
            extracted = [kp[0].lower() for kp in keyphrases if kp[0]]
            all_phrases.extend(extracted)
            progress_bar.progress((idx + 1) / len(keywords_list))
        progress_bar.empty()
        
        # Count the frequency of each candidate theme.
        phrase_counts = Counter(all_phrases)
        # Only keep those with frequency >= min_freq.
        candidate_themes = {phrase: freq for phrase, freq in phrase_counts.items() if freq >= min_freq}
        
        st.write("### Candidate Themes and Frequencies")
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
        
        # Optionally, cluster the candidate themes if a positive number of clusters is provided.
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
            Review the candidate themes and the recommended tag structure above.  
            These suggestions can serve as a starting point for designing your final
            programmatic tagging system.
            """
        )
