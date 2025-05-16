# Set page config first
import streamlit as st
st.set_page_config(
    page_title="Keyword Tagging Tool", # Changed title
    page_icon="🏷️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import other libraries
import pandas as pd
import re
from collections import Counter
import numpy as np
# import json # Not needed for tagging only
# import matplotlib.pyplot as plt # Not needed
from keybert import KeyBERT
import nltk
from nltk.stem import WordNetLemmatizer
# import openai # Not needed for tagging only
from sentence_transformers import SentenceTransformer
# from sklearn.cluster import DBSCAN, KMeans # Not needed
# from sklearn.metrics.pairwise import cosine_similarity # Not needed
# from tenacity import retry, stop_after_attempt, wait_exponential # Not needed for tagging
import gc
# import concurrent.futures # Not needed for tagging only

# NLTK Resource Downloads (essential for tagging)
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True) # May not be strictly needed by current pick_tags_b_c
nltk.download('wordnet', quiet=True)
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

lemmatizer = WordNetLemmatizer()

# Initialize session state variable for tagging
if 'full_tagging_processed' not in st.session_state:
    st.session_state.full_tagging_processed = False

# --- Model Loading (Simplified for Tagging) ---
@st.cache_resource
def load_tagging_models():
    models = {}
    try:
        # KeyBERT needs an embedding model. SentenceTransformer is good.
        sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
        models['kw_model_st'] = KeyBERT(model=sbert_model)
        models['models_loaded_successfully'] = True
    except Exception as e:
        st.error(f"Error loading SentenceTransformer/KeyBERT models: {e}")
        # Fallback for KeyBERT if SentenceTransformer fails
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            class SimpleKeyBERT_for_Tagging: # Renamed to avoid conflict if other script parts were present
                def __init__(self, model=None): pass # model is unused
                def extract_keywords(self, doc, keyphrase_ngram_range=(1,1), stop_words=None, top_n=5, **kwargs):
                    from sklearn.feature_extraction.text import CountVectorizer
                    import numpy as np
                    try:
                        sw_list = list(stop_words) if stop_words else None
                        count_model = CountVectorizer(ngram_range=keyphrase_ngram_range, stop_words=sw_list).fit([doc])
                        candidates = count_model.get_feature_names_out()
                        if not candidates.any(): return []

                        doc_vectorizer = TfidfVectorizer(vocabulary=candidates)
                        doc_tfidf = doc_vectorizer.fit_transform([doc])
                        
                        num_candidates = doc_tfidf.shape[1]
                        actual_top_n = min(top_n, num_candidates)
                        if actual_top_n == 0: return []

                        word_idx = np.argsort(doc_tfidf.toarray()[0])[-actual_top_n:]
                        scores = doc_tfidf.toarray()[0][word_idx]
                        
                        sorted_indices = np.argsort(scores)[::-1]
                        words = [candidates[word_idx[i]] for i in sorted_indices]
                        final_scores = scores[sorted_indices]
                        
                        return list(zip(words, final_scores))
                    except ValueError: return []

            models['kw_model_st'] = SimpleKeyBERT_for_Tagging()
            models['models_loaded_successfully'] = True
            st.warning("Using simplified TF-IDF based KeyBERT due to issues loading SentenceTransformer.")
        except Exception as e2:
            st.error(f"Failed to create fallback TF-IDF KeyBERT: {e2}")
            models['models_loaded_successfully'] = False
    return models

def get_tagging_keybert_model():
    if 'tagging_models_dict' not in st.session_state:
        st.session_state.tagging_models_dict = load_tagging_models()
    
    if not st.session_state.tagging_models_dict.get('models_loaded_successfully', False):
        st.error("KeyBERT model for tagging could not be loaded. Functionality will be limited.")
        return None
        
    return st.session_state.tagging_models_dict['kw_model_st']

# --- Text Preprocessing and Tagging Helper Functions ---
def normalize_token(token):
    token = token.lower()
    if token == "vs": token = "v"
    return lemmatizer.lemmatize(token, pos='n')

def normalize_phrase(phrase):
    if not isinstance(phrase, str): return ""
    tokens = [t.strip('.,;:!?()[]{}"\'') for t in phrase.lower().split()]
    return " ".join(normalize_token(t) for t in tokens if t.isalnum() or t.isnumeric())

def canonicalize_phrase(phrase): # Used by initial rule mapping if present
    if not isinstance(phrase, str): return ""
    tokens = [t.strip('.,;:!?()[]{}"\'') for t in phrase.lower().split()]
    norm = [normalize_token(t) for t in tokens if (t.isalnum() or t.isnumeric()) and normalize_token(t) != "series"]
    return " ".join(norm).replace("_", " ")

def pick_tags_b_c(tokens):
    """Assigns B and C tags from a list of remaining tokens."""
    filtered = [t for t in tokens if t.lower() not in stop_words and (len(t) > 1 or t.isnumeric()) and t.strip() != ""]
    b_tag = filtered[0] if len(filtered) >= 1 else ""
    c_tag = filtered[1] if len(filtered) >= 2 else ""
    return b_tag, c_tag

def classify_keyword_three_tags(keyword, seed_to_remove, other_omitted_list, user_a_tags_set, kw_model_runtime):
    """
    Classifies a keyword into A, B, and C tags.
    `kw_model_runtime` is the loaded KeyBERT model.
    """
    if kw_model_runtime is None:
        return "error-model", "error-model", "error-model"

    if not isinstance(keyword, str) or not keyword.strip():
        return "general-other", "", ""

    original_kw_lower = keyword.lower()
    identified_a_tag = "general-other"
    text_for_bc = original_kw_lower

    # 1. Identify A-tag from user_a_tags_set
    sorted_user_a_tags = sorted(list(user_a_tags_set), key=len, reverse=True)
    for a_tag_candidate in sorted_user_a_tags:
        # Check if the candidate A-tag is a whole word in the keyword
        if f" {a_tag_candidate} " in f" {original_kw_lower} " or \
           original_kw_lower.startswith(f"{a_tag_candidate} ") or \
           original_kw_lower.endswith(f" {a_tag_candidate}") or \
           original_kw_lower == a_tag_candidate:
            identified_a_tag = a_tag_candidate
            # Remove identified A-tag (as a whole word) for B,C processing
            text_for_bc = re.sub(rf'\b{re.escape(a_tag_candidate)}\b', '', text_for_bc, flags=re.IGNORECASE).strip()
            break # Found A-tag, no need to check others
    
    # 2. Remove seed_to_remove (if any and if different from identified A-tag)
    if seed_to_remove and seed_to_remove.lower() != identified_a_tag: # Avoid double removal
        text_for_bc = re.sub(rf'\b{re.escape(seed_to_remove.lower())}\b', '', text_for_bc, flags=re.IGNORECASE).strip()

    # 3. Remove other_omitted_list
    for omit_phrase in other_omitted_list:
        text_for_bc = re.sub(rf'\b{re.escape(omit_phrase.lower())}\b', '', text_for_bc, flags=re.IGNORECASE).strip()
    
    text_for_bc = ' '.join(text_for_bc.split()) # Normalize whitespace

    # 4. Use KeyBERT on remaining text to find candidate phrase for B, C tags
    if not text_for_bc: # If nothing left after removals
        return identified_a_tag, "", ""

    # Use NLTK's stop_words set for KeyBERT
    keyphrases = kw_model_runtime.extract_keywords(text_for_bc, keyphrase_ngram_range=(1,3), stop_words=stop_words, top_n=1)
    
    if not keyphrases:
        # Fallback: if KeyBERT returns nothing, try to get B, C from remaining text directly
        remaining_tokens = [token for token in text_for_bc.split() if token.strip()]
        b_tag, c_tag = pick_tags_b_c(remaining_tokens)
        return identified_a_tag, b_tag, c_tag

    candidate_for_bc = keyphrases[0][0].lower() # Keyphrases: [(text, score), ...]
    # No need to normalize/canonicalize again if KeyBERT output is reasonably clean
    tokens_for_bc = [t for t in candidate_for_bc.split() if t.strip()]
    
    b_tag, c_tag = pick_tags_b_c(tokens_for_bc)
    return identified_a_tag, b_tag, c_tag

def realign_tags_based_on_frequency(df, col_name="B:Tag", other_col="C:Tag"):
    """Post-processes B and C tags based on their overall frequency."""
    freq_in_col = Counter()
    freq_in_other = Counter()
    
    for _, row in df.iterrows():
        bval = row[col_name]
        oval = row[other_col]
        if isinstance(bval, str) and bval: 
            for token in bval.split(): freq_in_col[token] += 1
        if isinstance(oval, str) and oval: 
            for token in oval.split(): freq_in_other[token] += 1

    unify_map = {} # Maps token to its preferred column (col_name or other_col)
    all_tokens = set(freq_in_col.keys()) | set(freq_in_other.keys())
    for tok in all_tokens:
        c_freq = freq_in_col[tok]
        o_freq = freq_in_other[tok]
        if o_freq > c_freq:
            unify_map[tok] = other_col
        else: 
            unify_map[tok] = col_name

    new_b_col, new_o_col = [], []
    for _, row in df.iterrows():
        b_tokens = row[col_name].split() if isinstance(row[col_name], str) and row[col_name] else []
        o_tokens = row[other_col].split() if isinstance(row[other_col], str) and row[other_col] else []
        
        current_row_b_list, current_row_o_list = [], []
        
        # Process B tokens first
        for t in b_tokens:
            if unify_map.get(t, col_name) == col_name:
                current_row_b_list.append(t)
            else:
                current_row_o_list.append(t)
        
        # Process O tokens, ensuring they go to their designated preferred column
        # and avoid duplication if already placed from B_tokens.
        for t in o_tokens:
            if unify_map.get(t, other_col) == col_name: # Should go to B column
                 if t not in current_row_b_list: current_row_b_list.append(t)
            else: # Should go to C column
                 if t not in current_row_o_list: current_row_o_list.append(t)

        new_b_col.append(current_row_b_list[0] if current_row_b_list else "")
        new_o_col.append(current_row_o_list[0] if current_row_o_list else "")
        
    df[col_name] = new_b_col
    df[other_col] = new_o_col
    return df

# --- Main Streamlit UI for Full Tagging ---
st.title("🏷️ Keyword Tagging Tool")
st.markdown("""
This tool processes each keyword to assign it to categories using a three-tag system:
- **A:Tag** - Primary category (e.g., window, door)
- **B:Tag** - Secondary attribute or modifier
- **C:Tag** - Additional attribute
""")

# Settings in a more compact layout
col1, col2 = st.columns(2)
with col1:
    seed_input = st.text_input("Primary Seed Keyword to Remove (e.g., 'door', optional)", "", key="tag_seed_input")
    omit_str_input = st.text_input("Other Phrases to Omit (comma-separated, e.g., 'buy,price,cost')", "", key="tag_omit_input")
with col2:
    user_atags_str_input = st.text_input("Define Primary Categories (A:Tags, comma-separated, e.g., 'door,window,cabinet')", "door,window,cabinet,panel", key="tag_a_tags_main_input")
    do_realign_input = st.checkbox("Re-align B/C tags by frequency", value=True, key="tag_realign_input", help="Ensures consistent tag placement based on overall frequency.")

# Prepare lists/sets from inputs
omitted_list_runtime = [x.strip().lower() for x in omit_str_input.split(",") if x.strip()]
user_a_tags_runtime = set(normalize_token(x.strip()) for x in user_atags_str_input.split(",") if x.strip())

# Initial Tagging Rule (Optional)
initial_rule_file_input = st.file_uploader("Upload Initial Tagging Rule CSV (Optional)", type=["csv"], key="tag_rule_file_input", help="CSV with 'Candidate Theme', 'A:Tag', 'B:Tag', 'C:Tag' columns.")
use_initial_rule_input = st.checkbox("Use Initial Tagging Rule if available", value=False, key="use_tag_rule_input")

# Load initial rule mapping if provided
initial_rule_mapping_runtime = {}
if use_initial_rule_input and initial_rule_file_input is not None:
    try:
        rule_df = pd.read_csv(initial_rule_file_input)
        rule_df = rule_df.fillna('') # Replace NaN with empty strings
        # Expected columns: Candidate Theme, A:Tag, B:Tag, C:Tag.
        for _, row_rule in rule_df.iterrows():
            candidate_theme_rule = str(row_rule.get("Candidate Theme", "")) # Use .get for safety
            if candidate_theme_rule: # Only process if candidate theme is present
                canon_candidate_rule = canonicalize_phrase(normalize_phrase(candidate_theme_rule))
                initial_rule_mapping_runtime[canon_candidate_rule] = (
                    str(row_rule.get("A:Tag", "")),
                    str(row_rule.get("B:Tag", "")),
                    str(row_rule.get("C:Tag", ""))
                )
        if initial_rule_mapping_runtime:
            st.success(f"Loaded {len(initial_rule_mapping_runtime)} initial tagging rules.")
        else:
            st.warning("Initial rule file was provided but no valid rules could be loaded. Check file format and content.")
    except Exception as e_rule:
        st.error(f"Error reading initial tagging rule file: {e_rule}")
        initial_rule_mapping_runtime = {} # Ensure it's empty on error

# Reset button
if 'full_tagging_processed' in st.session_state and st.session_state.full_tagging_processed:
    if st.button("Process New File or Change Settings", key="reset_tagging_button"):
        st.session_state.full_tagging_processed = False
        # Clear previous results from session state
        if 'df_tagged_output' in st.session_state: del st.session_state.df_tagged_output
        if 'summary_ab_output' in st.session_state: del st.session_state.summary_ab_output
        st.experimental_rerun()

# File uploader and processing logic (only if not already processed)
if not st.session_state.get('full_tagging_processed', False):
    uploaded_file_tagging = st.file_uploader("Upload Keyword File (CSV/Excel)", type=["csv", "xls", "xlsx"], key="tag_file_main_uploader")
    
    if uploaded_file_tagging:
        # Load KeyBERT model here, only when a file is uploaded and ready for processing
        kw_model_for_processing = get_tagging_keybert_model()
        if kw_model_for_processing is None:
            st.error("Tagging model could not be loaded. Please check error messages above. Cannot proceed.")
            st.stop()

        try:
            df_input = pd.read_csv(uploaded_file_tagging) if uploaded_file_tagging.name.endswith(".csv") else pd.read_excel(uploaded_file_tagging)
        except Exception as e_file:
            st.error(f"Error reading keyword file: {e_file}")
            st.stop()

        if "Keywords" not in df_input.columns:
            st.error("The uploaded file must have a 'Keywords' column.")
            st.stop()

        # Prepare DataFrame for processing
        df_processing = df_input[["Keywords"]].copy() # Work with a copy of the relevant column
        df_processing["Keywords"] = df_processing["Keywords"].dropna().astype(str) # Ensure string type and drop NaNs
        df_processing = df_processing[df_processing["Keywords"].str.strip() != ""].reset_index(drop=True) # Remove empty strings

        if df_processing.empty:
            st.warning("No valid keywords found in the uploaded file after cleaning.")
            st.stop()

        with st.spinner(f"Tagging {len(df_processing)} keywords... This may take a moment."):
            gc.collect() # Garbage collect before memory-intensive loop
            A_list, B_list, C_list = [], [], []
            progress_bar_tagging = st.progress(0)
            keywords_to_process_list = df_processing["Keywords"].tolist()

            for i, kw_item in enumerate(keywords_to_process_list):
                # Check if keyword matches an initial rule first
                canon_kw_for_rule = canonicalize_phrase(normalize_phrase(kw_item))
                if use_initial_rule_input and canon_kw_for_rule in initial_rule_mapping_runtime:
                    a, b, c = initial_rule_mapping_runtime[canon_kw_for_rule]
                else:
                    # Pass the loaded KeyBERT model to the classification function
                    a, b, c = classify_keyword_three_tags(kw_item, seed_input, omitted_list_runtime, user_a_tags_runtime, kw_model_for_processing)
                
                A_list.append(a)
                B_list.append(b)
                C_list.append(c)
                progress_bar_tagging.progress((i + 1) / len(keywords_to_process_list))
            progress_bar_tagging.empty()
    
            df_processing["A:Tag"] = A_list
            df_processing["B:Tag"] = B_list
            df_processing["C:Tag"] = C_list
    
            if do_realign_input:
                with st.spinner("Re-aligning B/C tags based on frequency..."):
                    df_processing = realign_tags_based_on_frequency(df_processing, "B:Tag", "C:Tag")
    
            # Summary Report: A:Tag & B:Tag combination
            df_processing["A+B Combo"] = df_processing["A:Tag"].fillna("") + " - " + df_processing["B:Tag"].fillna("")
            summary_ab_current = df_processing.groupby("A+B Combo").size().reset_index(name="Count")
            summary_ab_current = summary_ab_current.sort_values("Count", ascending=False)
            
            # Store results in session state
            st.session_state.full_tagging_processed = True
            st.session_state.df_tagged_output = df_processing # This DataFrame contains Keywords and the new Tag columns
            st.session_state.summary_ab_output = summary_ab_current

# Display results from session state if already processed
if st.session_state.get('full_tagging_processed', False):
    df_display = st.session_state.get('df_tagged_output', pd.DataFrame())
    summary_ab_display = st.session_state.get('summary_ab_output', pd.DataFrame())
    
    if not df_display.empty:
        st.subheader("Tagged Keywords")
        # Display only relevant columns
        st.dataframe(df_display[["Keywords", "A:Tag", "B:Tag", "C:Tag"]])
        
        # Download button for full tagged data
        try:
            csv_tagged = df_display.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Full Tagged Keywords CSV",
                data=csv_tagged,
                file_name="tagged_keywords.csv",
                mime="text/csv",
                key="download_full_tags_button"
            )
        except Exception as e_dl_full:
            st.error(f"Error preparing full tagged data for download: {e_dl_full}")

    if not summary_ab_display.empty:
        st.subheader("Tag Summary (A:Tag - B:Tag Combinations)")
        st.dataframe(summary_ab_display)
        
        # Download button for tag summary
        try:
            csv_summary = summary_ab_display.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Tag Summary CSV",
                data=csv_summary,
                file_name="tag_summary.csv",
                mime="text/csv",
                key="download_tag_summary_button"
            )
        except Exception as e_dl_summary:
            st.error(f"Error preparing tag summary for download: {e_dl_summary}")
    
    if df_display.empty and summary_ab_display.empty:
        st.info("No tagging results to display. Please upload a file and process.")
