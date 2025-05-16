# Set page config first
import streamlit as st
st.set_page_config(
    page_title="Enhanced Keyword Tagging Tool",
    page_icon="🏷️✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

import nltk
import os

# --- NLTK Data Path Configuration ---
# Determine the base directory (directory of the current script)
# This is generally robust for Streamlit deployments.
try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError: # __file__ is not defined in certain interactive environments (like a raw Python interpreter)
    BASE_DIR = os.getcwd() # Fallback to current working directory

nltk_data_dir = os.path.join(BASE_DIR, "nltk_data")

# Create the directory if it doesn't exist
if not os.path.exists(nltk_data_dir):
    try:
        os.makedirs(nltk_data_dir)
        # st.sidebar.info(f"NLTK data directory created at: {nltk_data_dir}") # Optional: for debugging
    except OSError as e:
        st.sidebar.error(f"Could not create NLTK data directory at {nltk_data_dir}: {e}")
        # Fallback strategy or stop if NLTK data is absolutely critical
        # For now, we'll let it try to use default paths if creation fails.
        # NLTK might still find data if it's elsewhere.
        pass # Allow script to continue, NLTK will try default paths

# Add your custom path to NLTK's list of search paths
# This ensures NLTK looks in your specified directory first.
if os.path.exists(nltk_data_dir) and nltk_data_dir not in nltk.data.path:
    nltk.data.path.insert(0, nltk_data_dir) # Insert at the beginning for higher priority

# Now try to download to this specific path if resources are missing or verify they exist
nltk_resources = {
    'punkt': 'tokenizers/punkt',
    'averaged_perceptron_tagger': 'taggers/averaged_perceptron_tagger',
    'wordnet': 'corpora/wordnet', # Or 'corpora/omw-1.4' for newer NLTK with WordNet 2022
    'stopwords': 'corpora/stopwords'
}

for resource_name, resource_path in nltk_resources.items():
    try:
        nltk.data.find(resource_path)
        # st.sidebar.success(f"NLTK resource '{resource_name}' found.") # Optional: for debugging
    except LookupError:
        st.sidebar.warning(f"NLTK resource '{resource_name}' not found. Attempting download to {nltk_data_dir}...")
        try:
            nltk.download(resource_name, download_dir=nltk_data_dir, quiet=True, raise_on_error=True)
            st.sidebar.success(f"NLTK resource '{resource_name}' downloaded successfully.")
        except Exception as e_nltk_download:
            st.sidebar.error(f"Failed to download NLTK resource '{resource_name}' to {nltk_data_dir}: {e_nltk_download}")
            st.sidebar.error("Please ensure this path is writable or manually place NLTK resources.")
            # Depending on how critical, you might st.stop()
# --- End NLTK Data Path Configuration ---


# Import other libraries
import pandas as pd
import re
from collections import Counter
import numpy as np
from keybert import KeyBERT
from nltk.stem import WordNetLemmatizer # NLTK Stem imports after path config
from nltk.corpus import stopwords      # NLTK Corpus imports after path config
from sentence_transformers import SentenceTransformer
import gc

# stop_words can now be defined as NLTK path should be set
stop_words = set(stopwords.words('english'))
# --- Optional: Add domain-specific stopwords for B/C tagging ---
# custom_bc_stopwords = {"style", "type", "series", "version", "model"}
# stop_words.update(custom_bc_stopwords)
# ---

lemmatizer = WordNetLemmatizer() # Initialize after path config

# Initialize session state variable for tagging
if 'full_tagging_processed' not in st.session_state:
    st.session_state.full_tagging_processed = False

# --- Model Loading (Simplified for Tagging) ---
@st.cache_resource
def load_tagging_models():
    # ... (rest of the script from load_tagging_models onwards) ...
    # NO CHANGES NEEDED TO THE REST OF THE SCRIPT BELOW THIS POINT
    # The NLTK data path setup is done at the very beginning.
    models = {}
    try:
        sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
        models['kw_model_st'] = KeyBERT(model=sbert_model)
        models['models_loaded_successfully'] = True
    except Exception as e:
        st.error(f"Error loading SentenceTransformer/KeyBERT models: {e}")
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            class SimpleKeyBERT_for_Tagging:
                def __init__(self, model=None): pass
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
    token_lower = token.lower()
    if token_lower == "vs": return "v"
    return lemmatizer.lemmatize(token_lower, pos='n')

def normalize_phrase(phrase):
    if not isinstance(phrase, str): return ""
    tokens = [t.strip('.,;:!?()[]{}"\'') for t in phrase.lower().split()]
    return " ".join(normalize_token(t) for t in tokens if t.isalnum() or t.isnumeric())

def canonicalize_phrase(phrase): # Used by initial rule mapping if present
    if not isinstance(phrase, str): return ""
    tokens = [t.strip('.,;:!?()[]{}"\'') for t in phrase.lower().split()]
    norm = [normalize_token(t) for t in tokens if (t.isalnum() or t.isnumeric()) and normalize_token(t) != "series"]
    return " ".join(norm).replace("_", " ")

def pick_tags_b_c_from_tokens_pos(tokens_with_pos_tags):
    potential_bc_tags = []
    for token, tag in tokens_with_pos_tags:
        if tag.startswith('NN') or tag.startswith('JJ'):
            normalized = normalize_token(token) 
            if normalized not in stop_words and (len(normalized) > 1 or normalized.isnumeric()):
                potential_bc_tags.append(normalized)
    b_tag = potential_bc_tags[0] if len(potential_bc_tags) >= 1 else ""
    c_tag = potential_bc_tags[1] if len(potential_bc_tags) >= 2 else ""
    return b_tag, c_tag

def classify_keyword_three_tags_enhanced(keyword, seed_to_remove, other_omitted_list, user_a_tags_set, kw_model_runtime):
    from nltk import pos_tag as nltk_pos_tag, word_tokenize as nltk_word_tokenize 

    if kw_model_runtime is None: return "error-model", "error-model", "error-model"
    if not isinstance(keyword, str) or not keyword.strip(): return "general-other", "", ""

    original_kw_lower = keyword.lower()
    identified_a_tag = "general-other"
    text_for_bc = original_kw_lower

    sorted_user_a_tags = sorted(list(user_a_tags_set), key=len, reverse=True)
    for a_tag_candidate in sorted_user_a_tags:
        if f" {a_tag_candidate} " in f" {original_kw_lower} " or \
           original_kw_lower.startswith(f"{a_tag_candidate} ") or \
           original_kw_lower.endswith(f" {a_tag_candidate}") or \
           original_kw_lower == a_tag_candidate:
            identified_a_tag = a_tag_candidate
            text_for_bc = re.sub(rf'\b{re.escape(a_tag_candidate)}\b', '', text_for_bc, flags=re.IGNORECASE).strip()
            break
    
    if seed_to_remove and seed_to_remove.lower() != identified_a_tag:
        text_for_bc = re.sub(rf'\b{re.escape(seed_to_remove.lower())}\b', '', text_for_bc, flags=re.IGNORECASE).strip()
    for omit_phrase in other_omitted_list:
        text_for_bc = re.sub(rf'\b{re.escape(omit_phrase.lower())}\b', '', text_for_bc, flags=re.IGNORECASE).strip()
    text_for_bc = ' '.join(text_for_bc.split())

    common_product_terms = {"door", "doors", "cabinet", "cabinets", "panel", "panels", "window", "windows"} 
    normalized_text_for_bc = " ".join(normalize_token(t) for t in text_for_bc.lower().split())
    
    if normalized_text_for_bc in common_product_terms and identified_a_tag != "general-other":
        return identified_a_tag, "", normalize_token(text_for_bc.lower().strip()) 

    if not text_for_bc: return identified_a_tag, "", ""

    keyphrases = kw_model_runtime.extract_keywords(text_for_bc, keyphrase_ngram_range=(1,3), stop_words=stop_words, top_n=1)
    
    b_tag, c_tag = "", ""
    if keyphrases:
        candidate_for_bc = keyphrases[0][0].lower()
        tokens_from_candidate = nltk_word_tokenize(candidate_for_bc)
        tagged_candidate_tokens = nltk_pos_tag(tokens_from_candidate)
        b_tag, c_tag = pick_tags_b_c_from_tokens_pos(tagged_candidate_tokens)
        if not b_tag and tokens_from_candidate:
            simple_bc_tokens = [normalize_token(t) for t in tokens_from_candidate 
                                if normalize_token(t) not in stop_words and (len(normalize_token(t)) > 1 or normalize_token(t).isnumeric())]
            b_tag = simple_bc_tokens[0] if len(simple_bc_tokens) >= 1 else ""
            c_tag = simple_bc_tokens[1] if len(simple_bc_tokens) >= 2 else ""
    else:
        tokens_from_text_for_bc = nltk_word_tokenize(text_for_bc)
        tagged_text_for_bc_tokens = nltk_pos_tag(tokens_from_text_for_bc)
        b_tag, c_tag = pick_tags_b_c_from_tokens_pos(tagged_text_for_bc_tokens)
    return identified_a_tag, b_tag, c_tag

def realign_tags_based_on_frequency(df, col_name="B:Tag", other_col="C:Tag"):
    freq_in_col = Counter()
    freq_in_other = Counter()
    for _, row in df.iterrows():
        bval, oval = row[col_name], row[other_col]
        if isinstance(bval, str) and bval: 
            for token in bval.split(): freq_in_col[token] += 1
        if isinstance(oval, str) and oval: 
            for token in oval.split(): freq_in_other[token] += 1
    unify_map = {} 
    all_tokens = set(freq_in_col.keys()) | set(freq_in_other.keys())
    for tok in all_tokens:
        unify_map[tok] = other_col if freq_in_other[tok] > freq_in_col[tok] else col_name
    new_b_col, new_o_col = [], []
    for _, row in df.iterrows():
        b_tokens = row[col_name].split() if isinstance(row[col_name], str) and row[col_name] else []
        o_tokens = row[other_col].split() if isinstance(row[other_col], str) and row[other_col] else []
        current_row_b_list, current_row_o_list = [], []
        for t in b_tokens:
            if unify_map.get(t, col_name) == col_name: current_row_b_list.append(t)
            else: current_row_o_list.append(t)
        for t in o_tokens:
            if unify_map.get(t, other_col) == col_name: 
                 if t not in current_row_b_list: current_row_b_list.append(t) 
            else:
                 if t not in current_row_o_list: current_row_o_list.append(t)
        new_b_col.append(current_row_b_list[0] if current_row_b_list else "")
        new_o_col.append(current_row_o_list[0] if current_row_o_list else "")
    df[col_name], df[other_col] = new_b_col, new_o_col
    return df

# --- Main Streamlit UI for Full Tagging ---
st.title("🏷️ Enhanced Keyword Tagging Tool")
st.markdown("Processes keywords to assign A:Tag (Primary Category), B:Tag (Attribute 1), and C:Tag (Attribute 2).")

col1, col2 = st.columns(2)
with col1:
    seed_input = st.text_input("Primary Seed Keyword to Remove (e.g., 'door', optional)", "", key="tag_seed_input_enhanced")
    omit_str_input = st.text_input("Other Phrases to Omit (comma-separated, e.g., 'buy,price')", "buy,price,cost,for sale,near me,online", key="tag_omit_input_enhanced")
with col2:
    user_atags_str_input = st.text_input("Define Primary Categories (A:Tags, comma-separated)", "door,window,cabinet,panel,shaker,flat,raised,wood,metal,glass", key="tag_a_tags_main_input_enhanced")
    do_realign_input = st.checkbox("Re-align B/C tags by frequency", value=True, key="tag_realign_input_enhanced")

omitted_list_runtime = [x.strip().lower() for x in omit_str_input.split(",") if x.strip()]
user_a_tags_runtime = set(normalize_token(x.strip()) for x in user_atags_str_input.split(",") if x.strip())

initial_rule_file_input = st.file_uploader("Upload Initial Tagging Rule CSV (Optional)", type=["csv"], key="tag_rule_file_input_enhanced")
use_initial_rule_input = st.checkbox("Use Initial Tagging Rule if available", value=False, key="use_tag_rule_input_enhanced")

initial_rule_mapping_runtime = {}
if use_initial_rule_input and initial_rule_file_input is not None:
    try:
        rule_df = pd.read_csv(initial_rule_file_input).fillna('')
        for _, row_rule in rule_df.iterrows():
            candidate_theme_rule = str(row_rule.get("Candidate Theme", ""))
            if candidate_theme_rule:
                canon_candidate_rule = canonicalize_phrase(normalize_phrase(candidate_theme_rule))
                initial_rule_mapping_runtime[canon_candidate_rule] = (
                    str(row_rule.get("A:Tag", "")), str(row_rule.get("B:Tag", "")), str(row_rule.get("C:Tag", "")))
        if initial_rule_mapping_runtime: st.success(f"Loaded {len(initial_rule_mapping_runtime)} initial tagging rules.")
        else: st.warning("Initial rule file provided but no valid rules loaded.")
    except Exception as e_rule: st.error(f"Error reading initial tagging rule file: {e_rule}"); initial_rule_mapping_runtime = {}

if 'full_tagging_processed' in st.session_state and st.session_state.full_tagging_processed:
    if st.button("Process New File or Change Settings", key="reset_tagging_button_enhanced"):
        st.session_state.full_tagging_processed = False
        if 'df_tagged_output' in st.session_state: del st.session_state.df_tagged_output
        if 'summary_ab_output' in st.session_state: del st.session_state.summary_ab_output
        st.experimental_rerun()

if not st.session_state.get('full_tagging_processed', False):
    uploaded_file_tagging = st.file_uploader("Upload Keyword File (CSV/Excel)", type=["csv", "xls", "xlsx"], key="tag_file_main_uploader_enhanced")
    if uploaded_file_tagging:
        kw_model_for_processing = get_tagging_keybert_model()
        if kw_model_for_processing is None: st.error("Tagging model not loaded. Cannot proceed."); st.stop()
        try:
            df_input = pd.read_csv(uploaded_file_tagging) if uploaded_file_tagging.name.endswith(".csv") else pd.read_excel(uploaded_file_tagging)
        except Exception as e_file: st.error(f"Error reading keyword file: {e_file}"); st.stop()
        if "Keywords" not in df_input.columns: st.error("Uploaded file must have 'Keywords' column."); st.stop()

        df_processing = df_input[["Keywords"]].copy()
        df_processing["Keywords"] = df_processing["Keywords"].dropna().astype(str)
        df_processing = df_processing[df_processing["Keywords"].str.strip() != ""].reset_index(drop=True)
        if df_processing.empty: st.warning("No valid keywords found after cleaning."); st.stop()

        with st.spinner(f"Tagging {len(df_processing)} keywords..."):
            gc.collect()
            A_list, B_list, C_list = [], [], []
            progress_bar_tagging = st.progress(0)
            keywords_to_process_list = df_processing["Keywords"].tolist()
            for i, kw_item in enumerate(keywords_to_process_list):
                canon_kw_for_rule = canonicalize_phrase(normalize_phrase(kw_item))
                if use_initial_rule_input and canon_kw_for_rule in initial_rule_mapping_runtime:
                    a, b, c = initial_rule_mapping_runtime[canon_kw_for_rule]
                else:
                    a, b, c = classify_keyword_three_tags_enhanced(kw_item, seed_input, omitted_list_runtime, user_a_tags_runtime, kw_model_for_processing)
                A_list.append(a); B_list.append(b); C_list.append(c)
                progress_bar_tagging.progress((i + 1) / len(keywords_to_process_list))
            progress_bar_tagging.empty()
            df_processing["A:Tag"], df_processing["B:Tag"], df_processing["C:Tag"] = A_list, B_list, C_list
            if do_realign_input:
                with st.spinner("Re-aligning B/C tags..."):
                    df_processing = realign_tags_based_on_frequency(df_processing, "B:Tag", "C:Tag")
            df_processing["A+B Combo"] = df_processing["A:Tag"].fillna("") + " - " + df_processing["B:Tag"].fillna("")
            summary_ab_current = df_processing.groupby("A+B Combo").size().reset_index(name="Count").sort_values("Count", ascending=False)
            st.session_state.full_tagging_processed = True
            st.session_state.df_tagged_output = df_processing
            st.session_state.summary_ab_output = summary_ab_current

if st.session_state.get('full_tagging_processed', False):
    df_display = st.session_state.get('df_tagged_output', pd.DataFrame())
    summary_ab_display = st.session_state.get('summary_ab_output', pd.DataFrame())
    if not df_display.empty:
        st.subheader("Tagged Keywords")
        st.dataframe(df_display[["Keywords", "A:Tag", "B:Tag", "C:Tag"]])
        try:
            csv_tagged = df_display.to_csv(index=False).encode('utf-8')
            st.download_button(label="Download Full Tagged Keywords CSV", data=csv_tagged, file_name="tagged_keywords_enhanced.csv", mime="text/csv", key="dl_full_tags_btn_enh")
        except Exception as e_dl_full: st.error(f"Error preparing full tagged data for download: {e_dl_full}")
    if not summary_ab_display.empty:
        st.subheader("Tag Summary (A:Tag - B:Tag Combinations)")
        st.dataframe(summary_ab_display)
        try:
            csv_summary = summary_ab_display.to_csv(index=False).encode('utf-8')
            st.download_button(label="Download Tag Summary CSV", data=csv_summary, file_name="tag_summary_enhanced.csv", mime="text/csv", key="dl_tag_summary_btn_enh")
        except Exception as e_dl_summary: st.error(f"Error preparing tag summary for download: {e_dl_summary}")
    if df_display.empty and summary_ab_display.empty: st.info("No tagging results. Upload a file.")
