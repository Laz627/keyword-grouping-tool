# Set page config first
import streamlit as st
st.set_page_config(
    page_title="Maximized Keyword Tagging Tool with MSV",
    page_icon="🏷️📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

import nltk
import os
import shutil

# --- NLTK Data Path Configuration ---
try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE_DIR = os.getcwd()
NLTK_DATA_DIR = os.path.join(BASE_DIR, "nltk_data")
if not os.path.exists(NLTK_DATA_DIR):
    try: os.makedirs(NLTK_DATA_DIR)
    except OSError: pass
if os.path.isdir(NLTK_DATA_DIR) and NLTK_DATA_DIR not in nltk.data.path:
    nltk.data.path.insert(0, NLTK_DATA_DIR)

nltk_resource_ids = ["punkt", "averaged_perceptron_tagger", "wordnet", "stopwords"]
resource_verified_paths = {
    "punkt": "tokenizers/punkt/english.pickle",
    "averaged_perceptron_tagger": "taggers/averaged_perceptron_tagger/averaged_perceptron_tagger.pickle",
    "wordnet": "corpora/wordnet.zip", # NLTK handles unzipping
    "stopwords": "corpora/stopwords.zip"
}
all_nltk_resources_ok = True
for res_id in nltk_resource_ids:
    try:
        nltk.data.find(resource_verified_paths[res_id])
    except LookupError:
        st.sidebar.warning(f"NLTK '{res_id}' not found. Downloading to {NLTK_DATA_DIR}...")
        try:
            nltk.download(res_id, download_dir=NLTK_DATA_DIR, quiet=False, raise_on_error=True)
            nltk.data.find(resource_verified_paths[res_id])
            st.sidebar.success(f"NLTK '{res_id}' ready.")
        except Exception as e_dl:
            st.sidebar.error(f"Failed to load/download NLTK '{res_id}': {e_dl}")
            all_nltk_resources_ok = False
if not all_nltk_resources_ok:
    st.error("A critical NLTK resource failed to load. App cannot continue.")
    st.stop()
# --- End NLTK Data Path Configuration ---

import pandas as pd
import re
from collections import Counter
import numpy as np
from keybert import KeyBERT
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords as nltk_stopwords
from nltk import word_tokenize as nltk_word_tokenize_func, pos_tag as nltk_pos_tag_func
from sentence_transformers import SentenceTransformer
import gc

try:
    STOP_WORDS_ENGLISH = set(nltk_stopwords.words('english'))
    LEMMATIZER_INSTANCE = WordNetLemmatizer()
    nltk_word_tokenize_func("test")
    nltk_pos_tag_func(nltk_word_tokenize_func("test"))
    st.sidebar.success("NLTK components initialized.")
except Exception as e_init:
    st.error(f"Failed to initialize NLTK components: {e_init}. App cannot continue.")
    st.stop()

CUSTOMIZABLE_STOP_WORDS = set(STOP_WORDS_ENGLISH)
DOMAIN_SPECIFIC_BC_STOPWORDS = {
    "style", "type", "series", "version", "model", "standard", "item",
    "product", "new", "best", "top", "custom", "design", "feature", "collection"
}
# CUSTOMIZABLE_STOP_WORDS.update(DOMAIN_SPECIFIC_BC_STOPWORDS) # User will update via UI

if 'full_tagging_processed' not in st.session_state:
    st.session_state.full_tagging_processed = False

@st.cache_resource
def load_tagging_models():
    # ... (model loading code remains the same) ...
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
        st.error("KeyBERT model for tagging could not be loaded.")
        return None
    return st.session_state.tagging_models_dict['kw_model_st']

def normalize_token(token_str):
    token_lower = token_str.lower()
    if token_lower == "vs": return "v"
    return LEMMATIZER_INSTANCE.lemmatize(token_lower, pos='n')

def normalize_phrase(phrase_str):
    if not isinstance(phrase_str, str): return ""
    tokens = [t.strip('.,;:!?()[]{}"\'') for t in phrase_str.lower().split()]
    return " ".join(normalize_token(t) for t in tokens if t.isalnum() or t.isnumeric())

def canonicalize_phrase(phrase_str):
    if not isinstance(phrase_str, str): return ""
    tokens = [t.strip('.,;:!?()[]{}"\'') for t in phrase_str.lower().split()]
    norm = [normalize_token(t) for t in tokens if (t.isalnum() or t.isnumeric()) and normalize_token(t) != "series"]
    return " ".join(norm).replace("_", " ")

def get_pos_filtered_normalized_tokens(text_segment, user_a_tags_for_bc_filter):
    if not text_segment or not isinstance(text_segment, str): return []
    tokens_raw = nltk_word_tokenize_func(text_segment)
    tokens_with_pos = nltk_pos_tag_func(tokens_raw)
    attribute_candidates = []
    for token, tag in tokens_with_pos:
        if tag.startswith('NN') or tag.startswith('JJ'):
            normalized_token = normalize_token(token)
            if normalized_token not in CUSTOMIZABLE_STOP_WORDS and \
               (len(normalized_token) > 1 or normalized_token.isnumeric()):
                is_an_a_tag = normalized_token in user_a_tags_for_bc_filter
                attribute_candidates.append({"token": normalized_token, "is_a_tag": is_an_a_tag})
    attribute_candidates.sort(key=lambda x: x["is_a_tag"])
    return [cand["token"] for cand in attribute_candidates]

def assign_b_c_tags(attribute_tokens):
    b_tag = attribute_tokens[0] if len(attribute_tokens) >= 1 else ""
    c_tag = attribute_tokens[1] if len(attribute_tokens) >= 2 else ""
    if b_tag and b_tag == c_tag:
        c_tag = attribute_tokens[2] if len(attribute_tokens) >= 3 else ""
    if b_tag and b_tag == c_tag: c_tag = ""
    return b_tag, c_tag

def classify_keyword_maximized(keyword_str, seed_to_remove_str, other_omitted_list_str, user_a_tags_set_runtime, kw_model_runtime):
    # ... (classify_keyword_maximized logic remains the same) ...
    if kw_model_runtime is None: return "error-model", "error-model", "error-model"
    if not isinstance(keyword_str, str) or not keyword_str.strip(): return "general-other", "", ""

    original_kw_lower = keyword_str.lower()
    identified_a_tag = "general-other"
    text_for_bc_processing = original_kw_lower

    sorted_user_a_tags = sorted(list(user_a_tags_set_runtime), key=len, reverse=True)
    for a_tag_cand in sorted_user_a_tags:
        if f" {a_tag_cand} " in f" {original_kw_lower} " or \
           original_kw_lower.startswith(f"{a_tag_cand} ") or \
           original_kw_lower.endswith(f" {a_tag_cand}") or \
           original_kw_lower == a_tag_cand:
            identified_a_tag = a_tag_cand
            text_for_bc_processing = re.sub(rf'\b{re.escape(a_tag_cand)}\b', '', text_for_bc_processing, flags=re.IGNORECASE).strip()
            break
    
    if seed_to_remove_str and seed_to_remove_str.lower() != identified_a_tag:
        text_for_bc_processing = re.sub(rf'\b{re.escape(seed_to_remove_str.lower())}\b', '', text_for_bc_processing, flags=re.IGNORECASE).strip()
    for omit_phrase in other_omitted_list_str:
        text_for_bc_processing = re.sub(rf'\b{re.escape(omit_phrase.lower())}\b', '', text_for_bc_processing, flags=re.IGNORECASE).strip()
    text_for_bc_processing = ' '.join(text_for_bc_processing.split())

    common_product_terms = {"door", "doors", "cabinet", "cabinets", "panel", "panels", "window", "windows"}
    words_in_remaining_text = text_for_bc_processing.lower().split()
    normalized_remaining_words = [normalize_token(w) for w in words_in_remaining_text]

    b_tag, c_tag = "", ""

    if identified_a_tag != "general-other" and len(normalized_remaining_words) == 1 and normalized_remaining_words[0] in common_product_terms:
        b_tag = normalized_remaining_words[0]
        c_tag = "" 
        return identified_a_tag, b_tag, c_tag
    
    if not text_for_bc_processing:
        return identified_a_tag, "", "" 

    keyphrases = kw_model_runtime.extract_keywords(text_for_bc_processing, keyphrase_ngram_range=(1,3), stop_words=CUSTOMIZABLE_STOP_WORDS, top_n=1)
    
    attribute_tokens_for_bc = []
    if keyphrases:
        keybert_candidate_phrase = keyphrases[0][0].lower()
        attribute_tokens_for_bc = get_pos_filtered_normalized_tokens(keybert_candidate_phrase, user_a_tags_set_runtime)
    
    if not attribute_tokens_for_bc:
        attribute_tokens_for_bc = get_pos_filtered_normalized_tokens(text_for_bc_processing, user_a_tags_set_runtime)
        
    b_tag, c_tag = assign_b_c_tags(attribute_tokens_for_bc)
    return identified_a_tag, b_tag, c_tag

def realign_tags_maximized(df, col_name="B:Tag", other_col="C:Tag"):
    # ... (realign_tags_maximized logic remains the same) ...
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
st.title("🏷️ Maximized Keyword Tagging Tool with MSV")
st.markdown("Processes keywords to assign A:Tag, B:Tag, and C:Tag, and aggregates Monthly Search Volume (MSV).")

# --- MSV Column Name Input ---
msv_column_name = st.sidebar.text_input("Column Name for Monthly Searches in your file", "Monthly Searches")
st.sidebar.info(f"The tool will look for a column named '{msv_column_name}' for search volume data.")

col1, col2 = st.columns(2)
with col1:
    seed_input = st.text_input("Primary Seed Keyword to Remove (optional)", "", key="tag_seed_msv")
    omit_str_input = st.text_input("Other Phrases to Omit (comma-separated)", "buy,price,cost,for sale,near me,online,cheap,best,top", key="tag_omit_msv")
with col2:
    user_atags_str_input = st.text_input("Define Primary Categories (A:Tags, comma-separated)", "door,window,cabinet,panel,shaker,flat,raised,wood,metal,glass,interior,exterior,kitchen,bathroom", key="tag_a_tags_main_msv")
    do_realign_input = st.checkbox("Re-align B/C tags by frequency", value=True, key="tag_realign_msv")

st.sidebar.subheader("B/C Tag Stopwords")
st.sidebar.info("Define terms that should NOT become B or C tags.")
custom_bc_stopwords_input = st.sidebar.text_area("Custom B/C Stopwords (comma-separated)", ", ".join(sorted(list(DOMAIN_SPECIFIC_BC_STOPWORDS))))
if custom_bc_stopwords_input:
    CUSTOMIZABLE_STOP_WORDS = set(STOP_WORDS_ENGLISH)
    CUSTOMIZABLE_STOP_WORDS.update([normalize_token(sw.strip()) for sw in custom_bc_stopwords_input.split(',') if sw.strip()])
else:
    CUSTOMIZABLE_STOP_WORDS = set(STOP_WORDS_ENGLISH)

omitted_list_runtime = [x.strip().lower() for x in omit_str_input.split(",") if x.strip()]
user_a_tags_runtime = set(normalize_token(x.strip()) for x in user_atags_str_input.split(",") if x.strip())

initial_rule_file_input = st.file_uploader("Upload Initial Tagging Rule CSV (Optional)", type=["csv"], key="tag_rule_file_msv")
use_initial_rule_input = st.checkbox("Use Initial Tagging Rule if available", value=False, key="use_tag_rule_msv")
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
    if st.button("Process New File or Change Settings", key="reset_tagging_msv"):
        st.session_state.full_tagging_processed = False
        if 'df_tagged_output' in st.session_state: del st.session_state.df_tagged_output
        if 'summary_ab_output' in st.session_state: del st.session_state.summary_ab_output
        st.experimental_rerun()

if not st.session_state.get('full_tagging_processed', False):
    uploaded_file_tagging = st.file_uploader("Upload Keyword File with MSV (CSV/Excel)", type=["csv", "xls", "xlsx"], key="tag_file_main_msv")
    if uploaded_file_tagging:
        kw_model_for_processing = get_tagging_keybert_model()
        if kw_model_for_processing is None: st.error("Tagging model not loaded. Cannot proceed."); st.stop()
        try:
            df_input = pd.read_csv(uploaded_file_tagging) if uploaded_file_tagging.name.endswith(".csv") else pd.read_excel(uploaded_file_tagging)
        except Exception as e_file: st.error(f"Error reading keyword file: {e_file}"); st.stop()
        
        required_cols = ["Keywords", msv_column_name]
        if not all(col in df_input.columns for col in required_cols):
            st.error(f"Uploaded file must have '{required_cols[0]}' and '{required_cols[1]}' columns.")
            st.stop()

        # Prepare DataFrame, ensuring MSV is numeric and handling errors
        df_processing = df_input[required_cols].copy()
        df_processing["Keywords"] = df_processing["Keywords"].dropna().astype(str)
        
        # Convert MSV column to numeric, coercing errors to NaN, then fill NaN with 0
        df_processing[msv_column_name] = pd.to_numeric(df_processing[msv_column_name], errors='coerce').fillna(0).astype(int)
        
        df_processing = df_processing[df_processing["Keywords"].str.strip() != ""].reset_index(drop=True)
        if df_processing.empty: st.warning("No valid keywords found after cleaning."); st.stop()

        with st.spinner(f"Tagging {len(df_processing)} keywords..."):
            gc.collect()
            A_list, B_list, C_list = [], [], []
            progress_bar_tagging = st.progress(0)
            for i, row_data in df_processing.iterrows(): # Iterate rows to keep MSV aligned
                kw_item = row_data["Keywords"]
                canon_kw_for_rule = canonicalize_phrase(normalize_phrase(kw_item))
                if use_initial_rule_input and canon_kw_for_rule in initial_rule_mapping_runtime:
                    a, b, c = initial_rule_mapping_runtime[canon_kw_for_rule]
                else:
                    a, b, c = classify_keyword_maximized(kw_item, seed_input, omitted_list_runtime, user_a_tags_runtime, kw_model_for_processing)
                A_list.append(a); B_list.append(b); C_list.append(c)
                progress_bar_tagging.progress((i + 1) / len(df_processing))
            progress_bar_tagging.empty()
            
            df_processing["A:Tag"] = A_list
            df_processing["B:Tag"] = B_list
            df_processing["C:Tag"] = C_list
            
            if do_realign_input:
                with st.spinner("Re-aligning B/C tags..."):
                    df_processing = realign_tags_maximized(df_processing, "B:Tag", "C:Tag")
            
            df_processing["A+B Combo"] = df_processing["A:Tag"].fillna("") + " - " + df_processing["B:Tag"].fillna("")
            
            # Aggregate for summary: count keywords and sum MSV
            summary_ab_current = df_processing.groupby("A+B Combo").agg(
                Keyword_Count=(msv_column_name, 'count'), # Count of keywords
                Total_MSV=(msv_column_name, 'sum')      # Sum of MSV
            ).reset_index()
            summary_ab_current = summary_ab_current.sort_values("Total_MSV", ascending=False) # Sort by MSV
            
            st.session_state.full_tagging_processed = True
            st.session_state.df_tagged_output = df_processing
            st.session_state.summary_ab_output = summary_ab_current

if st.session_state.get('full_tagging_processed', False):
    df_display = st.session_state.get('df_tagged_output', pd.DataFrame())
    summary_ab_display = st.session_state.get('summary_ab_output', pd.DataFrame())
    
    if not df_display.empty:
        st.subheader("Tagged Keywords with Monthly Search Volume")
        # Display MSV in the main table
        display_cols = ["Keywords", msv_column_name, "A:Tag", "B:Tag", "C:Tag"]
        # Ensure msv_column_name is actually in df_display before trying to show it
        if msv_column_name not in df_display.columns: # Should not happen if input validation worked
            st.error(f"Error: MSV column '{msv_column_name}' not found in processed data for display.")
        else:
             st.dataframe(df_display[[col for col in display_cols if col in df_display.columns]]) # Show only existing cols

        try:
            csv_tagged = df_display.to_csv(index=False).encode('utf-8')
            st.download_button(label="Download Full Tagged Keywords CSV (with MSV)", data=csv_tagged, file_name="tagged_keywords_msv.csv", mime="text/csv", key="dl_full_tags_btn_msv")
        except Exception as e_dl_full: st.error(f"Error preparing full tagged data for download: {e_dl_full}")

    if not summary_ab_display.empty:
        st.subheader("Tag Summary (A:Tag - B:Tag Combinations) with MSV")
        st.dataframe(summary_ab_display)
        try:
            csv_summary = summary_ab_display.to_csv(index=False).encode('utf-8')
            st.download_button(label="Download Tag Summary CSV (with MSV)", data=csv_summary, file_name="tag_summary_msv.csv", mime="text/csv", key="dl_tag_summary_btn_msv")
        except Exception as e_dl_summary: st.error(f"Error preparing tag summary for download: {e_dl_summary}")
    
    if df_display.empty and summary_ab_display.empty: st.info("No tagging results. Upload a file.")
