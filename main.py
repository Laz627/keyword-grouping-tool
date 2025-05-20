# Set page config first - THIS MUST BE THE FIRST STREAMLIT COMMAND
import streamlit as st
st.set_page_config(
    page_title="Advanced Keyword Tagging with MSV & Semantic Insights",
    page_icon="🏷️🧠📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

import os
import torch # Import torch early
if hasattr(torch, 'classes') and hasattr(torch.classes, '__path__'):
    torch.classes.__path__ = []

import nltk
import shutil
import openai # Added for OpenAI embeddings
from tenacity import retry, stop_after_attempt, wait_exponential # For OpenAI API calls

# --- NLTK Data Path Configuration ---
try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError: BASE_DIR = os.getcwd()
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
    "wordnet": "corpora/wordnet.zip",
    "stopwords": "corpora/stopwords.zip"}
all_nltk_resources_ok = True
for res_id in nltk_resource_ids:
    try: nltk.data.find(resource_verified_paths[res_id])
    except LookupError:
        st.sidebar.warning(f"NLTK '{res_id}' not found. Downloading...")
        try:
            nltk.download(res_id, download_dir=NLTK_DATA_DIR, quiet=False, raise_on_error=True)
            nltk.data.find(resource_verified_paths[res_id])
            st.sidebar.success(f"NLTK '{res_id}' ready.")
        except Exception as e_dl:
            st.sidebar.error(f"Failed to load/download NLTK '{res_id}': {e_dl}"); all_nltk_resources_ok = False
if not all_nltk_resources_ok: st.error("A critical NLTK resource failed. App cannot continue."); st.stop()
# --- End NLTK Data Path Configuration ---

import pandas as pd
import re
from collections import Counter
import numpy as np
from keybert import KeyBERT
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords as nltk_stopwords
from nltk import word_tokenize as nltk_word_tokenize_func, pos_tag as nltk_pos_tag_func
from sentence_transformers import SentenceTransformer, util as sbert_util # Added util for community_detection
import gc

try:
    STOP_WORDS_ENGLISH = set(nltk_stopwords.words('english'))
    LEMMATIZER_INSTANCE = WordNetLemmatizer()
    nltk_word_tokenize_func("test"); nltk_pos_tag_func(nltk_word_tokenize_func("test"))
    st.sidebar.success("NLTK components initialized.")
except Exception as e_init: st.error(f"Failed to initialize NLTK: {e_init}. App cannot continue."); st.stop()

CUSTOMIZABLE_STOP_WORDS = set(STOP_WORDS_ENGLISH)
DOMAIN_SPECIFIC_BC_STOPWORDS = {"style","type","series","version","model","standard","item","product","new","best","top","custom","design","feature","collection"}

if 'full_tagging_processed' not in st.session_state: st.session_state.full_tagging_processed = False
if 'semantic_clusters_generated' not in st.session_state: st.session_state.semantic_clusters_generated = False

# --- OpenAI Embedding Function ---
@st.cache_data(ttl=3600) # Cache OpenAI embeddings
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def get_openai_embeddings_for_clustering(texts, api_key, model="text-embedding-3-small"):
    openai.api_key = api_key
    if not texts: return np.array([])
    
    # Ensure all texts are strings and not empty, replace None or empty with a placeholder if necessary
    processed_texts = []
    for t in texts:
        if isinstance(t, str) and t.strip():
            processed_texts.append(t)
        else:
            # OpenAI API might error on empty strings, handle as needed or filter out.
            # For clustering, it's better to filter out completely empty keywords.
            # Here, we assume `texts` passed are already cleaned.
            processed_texts.append(str(t) if t is not None else "empty") # Ensure string

    if not processed_texts: return np.array([])

    all_embeddings = []
    # Max batch size for text-embedding-3-small is 2048 tokens per item, and 2048 items (though practically fewer due to token limits)
    # Let's use a smaller batch size for items to be safe and manage request size.
    batch_size_items = 100 
    with st.spinner(f"Getting OpenAI embeddings with {model} ({len(processed_texts)} texts)..."):
        prog_bar = st.progress(0, text="Embedding texts...")
        for i in range(0, len(processed_texts), batch_size_items):
            batch = processed_texts[i:min(i + batch_size_items, len(processed_texts))]
            try:
                response = openai.embeddings.create(model=model, input=batch)
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
            except Exception as e_openai:
                st.error(f"OpenAI API error during embedding batch {i//batch_size_items + 1}: {e_openai}")
                # Optionally, append None or zero vectors for failed items to maintain length,
                # or raise error to stop. For clustering, partial results might be okay.
                # For simplicity, we'll let it potentially create shorter all_embeddings list if a batch fails hard.
                # A more robust solution would handle per-item errors if API supports it.
                pass # Continue to next batch if one fails, or handle more gracefully
            prog_bar.progress(min(i + batch_size_items, len(processed_texts)) / len(processed_texts))
        prog_bar.empty()
    return np.array(all_embeddings)


@st.cache_resource
def load_models(): # Combined model loader
    models = {}
    # SBERT models for KeyBERT and potentially semantic clustering if OpenAI not used
    try:
        sbert_model = SentenceTransformer('all-MiniLM-L6-v2') # General purpose SBERT
        models['sbert_model'] = sbert_model
        models['kw_model_st'] = KeyBERT(model=sbert_model)
        models['models_loaded_successfully'] = True
    except Exception as e:
        st.error(f"Error loading SentenceTransformer/KeyBERT models: {e}")
        # Fallback for KeyBERT (as before)
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            class SimpleKeyBERT_for_Tagging: # Fallback KeyBERT
                def __init__(self, model=None): pass
                def extract_keywords(self, doc, keyphrase_ngram_range=(1,1), stop_words=None, top_n=5, **kwargs):
                    from sklearn.feature_extraction.text import CountVectorizer
                    import numpy as np
                    try:
                        sw_list = list(stop_words) if stop_words else None
                        count_model = CountVectorizer(ngram_range=keyphrase_ngram_range, stop_words=sw_list).fit([doc])
                        candidates = count_model.get_feature_names_out()
                        if not candidates.any(): return []
                        # ... (rest of SimpleKeyBERT implementation)
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
            models['sbert_model'] = None # Indicate SBERT itself failed
            models['models_loaded_successfully'] = True # Fallback KeyBERT loaded
            st.warning("Using simplified TF-IDF KeyBERT. SBERT model for clustering unavailable locally.")
        except Exception as e2:
            st.error(f"Failed to create fallback models: {e2}")
            models['models_loaded_successfully'] = False
    return models

def get_sbert_model():
    if 'all_models_dict' not in st.session_state:
        st.session_state.all_models_dict = load_models()
    if not st.session_state.all_models_dict.get('models_loaded_successfully', False):
        return None
    return st.session_state.all_models_dict.get('sbert_model')

def get_keybert_model():
    if 'all_models_dict' not in st.session_state:
        st.session_state.all_models_dict = load_models()
    if not st.session_state.all_models_dict.get('models_loaded_successfully', False):
        return None
    return st.session_state.all_models_dict.get('kw_model_st')


def normalize_token(token_str):
    # ... (same as before)
    token_lower = token_str.lower()
    if token_lower == "vs": return "v"
    return LEMMATIZER_INSTANCE.lemmatize(token_lower, pos='n')

def normalize_phrase(phrase_str):
    # ... (same as before)
    if not isinstance(phrase_str, str): return ""
    tokens = [t.strip('.,;:!?()[]{}"\'') for t in phrase_str.lower().split()]
    return " ".join(normalize_token(t) for t in tokens if t.isalnum() or t.isnumeric())

def canonicalize_phrase(phrase_str):
    # ... (same as before)
    if not isinstance(phrase_str, str): return ""
    tokens = [t.strip('.,;:!?()[]{}"\'') for t in phrase_str.lower().split()]
    norm = [normalize_token(t) for t in tokens if (t.isalnum() or t.isnumeric()) and normalize_token(t) != "series"]
    return " ".join(norm).replace("_", " ")


def get_pos_filtered_normalized_tokens(text_segment, user_a_tags_for_bc_filter):
    # ... (same as before)
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
    # ... (same as before)
    b_tag = attribute_tokens[0] if len(attribute_tokens) >= 1 else ""
    c_tag = attribute_tokens[1] if len(attribute_tokens) >= 2 else ""
    if b_tag and b_tag == c_tag:
        c_tag = attribute_tokens[2] if len(attribute_tokens) >= 3 else ""
    if b_tag and b_tag == c_tag: c_tag = ""
    return b_tag, c_tag


def classify_keyword_maximized(keyword_str, seed_to_remove_str, other_omitted_list_str, user_a_tags_set_runtime, kw_model_runtime):
    # ... (classify_keyword_maximized logic remains the same)
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
    
    if not attribute_tokens_for_bc: # Fallback to whole remaining text if KeyBERT phrase yields nothing
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

# --- Semantic Clustering Function (inspired by LeeFootSEO) ---
@st.cache_data # Cache the clustering result for a given set of keywords and params
def run_semantic_clustering(keywords_list, embedding_model_name_or_path, cluster_accuracy_percent, min_cluster_size_val, 
                            use_openai_for_clustering=False, openai_api_key_for_clustering=None, openai_model_for_clustering="text-embedding-3-small"):
    if not keywords_list:
        return pd.DataFrame(columns=["Semantic Cluster Name", "Keyword"]), 0, 0

    corpus_sentences = list(set(kw for kw in keywords_list if isinstance(kw, str) and kw.strip())) # Unique, non-empty keywords
    if not corpus_sentences:
        return pd.DataFrame(columns=["Semantic Cluster Name", "Keyword"]), 0, 0

    st.write(f"Embedding {len(corpus_sentences)} unique keywords for semantic clustering...")
    
    corpus_embeddings = None
    if use_openai_for_clustering and openai_api_key_for_clustering:
        corpus_embeddings = get_openai_embeddings_for_clustering(corpus_sentences, openai_api_key_for_clustering, openai_model_for_clustering)
    else:
        sbert_model_runtime = get_sbert_model() # Get the local SBERT model
        if sbert_model_runtime:
            corpus_embeddings = sbert_model_runtime.encode(corpus_sentences, batch_size=128, show_progress_bar=True, convert_to_tensor=False) # Keep as numpy for sbert_util
        else:
            st.error("Local SBERT model not available for semantic clustering.")
            return pd.DataFrame(columns=["Semantic Cluster Name", "Keyword"]), 0, 0
    
    if corpus_embeddings is None or len(corpus_embeddings) == 0:
        st.warning("Embeddings could not be generated. Cannot perform semantic clustering.")
        return pd.DataFrame(columns=["Semantic Cluster Name", "Keyword"]), 0, 0

    st.write("Performing community detection for clustering...")
    # The community_detection function expects a threshold between 0 and 1
    threshold_float = cluster_accuracy_percent / 100.0
    
    # Check if PyTorch tensor is needed (sbert_util.community_detection can take numpy or tensor)
    # If using OpenAI, embeddings are numpy. If SBERT, they are numpy by default unless convert_to_tensor=True
    # For util.community_detection, numpy is fine.
    
    clusters = sbert_util.community_detection(torch.tensor(corpus_embeddings) if not isinstance(corpus_embeddings, torch.Tensor) else corpus_embeddings, # Ensure tensor for PyTorch util
                                         min_community_size=min_cluster_size_val, 
                                         threshold=threshold_float)
    
    clustered_keywords_data = []
    clustered_keyword_set = set()

    for i, cluster in enumerate(clusters):
        cluster_name_temp = f"Semantic Cluster {i + 1}" # Temporary name
        shortest_keyword_in_cluster = ""
        min_len = float('inf')
        
        current_cluster_keywords = []
        for sentence_id in cluster:
            keyword = corpus_sentences[sentence_id]
            current_cluster_keywords.append(keyword)
            clustered_keyword_set.add(keyword)
            if len(keyword) < min_len:
                min_len = len(keyword)
                shortest_keyword_in_cluster = keyword
        
        # Use shortest keyword as cluster name (or another heuristic if preferred)
        final_cluster_name = shortest_keyword_in_cluster if shortest_keyword_in_cluster else cluster_name_temp
        
        for keyword in current_cluster_keywords:
            clustered_keywords_data.append({"Semantic Cluster Name": final_cluster_name, "Keyword": keyword})

    df_clustered = pd.DataFrame(clustered_keywords_data)
    
    # Add unclustered keywords
    unclustered_keywords = [kw for kw in corpus_sentences if kw not in clustered_keyword_set]
    if unclustered_keywords:
        df_unclustered = pd.DataFrame({"Semantic Cluster Name": "zzz_no_cluster", "Keyword": unclustered_keywords})
        df_clustered = pd.concat([df_clustered, df_unclustered], ignore_index=True)
        
    total_keywords_processed = len(corpus_sentences)
    num_actually_clustered = len(clustered_keyword_set)

    return df_clustered, total_keywords_processed, num_actually_clustered


# --- Main Streamlit UI for Full Tagging ---
st.title("🏷️ Advanced Keyword Tagging with MSV & Semantic Insights")
st.markdown("Processes keywords to assign A/B/C Tags, aggregates MSV, and offers optional semantic clustering to inform tagging.")

# --- OpenAI API Key (Moved to Sidebar for global access if needed) ---
openai_api_key_input = st.sidebar.text_input("OpenAI API Key (for optional semantic clustering)", type="password", key="openai_api_key_sidebar")

# --- MSV Column Name Input ---
msv_column_name = st.sidebar.text_input("Column Name for MSV in your file", "Monthly Searches", key="msv_col_name_sidebar")
st.sidebar.info(f"The tool will look for a column named '{msv_column_name}' for MSV data.")

# --- Tagging Settings (Main Page) ---
st.header("1. A/B/C Tagging Configuration")
col1_tag_cfg, col2_tag_cfg = st.columns(2)
with col1_tag_cfg:
    seed_input = st.text_input("Primary Seed Keyword to Remove (optional)", "", key="tag_seed_main")
    omit_str_input = st.text_input("Other Phrases to Omit (comma-separated)", "buy,price,cost,for sale,near me,online,cheap,best,top", key="tag_omit_main")
with col2_tag_cfg:
    user_atags_str_input = st.text_input("Define Primary Categories (A:Tags, comma-separated)", "door,window,cabinet,panel,shaker,flat,raised,wood,metal,glass,interior,exterior,kitchen,bathroom", key="tag_a_tags_main")
    do_realign_input = st.checkbox("Re-align B/C tags by frequency", value=True, key="tag_realign_main")

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

# --- Initial Tagging Rules (Main Page) ---
with st.expander("Optional: Initial Tagging Rules (CSV Upload)"):
    initial_rule_file_input = st.file_uploader("Upload Rule CSV", type=["csv"], key="tag_rule_file_main")
    use_initial_rule_input = st.checkbox("Use Initial Tagging Rule if available", value=False, key="use_tag_rule_main")
    initial_rule_mapping_runtime = {}
    if use_initial_rule_input and initial_rule_file_input is not None:
        try:
            rule_df = pd.read_csv(initial_rule_file_input).fillna('')
            for _, row_rule in rule_df.iterrows():
                ct_rule = str(row_rule.get("Candidate Theme", "")); a_rule=str(row_rule.get("A:Tag",""));b_rule=str(row_rule.get("B:Tag",""));c_rule=str(row_rule.get("C:Tag",""))
                if ct_rule: initial_rule_mapping_runtime[canonicalize_phrase(normalize_phrase(ct_rule))] = (a_rule,b_rule,c_rule)
            if initial_rule_mapping_runtime: st.success(f"Loaded {len(initial_rule_mapping_runtime)} initial rules.")
            else: st.warning("Rule file provided but no valid rules loaded.")
        except Exception as e_rule: st.error(f"Error reading rule file: {e_rule}"); initial_rule_mapping_runtime = {}

# --- File Upload (Main Page) ---
st.header("2. Upload Keyword Data")
uploaded_file_main = st.file_uploader("Upload Keyword File (CSV/Excel - must include 'Keywords' and MSV column)", type=["csv", "xls", "xlsx"], key="main_file_uploader")

# --- Optional Semantic Pre-Analysis (Expander) ---
if uploaded_file_main is not None and not st.session_state.get('full_tagging_processed', False): # Only show if file uploaded and not yet tagged
    with st.expander("Optional: Pre-Analysis - Semantic Keyword Clustering to Inform Tagging Settings", expanded=False):
        st.write("""
        This step can help you discover natural groupings in your keywords based on their meaning.
        The insights (common terms in clusters) can help you define better A:Tags, Omissions, or B/C Stopwords for the main tagging process below.
        This uses sentence embeddings and community detection.
        """)
        
        sem_cluster_model_choice = st.selectbox("Embedding Model for Pre-Analysis", 
                                                ("all-MiniLM-L6-v2 (Local, Faster)", 
                                                 "text-embedding-3-small (OpenAI, Good Quality)", 
                                                 "text-embedding-3-large (OpenAI, Higher Quality)"), 
                                                key="sem_cluster_model")
        
        use_openai_sem_clustering = "OpenAI" in sem_cluster_model_choice
        openai_model_name = sem_cluster_model_choice.split(" (")[0] if use_openai_sem_clustering else None

        col_sem1, col_sem2 = st.columns(2)
        with col_sem1:
            sem_cluster_accuracy = st.slider("Semantic Similarity Threshold (%)", 0, 100, 80, key="sem_accuracy")
        with col_sem2:
            sem_min_cluster_size = st.number_input("Min Keywords per Semantic Cluster", 2, 20, 3, key="sem_min_size")

        if st.button("Run Semantic Pre-Analysis", key="run_semantic_pre_analysis"):
            if use_openai_sem_clustering and not openai_api_key_input:
                st.error("OpenAI API Key is required for OpenAI embedding models.")
            else:
                # Read just keywords for this pre-analysis
                try:
                    df_sem_input = pd.read_csv(uploaded_file_main) if uploaded_file_main.name.endswith(".csv") else pd.read_excel(uploaded_file_main)
                    uploaded_file_main.seek(0) # Reset file pointer for later use by main tagging
                except Exception as e_sem_file: st.error(f"Error reading file for semantic analysis: {e_sem_file}"); st.stop()

                if "Keywords" not in df_sem_input.columns: st.error("File must have 'Keywords' column."); st.stop()
                
                keywords_for_sem_clustering = df_sem_input["Keywords"].dropna().astype(str).tolist()
                if not keywords_for_sem_clustering: st.warning("No keywords found for semantic pre-analysis."); st.stop()

                with st.spinner("Performing semantic clustering for pre-analysis... This may take a while."):
                    df_semantic_clusters, total_kws, num_clustered_kws = run_semantic_clustering(
                        keywords_for_sem_clustering,
                        embedding_model_name_or_path='all-MiniLM-L6-v2' if not use_openai_sem_clustering else openai_model_name,
                        cluster_accuracy_percent=sem_cluster_accuracy,
                        min_cluster_size_val=sem_min_cluster_size,
                        use_openai_for_clustering=use_openai_sem_clustering,
                        openai_api_key_for_clustering=openai_api_key_input,
                        openai_model_for_clustering=openai_model_name
                    )
                    st.session_state.df_semantic_clusters_preview = df_semantic_clusters
                    st.session_state.semantic_clusters_generated = True
                    
                    if not df_semantic_clusters.empty:
                        st.success(f"Semantic pre-analysis complete! {num_clustered_kws} of {total_kws} unique keywords grouped.")
                    else:
                        st.warning("Semantic pre-analysis did not result in any clusters or an error occurred.")
                        
    if st.session_state.get('semantic_clusters_generated', False) and 'df_semantic_clusters_preview' in st.session_state:
        st.subheader("Semantic Pre-Analysis Results (Top Clusters)")
        df_sem_preview = st.session_state.df_semantic_clusters_preview
        # Show a sample of clusters
        if not df_sem_preview.empty:
            # Display top N clusters by size (excluding 'zzz_no_cluster')
            cluster_counts = df_sem_preview[df_sem_preview['Semantic Cluster Name'] != 'zzz_no_cluster']['Semantic Cluster Name'].value_counts()
            st.write("Top clusters by number of keywords:")
            st.dataframe(cluster_counts.head(10))

            st.write("Example keywords from top clusters (use these insights to refine A:Tags, Omissions, etc. above):")
            for cluster_name_preview in cluster_counts.head(5).index:
                st.write(f"**{cluster_name_preview}** (Sample):")
                st.dataframe(df_sem_preview[df_sem_preview['Semantic Cluster Name'] == cluster_name_preview]['Keyword'].sample(min(5, len(df_sem_preview[df_sem_preview['Semantic Cluster Name'] == cluster_name_preview]))).reset_index(drop=True))
            st.download_button("Download Full Semantic Pre-Analysis CSV", df_sem_preview.to_csv(index=False).encode('utf-8'), "semantic_pre_analysis_clusters.csv", "text/csv", key="dl_sem_pre_analysis")
        else:
            st.info("No semantic clusters to display from pre-analysis.")


# --- Main Tagging Process Trigger ---
st.header("3. Run A/B/C Tagging")
if 'full_tagging_processed' in st.session_state and st.session_state.full_tagging_processed:
    if st.button("Re-Process File with Current Settings", key="reset_tagging_main_reprocess"): # Renamed for clarity
        st.session_state.full_tagging_processed = False
        if 'df_tagged_output' in st.session_state: del st.session_state.df_tagged_output
        if 'summary_ab_output' in st.session_state: del st.session_state.summary_ab_output
        st.experimental_rerun() # Rerun to allow reprocessing if file is still loaded or settings changed

if not st.session_state.get('full_tagging_processed', False):
    if uploaded_file_main is not None: # Only show process button if file is present
        if st.button("Run Full A/B/C Tagging Process", key="run_tagging_main_button"):
            kw_model_for_processing = get_keybert_model() # Changed function name
            if kw_model_for_processing is None: st.error("Tagging model not loaded."); st.stop()
            
            # Ensure file pointer is reset if it was read by semantic pre-analysis
            uploaded_file_main.seek(0)
            try:
                df_input = pd.read_csv(uploaded_file_main) if uploaded_file_main.name.endswith(".csv") else pd.read_excel(uploaded_file_main)
            except Exception as e_file: st.error(f"Error reading keyword file: {e_file}"); st.stop()
            
            required_cols = ["Keywords", msv_column_name]
            if not all(col in df_input.columns for col in required_cols):
                st.error(f"File must have '{required_cols[0]}' and '{required_cols[1]}' columns."); st.stop()

            df_processing = df_input[required_cols].copy()
            df_processing["Keywords"] = df_processing["Keywords"].dropna().astype(str)
            df_processing[msv_column_name] = pd.to_numeric(df_processing[msv_column_name], errors='coerce').fillna(0).astype(int)
            df_processing = df_processing[df_processing["Keywords"].str.strip() != ""].reset_index(drop=True)
            if df_processing.empty: st.warning("No valid keywords for tagging."); st.stop()

            with st.spinner(f"Tagging {len(df_processing)} keywords..."):
                # ... (Tagging loop and summary generation as before) ...
                gc.collect()
                A_list, B_list, C_list = [], [], []
                progress_bar_tagging = st.progress(0)
                for i, row_data in df_processing.iterrows(): 
                    kw_item = row_data["Keywords"]
                    canon_kw_for_rule = canonicalize_phrase(normalize_phrase(kw_item))
                    if use_initial_rule_input and canon_kw_for_rule in initial_rule_mapping_runtime:
                        a, b, c = initial_rule_mapping_runtime[canon_kw_for_rule]
                    else:
                        a, b, c = classify_keyword_maximized(kw_item, seed_input, omitted_list_runtime, user_a_tags_runtime, kw_model_for_processing)
                    A_list.append(a); B_list.append(b); C_list.append(c)
                    progress_bar_tagging.progress((i + 1) / len(df_processing))
                progress_bar_tagging.empty()
                
                df_processing["A:Tag"] = A_list; df_processing["B:Tag"] = B_list; df_processing["C:Tag"] = C_list
                if do_realign_input:
                    with st.spinner("Re-aligning B/C tags..."):
                        df_processing = realign_tags_maximized(df_processing, "B:Tag", "C:Tag")
                df_processing["A+B Combo"] = df_processing["A:Tag"].fillna("") + " - " + df_processing["B:Tag"].fillna("")
                summary_ab_current = df_processing.groupby("A+B Combo").agg(
                    Keyword_Count=(msv_column_name, 'count'), Total_MSV=(msv_column_name, 'sum')
                ).reset_index().sort_values("Total_MSV", ascending=False)
                
                st.session_state.full_tagging_processed = True
                st.session_state.df_tagged_output = df_processing
                st.session_state.summary_ab_output = summary_ab_current
    else:
        st.info("Please upload a keyword file to begin the tagging process.")


# --- Display Tagging Results ---
if st.session_state.get('full_tagging_processed', False):
    st.header("4. Tagging Results")
    df_display = st.session_state.get('df_tagged_output', pd.DataFrame())
    summary_ab_display = st.session_state.get('summary_ab_output', pd.DataFrame())
    
    if not df_display.empty:
        st.subheader("Tagged Keywords with Monthly Search Volume")
        display_cols = ["Keywords", msv_column_name, "A:Tag", "B:Tag", "C:Tag"]
        if msv_column_name not in df_display.columns: 
            st.error(f"MSV column '{msv_column_name}' not found in processed data.")
        else: st.dataframe(df_display[[col for col in display_cols if col in df_display.columns]])
        try:
            csv_tagged = df_display.to_csv(index=False).encode('utf-8')
            st.download_button(label="Download Full Tagged Keywords CSV (with MSV)", data=csv_tagged, file_name="tagged_keywords_msv_final.csv", mime="text/csv", key="dl_full_tags_btn_msv_final")
        except Exception as e_dl_full: st.error(f"Error preparing download: {e_dl_full}")

    if not summary_ab_display.empty:
        st.subheader("Tag Summary (A:Tag - B:Tag Combinations) with MSV")
        st.dataframe(summary_ab_display)
        try:
            csv_summary = summary_ab_display.to_csv(index=False).encode('utf-8')
            st.download_button(label="Download Tag Summary CSV (with MSV)", data=csv_summary, file_name="tag_summary_msv_final.csv", mime="text/csv", key="dl_tag_summary_btn_msv_final")
        except Exception as e_dl_summary: st.error(f"Error preparing download: {e_dl_summary}")
    
    if df_display.empty and summary_ab_display.empty: st.info("No tagging results to display.")
