import streamlit as st

# Set page config first
st.set_page_config(
    page_title="Keyword Tagging & Topic Generation Tool",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import other libraries
import pandas as pd
import re
from collections import Counter
import numpy as np
import json
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from keybert import KeyBERT
import nltk
from nltk.stem import WordNetLemmatizer
import openai
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics.pairwise import cosine_similarity
from tenacity import retry, stop_after_attempt, wait_exponential
import gc

# Try to import docx for Word document export
try:
    from docx import Document
    from io import BytesIO
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False

# Download NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

lemmatizer = WordNetLemmatizer()

# Initialize session state variables
if 'candidate_themes_processed' not in st.session_state:
    st.session_state.candidate_themes_processed = False
if 'full_tagging_processed' not in st.session_state:
    st.session_state.full_tagging_processed = False
if 'content_topics_processed' not in st.session_state:
    st.session_state.content_topics_processed = False

# Load models on demand using session state
def get_models():
    """Load models on demand to avoid Streamlit/PyTorch conflicts"""
    if 'models_loaded' not in st.session_state:
        with st.spinner("Loading NLP models... (this might take a moment)"):
            try:
                # Try to load models into session state
                st.session_state.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                st.session_state.kw_model = KeyBERT(model=st.session_state.embedding_model)
                st.session_state.models_loaded = True
            except Exception as e:
                st.error(f"Error loading models: {e}")
                
                # Use a simplified fallback approach
                try:
                    # Create a basic embedding model using TF-IDF as fallback
                    from sklearn.feature_extraction.text import TfidfVectorizer
                    from sklearn.metrics.pairwise import cosine_similarity
                    
                    class SimpleSentenceTransformer:
                        def __init__(self):
                            self.vectorizer = TfidfVectorizer()
                            self.fitted = False
                            
                        def encode(self, sentences, **kwargs):
                            if not isinstance(sentences, list):
                                sentences = [sentences]
                            
                            if not self.fitted:
                                vectors = self.vectorizer.fit_transform(sentences).toarray()
                                self.fitted = True
                            else:
                                vectors = self.vectorizer.transform(sentences).toarray()
                            return vectors
                    
                    class SimpleKeyBERT:
                        def __init__(self, model=None):
                            self.vectorizer = TfidfVectorizer(ngram_range=(1, 2))
                        
                        def extract_keywords(self, doc, keyphrase_ngram_range=(1, 1), stop_words=None, top_n=5, **kwargs):
                            # Simple extraction using TF-IDF
                            from sklearn.feature_extraction.text import CountVectorizer
                            import numpy as np
                            
                            # Extract candidate words/phrases
                            count = CountVectorizer(ngram_range=keyphrase_ngram_range, stop_words=stop_words).fit([doc])
                            candidates = count.get_feature_names_out()
                            
                            # Get TFIDF scores for candidates
                            doc_vectorizer = TfidfVectorizer(vocabulary=candidates)
                            doc_tfidf = doc_vectorizer.fit_transform([doc])
                            
                            # Get top candidates based on scores
                            word_idx = np.argsort(doc_tfidf.toarray()[0])[-top_n:]
                            scores = doc_tfidf.toarray()[0][word_idx]
                            
                            # Sort in descending order of scores
                            word_idx = word_idx[::-1]
                            scores = scores[::-1]
                            
                            # Get candidate names
                            words = [count.get_feature_names_out()[i] for i in word_idx]
                            
                            # Return results in KeyBERT's expected format
                            return [(words[i], scores[i]) for i in range(len(words))]
                    
                    st.session_state.embedding_model = SimpleSentenceTransformer()
                    st.session_state.kw_model = SimpleKeyBERT()
                    st.session_state.models_loaded = True
                    st.warning("Using simplified models with reduced functionality due to PyTorch issues")
                    
                except Exception as e2:
                    st.error(f"Failed to create fallback models: {e2}")
                    st.session_state.models_loaded = False
                    st.stop()
    
    return st.session_state.embedding_model, st.session_state.kw_model

# Get OpenAI embeddings function - UPDATED with caching to prevent recalculation
@st.cache_data(ttl=3600)
def get_cached_embeddings(texts, api_key, model="text-embedding-3-small"):
    """Cached version of embeddings to prevent recalculation"""
    return get_openai_embeddings(texts, api_key, model)

@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
def get_openai_embeddings(texts, api_key, model="text-embedding-3-small"):
    """Get embeddings from OpenAI with improved model"""
    openai.api_key = api_key
    
    # Process in batches (up to 100 embeddings per call)
    all_embeddings = []
    
    with st.spinner(f"Getting OpenAI embeddings with {model} (processing {len(texts)} texts in batches)..."):
        progress_bar = st.progress(0)
        for i in range(0, len(texts), 100):
            batch = texts[i:min(i+100, len(texts))]
            response = openai.embeddings.create(
                model=model,
                input=batch
            )
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)
            progress_bar.progress((i + len(batch)) / len(texts))
        
        progress_bar.empty()
    
    return np.array(all_embeddings)

# Cached enriched embeddings
@st.cache_data(ttl=3600)
def get_cached_enriched_embeddings(texts, tags, api_key, model="text-embedding-3-small"):
    """Cached version of enriched embeddings to prevent recalculation"""
    return get_enriched_openai_embeddings(texts, tags, api_key, model)

# Updated function to get enriched OpenAI embeddings
def get_enriched_openai_embeddings(texts, tags, api_key, model="text-embedding-3-small"):
    """Enhance embeddings by adding tag context with improved model"""
    enriched_texts = []
    
    # Combine keywords with their tags for richer context
    for i, text in enumerate(texts):
        # Fixed the order of condition checks to prevent KeyError
        a_tag = tags['A'][i] if 'A' in tags and i < len(tags['A']) else ""
        b_tag = tags['B'][i] if 'B' in tags and i < len(tags['B']) else ""
        c_tag = tags['C'][i] if 'C' in tags and i < len(tags['C']) else ""
        
        context = f"keyword: {text}"
        
        if a_tag:
            context += f", category: {a_tag}"
            
            if b_tag and c_tag:
                context += f", attributes: {b_tag} {c_tag}"
            elif b_tag:
                context += f", attribute: {b_tag}"
        
        enriched_texts.append(context)
    
    # Get embeddings using the enriched texts
    return get_openai_embeddings(enriched_texts, api_key, model)

# Helper function to determine optimal threshold based on embedding distribution
def get_optimal_threshold(embeddings, min_keywords=5, default_threshold=0.65):
    """Automatically determine optimal clustering threshold based on embedding distribution"""
    try:
        # Normalize embeddings
        normalized_embeddings = embeddings.copy()
        for i in range(len(normalized_embeddings)):
            norm = np.linalg.norm(normalized_embeddings[i])
            if norm > 0:
                normalized_embeddings[i] = normalized_embeddings[i] / norm
                
        # Sample if dataset is large
        if len(normalized_embeddings) > 300:
            sample_size = 300
            indices = np.random.choice(len(normalized_embeddings), sample_size, replace=False)
            sample_embeddings = normalized_embeddings[indices]
        else:
            sample_embeddings = normalized_embeddings
        
        # Calculate similarity matrix
        sim_matrix = cosine_similarity(sample_embeddings)
        flat_sim = sim_matrix[np.triu_indices(len(sim_matrix), k=1)]
        
        # Calculate statistics
        min_sim = np.min(flat_sim)
        mean_sim = np.mean(flat_sim)
        median_sim = np.median(flat_sim)
        p75 = np.percentile(flat_sim, 75)
        p90 = np.percentile(flat_sim, 90)
        
        # Determine strategy based on distribution
        if min_sim > 0.6:  # Compressed similarity range
            # More aggressive threshold for highly similar embeddings
            optimal = (median_sim + p75) / 2
        elif mean_sim > 0.75:  # Somewhat compressed
            # Use 65th percentile
            optimal = np.percentile(flat_sim, 65)
        else:  # Normal range
            # Use a value between mean and 75th percentile
            optimal = (mean_sim + p75) / 2
        
        return optimal
    except Exception as e:
        # Return default if analysis fails
        return default_threshold

# IMPROVED: Semantic intent-based clustering function for text-embedding-3-small
def semantic_intent_clustering(keywords, embeddings, min_shared_keywords=3, similarity_threshold=0.65):
    """Improved clustering with adaptive threshold for text-embedding-3-small"""
    from sklearn.cluster import DBSCAN, KMeans
    import numpy as np
    from scipy.cluster.hierarchy import linkage, fcluster
    
    # For very small datasets, just put everything in one cluster
    if len(keywords) < min_shared_keywords:
        return np.zeros(len(keywords), dtype=int), np.ones(len(keywords))
    
    # Normalize embeddings for better distance calculation
    normalized_embeddings = embeddings.copy()
    for i in range(len(normalized_embeddings)):
        norm = np.linalg.norm(normalized_embeddings[i])
        if norm > 0:
            normalized_embeddings[i] = normalized_embeddings[i] / norm
    
    # Use adaptive threshold if needed (for datasets with enough samples)
    if len(keywords) >= 10:
        adaptive_threshold = get_optimal_threshold(
            normalized_embeddings, 
            min_keywords=min_shared_keywords,
            default_threshold=similarity_threshold
        )
        # Only use adaptive threshold if it's reasonable
        if 0.4 <= adaptive_threshold <= 0.9:
            similarity_threshold = adaptive_threshold
    
    # Convert similarity threshold to distance
    eps = 1 - similarity_threshold
    
    # Try DBSCAN first - often works better for semantic embeddings
    dbscan = DBSCAN(eps=eps, min_samples=min_shared_keywords, metric='cosine')
    dbscan_labels = dbscan.fit_predict(normalized_embeddings)
    
    # Count clusters (excluding outliers)
    n_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
    
    # If DBSCAN fails to create enough clusters, try K-means with forced clusters
    if n_clusters <= 2:
        # Estimate a reasonable number of clusters - aim for 10-15 keywords per cluster
        cluster_size_target = min(15, max(8, min_shared_keywords * 2))
        n_kmeans_clusters = max(3, min(50, len(keywords) // cluster_size_target))
        
        kmeans = KMeans(n_clusters=n_kmeans_clusters, random_state=42, n_init=10)
        kmeans_labels = kmeans.fit_predict(normalized_embeddings)
        
        # Use K-means result if it created multiple clusters
        if len(set(kmeans_labels)) > 2:
            cluster_labels = kmeans_labels
        else:
            # Last resort: hierarchical clustering with lower threshold
            Z = linkage(normalized_embeddings, method='average', metric='cosine')
            
            # Try a more lenient threshold
            relaxed_threshold = max(0.3, similarity_threshold * 0.7)  # Don't go below 0.3
            distance_threshold = 1 - relaxed_threshold
            cluster_labels = fcluster(Z, t=distance_threshold, criterion='distance') - 1
    else:
        # DBSCAN gave good results, use those
        cluster_labels = dbscan_labels
    
    # Split any massive clusters (containing > 30% of all keywords)
    unique_clusters, cluster_sizes = np.unique(cluster_labels, return_counts=True)
    for cluster, size in zip(unique_clusters, cluster_sizes):
        if cluster == -1:  # Skip outliers
            continue
            
        if size > len(keywords) * 0.3:  # If cluster has >30% of all keywords
            # Get indices of this large cluster
            cluster_indices = np.where(cluster_labels == cluster)[0]
            
            # Extract embeddings for this cluster
            cluster_embeddings = normalized_embeddings[cluster_indices]
            
            # Force split into appropriate number of subclusters
            n_subclusters = max(3, size // 15)  # Aim for ~15 keywords per subcluster
            
            # Apply K-means to split
            sub_kmeans = KMeans(n_clusters=n_subclusters, random_state=42, n_init=10)
            sub_labels = sub_kmeans.fit_predict(cluster_embeddings)
            
            # Generate new cluster IDs - use max existing ID + 1, 2, 3...
            max_id = np.max(cluster_labels)
            for i, idx in enumerate(cluster_indices):
                cluster_labels[idx] = max_id + 1 + sub_labels[i]
    
    # Calculate cluster centers and confidence scores
    unique_clusters = np.unique(cluster_labels)
    cluster_centers = {}
    
    # Calculate cluster centers for non-outlier clusters
    for cluster in unique_clusters:
        if cluster == -1:
            continue
        cluster_indices = np.where(cluster_labels == cluster)[0]
        cluster_centers[cluster] = np.mean(normalized_embeddings[cluster_indices], axis=0)
    
    # Calculate confidence scores
    confidence_scores = np.full(len(keywords), 0.3)  # Base confidence for outliers
    
    for i, label in enumerate(cluster_labels):
        if label == -1:
            continue
        
        center = cluster_centers[label]
        embedding = normalized_embeddings[i]
        
        # Cosine similarity
        similarity = np.dot(center, embedding)
        confidence_scores[i] = 0.5 + (similarity * 0.5)
    
    return cluster_labels, confidence_scores

# Helper Functions for Tagging
def normalize_token(token):
    """Convert token to lowercase and lemmatize (noun mode); also converts 'vs' to 'v'."""
    token = token.lower()
    if token == "vs":
        token = "v"
    return lemmatizer.lemmatize(token, pos='n')

def normalize_phrase(phrase):
    """
    Lowercase, tokenize, keep only alphanumeric tokens, and lemmatize.
    E.g., 'Pella Windows Cost' becomes 'pella window cost'.
    """
    # Simple tokenization by whitespace instead of using word_tokenize
    tokens = [t.strip('.,;:!?()[]{}"\'') for t in phrase.lower().split()]
    return " ".join(normalize_token(t) for t in tokens if t.isalnum())

def canonicalize_phrase(phrase):
    """
    Remove unwanted tokens (e.g., "series") while preserving the original order.
    Also replace underscores with spaces.
    """
    # Simple tokenization by whitespace instead of using word_tokenize
    tokens = [t.strip('.,;:!?()[]{}"\'') for t in phrase.lower().split()]
    norm = [normalize_token(t) for t in tokens if t.isalnum() and normalize_token(t) != "series"]
    return " ".join(norm).replace("_", " ")

def pick_tags_pos_based(tokens, user_a_tags):
    """
    Given a list of candidate tokens (in original order), assign one-word tags for A, B, and C.
    
    Simplified version that doesn't depend on NLTK's POS tagging
    """
    # Flatten tokens (in case any token contains embedded whitespace)
    flat_tokens = []
    for token in tokens:
        flat_tokens.extend(token.split())
    tokens_copy = flat_tokens[:]  # Work on a copy

    a_tag = None
    a_index = None
    for i, token in enumerate(tokens_copy):
        for allowed in user_a_tags:
            if allowed in token or token in allowed:
                a_tag = allowed
                a_index = i
                break
        if a_tag is not None:
            break

    if a_tag is not None:
        tokens_copy.pop(a_index)
    else:
        a_tag = "general-other"

    # Filter out stopwords
    filtered = [t for t in tokens_copy if t.lower() not in stop_words and t.strip() != ""]

    if len(filtered) >= 2:
        b_tag, c_tag = filtered[0], filtered[1]
        # Simplified - don't rely on POS tagging
        # Just use the order of tokens
    elif len(filtered) == 1:
        b_tag = filtered[0]
        c_tag = ""
    else:
        b_tag = ""
        c_tag = ""
        
    return a_tag, b_tag, c_tag

def classify_keyword_three(keyword, seed, omitted_list, user_a_tags):
    """
    Process a keyword string:
      1) Remove the seed (if provided) and any omitted phrases.
      2) Use KeyBERT to extract the top candidate keyphrase (n-gram range: 1-4).
      3) Normalize and canonicalize the candidate (preserving word order).
      4) Split the candidate into tokens and assign A, B, and C tags via pick_tags_pos_based.
         If no candidate is found, return ("general-other", "", "").
    """
    _, kw_model = get_models()  # Get models when needed
    
    text = keyword.lower()
    if seed:
        pat = rf'\b{re.escape(seed.lower())}\b'
        text = re.sub(pat, '', text)
    for omit in omitted_list:
        pat = rf'\b{re.escape(omit)}\b'
        text = re.sub(pat, '', text)
    text = text.strip()

    keyphrases = kw_model.extract_keywords(text, keyphrase_ngram_range=(1,4), stop_words='english', top_n=1)
    if not keyphrases:
        return ("general-other", "", "")
    candidate = keyphrases[0][0].lower()
    norm_candidate = normalize_phrase(candidate)
    canon = canonicalize_phrase(norm_candidate)
    if not canon:
        return ("general-other", "", "")
    tokens = [t for t in canon.split() if t.strip() != ""]
    return pick_tags_pos_based(tokens, user_a_tags)

def extract_candidate_themes(keywords_list, top_n, progress_bar=None):
    """
    For each keyword, extract up to top_n candidate keyphrases using KeyBERT.
    If a progress_bar object is provided, update it during processing.
    Returns a list of candidate phrases.
    """
    _, kw_model = get_models()  # Get models when needed
    
    all_phrases = []
    total = len(keywords_list)
    for i, kw in enumerate(keywords_list):
        kps = kw_model.extract_keywords(kw, keyphrase_ngram_range=(1,4), stop_words='english', top_n=top_n)
        for kp in kps:
            if kp[0]:
                all_phrases.append(kp[0].lower())
        if progress_bar is not None:
            progress_bar.progress((i+1)/total)
    return all_phrases

def group_candidate_themes(all_phrases, min_freq):
    """
    Group candidate phrases by their canonical form. For each group that meets the minimum frequency,
    select the most common normalized form as the representative.
    Returns a dictionary mapping the representative candidate phrase to its frequency.
    """
    grouped = {}
    for phr in all_phrases:
        norm = normalize_phrase(phr)
        canon = canonicalize_phrase(norm)
        if canon:
            grouped.setdefault(canon, []).append(norm)
    candidate_map = {}
    for canon, arr in grouped.items():
        freq = len(arr)
        if freq >= min_freq:
            rep = Counter(arr).most_common(1)[0][0]
            candidate_map[rep] = freq
    return candidate_map

def realign_tags_based_on_frequency(df, col_name="B:Tag", other_col="C:Tag"):
    """
    Post-processing re-alignment:
      1) For each token in the specified columns, compute overall frequency.
      2) For each row, reassign tokens based on frequency.
      3) Ensure that each cell gets only one token (by taking the first token after re-assignment).
    """
    freq_in_col = Counter()
    freq_in_other = Counter()
    
    for i, row in df.iterrows():
        bval = row[col_name]
        oval = row[other_col]
        if bval:
            for token in bval.split():
                freq_in_col[token] += 1
        if oval:
            for token in oval.split():
                freq_in_other[token] += 1

    unify_map = {}
    all_tokens = set(freq_in_col.keys()) | set(freq_in_other.keys())
    for tok in all_tokens:
        c_freq = freq_in_col[tok]
        o_freq = freq_in_other[tok]
        if o_freq > c_freq:
            unify_map[tok] = other_col
        else:
            unify_map[tok] = col_name

    new_b_col, new_o_col = [], []
    for i, row in df.iterrows():
        b_tokens = row[col_name].split() if row[col_name] else []
        o_tokens = row[other_col].split() if row[other_col] else []
        combined = [(t, "b") for t in b_tokens] + [(t, "o") for t in o_tokens]
        new_b_list = []
        new_o_list = []
        for (t, orig) in combined:
            if unify_map.get(t, col_name) == col_name:
                new_b_list.append(t)
            else:
                new_o_list.append(t)
        new_b_col.append(new_b_list[0] if new_b_list else "")
        new_o_col.append(new_o_list[0] if new_o_list else "")
    df[col_name] = new_b_col
    df[other_col] = new_o_col
    return df

# Enhanced semantic intent-based clustering - UPDATED for text-embedding-3-small
def two_stage_clustering(df, cluster_method="Tag-based", embedding_model=None, api_key=None,
                        use_openai_embeddings=False, min_shared_keywords=3, similarity_threshold=0.65):
    """
    Perform two-stage clustering with semantic intent-based approach:
    1. Group by A tag
    2. Within each A tag group, cluster by semantic intent with minimum shared keywords
    
    Parameters:
    -----------
    df : DataFrame
        DataFrame containing Keywords, A:Tag, B:Tag, C:Tag columns
    cluster_method : str
        Method for clustering within A tag groups: "Tag-based", "Semantic", "Hybrid"
    embedding_model : SentenceTransformer model
        Model for generating embeddings (required for semantic and hybrid methods)
    api_key : str
        OpenAI API key (required if use_openai_embeddings is True)
    use_openai_embeddings : bool
        Whether to use OpenAI's text-embedding-3-small embeddings instead of SentenceTransformer
    min_shared_keywords : int
        Minimum number of keywords required to form a cluster
    similarity_threshold : float
        Similarity threshold for clustering (higher means more similar)
        
    Returns:
    --------
    clustered_df : DataFrame
        Original DataFrame with additional columns including confidence score
    cluster_info : dict
        Dictionary with information about each cluster
    """
    # Step 1: Group by A tag
    a_tag_groups = df.groupby("A:Tag")
    
    # Prepare the clustered dataframe
    clustered_df = pd.DataFrame()
    
    # Store cluster information
    cluster_info = {}
    global_cluster_id = 0
    
    # Process each A tag group
    for a_tag, group_df in a_tag_groups:
        group_size = len(group_df)
        
        # Skip empty groups
        if group_size == 0:
            continue
            
        # Create a copy of the group with index reset
        group_df = group_df.copy().reset_index(drop=True)
        
        # Add A_Group column
        group_df["A_Group"] = a_tag
        
        # For confidence scoring
        group_df["Cluster_Confidence"] = 1.0  # Default confidence
        group_df["Is_Outlier"] = False  # Default not an outlier
        
        # Handle clustering within this A tag group
        if group_size <= 1:
            # Not enough samples to cluster meaningfully
            group_df["Subcluster"] = 0
        else:
            # Enough samples to cluster - generate features based on selected method
            if cluster_method == "Tag-based":
                # Use only B and C tags for clustering within A tag groups
                b_dummies = pd.get_dummies(group_df["B:Tag"].fillna(""), prefix="B")
                c_dummies = pd.get_dummies(group_df["C:Tag"].fillna(""), prefix="C")
                
                # Handle empty dummies case
                if b_dummies.empty and c_dummies.empty:
                    group_df["Subcluster"] = 0
                elif b_dummies.empty:
                    features = c_dummies
                elif c_dummies.empty:
                    features = b_dummies
                else:
                    features = pd.concat([b_dummies, c_dummies], axis=1)
                    
                # Apply clustering if we have features
                if "Subcluster" not in group_df.columns and features.shape[1] > 0:
                    # Apply semantic intent-based clustering with improved function
                    cluster_labels, confidence_scores = semantic_intent_clustering(
                        group_df["Keywords"].tolist(), 
                        features.values, 
                        min_shared_keywords=min_shared_keywords,
                        similarity_threshold=similarity_threshold
                    )
                    
                    # Assign subclusters and confidence scores
                    group_df["Subcluster"] = cluster_labels
                    group_df["Cluster_Confidence"] = confidence_scores
                    group_df["Is_Outlier"] = (cluster_labels == -1)
                else:
                    group_df["Subcluster"] = 0
                    
            elif cluster_method == "Semantic":
                # Use keyword embeddings
                keywords = group_df["Keywords"].tolist()
                
                if not keywords or (not embedding_model and not use_openai_embeddings):
                    group_df["Subcluster"] = 0
                else:
                    # Generate embeddings based on selected method
                    if use_openai_embeddings and api_key:
                        # Use enriched OpenAI embeddings for better results
                        a_tags = group_df["A:Tag"].fillna("").tolist()
                        b_tags = group_df["B:Tag"].fillna("").tolist()
                        c_tags = group_df["C:Tag"].fillna("").tolist()
                        
                        # Get enriched embeddings that include tag context
                        embeddings = get_cached_enriched_embeddings(
                            keywords, 
                            {'A': a_tags, 'B': b_tags, 'C': c_tags}, 
                            api_key
                        )
                    else:
                        # Use SentenceTransformer embeddings
                        embeddings = embedding_model.encode(keywords)
                    
                    # Apply improved semantic intent-based clustering
                    cluster_labels, confidence_scores = semantic_intent_clustering(
                        keywords, 
                        embeddings,
                        min_shared_keywords=min_shared_keywords,
                        similarity_threshold=similarity_threshold
                    )
                    
                    # Assign subclusters and confidence scores
                    group_df["Subcluster"] = cluster_labels
                    group_df["Cluster_Confidence"] = confidence_scores
                    group_df["Is_Outlier"] = (cluster_labels == -1)
                    
            else:  # Hybrid method
                # Combine keywords with B and C tags
                combined_texts = group_df.apply(
                    lambda row: f"{row['Keywords']} {row['B:Tag']} {row['C:Tag']}",
                    axis=1
                ).tolist()
                
                if not combined_texts or (not embedding_model and not use_openai_embeddings):
                    group_df["Subcluster"] = 0
                else:
                    # Generate embeddings based on selected method
                    if use_openai_embeddings and api_key:
                        # Use enriched OpenAI embeddings
                        a_tags = group_df["A:Tag"].fillna("").tolist()
                        embeddings = get_cached_enriched_embeddings(
                            combined_texts, 
                            {'A': a_tags}, 
                            api_key
                        )
                    else:
                        # Use SentenceTransformer embeddings
                        embeddings = embedding_model.encode(combined_texts)
                    
                    # Apply improved semantic intent-based clustering
                    cluster_labels, confidence_scores = semantic_intent_clustering(
                        combined_texts, 
                        embeddings,
                        min_shared_keywords=min_shared_keywords,
                        similarity_threshold=similarity_threshold
                    )
                    
                    # Assign subclusters and confidence scores
                    group_df["Subcluster"] = cluster_labels
                    group_df["Cluster_Confidence"] = confidence_scores
                    group_df["Is_Outlier"] = (cluster_labels == -1)
        
        # Create a unique global cluster ID for each subcluster
        subcluster_map = {}
        subclusters = group_df["Subcluster"].unique()
        
        # Create an outlier cluster for this A tag
        outlier_cluster_id = global_cluster_id + len([s for s in subclusters if s != -1])
        
        # Process regular subclusters
        for subcluster in subclusters:
            # Skip -1 (outliers) for now
            if subcluster == -1:
                continue
                
            subcluster_map[subcluster] = global_cluster_id
            
            # Get the subcluster dataframe excluding outliers
            subcluster_df = group_df[group_df["Subcluster"] == subcluster]
            
            # Get the most common B and C tags in this cluster
            top_b_tags = subcluster_df["B:Tag"].value_counts().head(3).index.tolist()
            top_c_tags = subcluster_df["C:Tag"].value_counts().head(3).index.tolist()
            
            # Get keywords in this cluster
            keywords_in_cluster = subcluster_df["Keywords"].tolist()
            
            # Only create the cluster if it has keywords
            if len(keywords_in_cluster) > 0:
                cluster_info[global_cluster_id] = {
                    "a_tag": a_tag,
                    "size": len(subcluster_df),
                    "b_tags": top_b_tags,
                    "c_tags": top_c_tags,
                    "keywords": keywords_in_cluster,
                    "is_outlier_cluster": False
                }
                global_cluster_id += 1
        
        # Create outlier cluster for this A tag if needed
        outlier_df = group_df[group_df["Subcluster"] == -1]
        if len(outlier_df) > 0:
            outlier_b_tags = outlier_df["B:Tag"].value_counts().head(3).index.tolist()
            outlier_c_tags = outlier_df["C:Tag"].value_counts().head(3).index.tolist()
            outlier_keywords = outlier_df["Keywords"].tolist()
            
            cluster_info[outlier_cluster_id] = {
                "a_tag": a_tag,
                "size": len(outlier_df),
                "b_tags": outlier_b_tags,
                "c_tags": outlier_c_tags,
                "keywords": outlier_keywords,
                "is_outlier_cluster": True
            }
            
            # Map outliers to outlier cluster
            subcluster_map[-1] = outlier_cluster_id
        
        # Map subclusters to global cluster IDs
        group_df["Cluster"] = group_df["Subcluster"].map(subcluster_map)
        
        # Add to the result dataframe
        clustered_df = pd.concat([clustered_df, group_df], ignore_index=True)
    
    return clustered_df, cluster_info

# NEW: Batch processing for cluster descriptors to improve performance
def batch_generate_cluster_descriptors(cluster_info, api_key, max_clusters_per_batch=5):
    """Generate descriptive labels for clusters in batches for better performance"""
    openai.api_key = api_key
    all_descriptors = {}
    cluster_batches = []
    
    # Create batches of clusters
    batch = []
    total_keywords = 0
    
    # First sort clusters by size (largest first)
    sorted_clusters = sorted(
        [(k, v) for k, v in cluster_info.items()], 
        key=lambda x: x[1]["size"], 
        reverse=True
    )
    
    for cluster_id, info in sorted_clusters:
        # Skip if this would make batch too large
        if len(batch) >= max_clusters_per_batch or total_keywords > 50:
            if batch:
                cluster_batches.append(batch)
            batch = []
            total_keywords = 0
            
        # Add to current batch
        sample_kws = info["keywords"][:min(10, len(info["keywords"]))]
        is_outlier = info.get("is_outlier_cluster", False)
        
        # Skip batching for outliers - handle them separately
        if is_outlier:
            all_descriptors[cluster_id] = f"{info['a_tag'].title()}-Miscellaneous"
            continue
            
        batch.append({
            "id": cluster_id,
            "a_tag": info["a_tag"],
            "b_tags": info["b_tags"][:3] if info["b_tags"] else [],
            "keywords": sample_kws
        })
        total_keywords += len(sample_kws)
    
    # Add final batch if not empty
    if batch:
        cluster_batches.append(batch)
    
    # Process each batch
    for i, batch in enumerate(cluster_batches):
        prompt = "Generate brief, specific 2-4 word descriptors for each keyword cluster:\n\n"
        
        # Add each cluster to the prompt
        for cluster in batch:
            prompt += f"CLUSTER {cluster['id']}:\n"
            prompt += f"Category: {cluster['a_tag']}\n"
            prompt += f"Attributes: {', '.join(cluster['b_tags'])}\n"
            prompt += f"Keywords: {', '.join(cluster['keywords'])}\n\n"
        
        prompt += """For each cluster, provide ONLY a short 2-4 word descriptor that captures its essence.
        Format your response as JSON:
        {
          "descriptors": [
            {"cluster_id": 1, "descriptor": "Vinyl Window Installation"},
            {"cluster_id": 2, "descriptor": "Door Hardware Options"}
          ]
        }
        
        Keep each descriptor specific, concise (2-4 words), and clear.
        """
        
        try:
            response = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.4,
                response_format={"type": "json_object"},
                max_tokens=1000
            )
            
            # Parse results
            try:
                result_json = json.loads(response.choices[0].message.content)
                for item in result_json.get("descriptors", []):
                    cluster_id = item.get("cluster_id")
                    descriptor = item.get("descriptor", "")
                    
                    if cluster_id is not None and descriptor:
                        # Validate descriptor isn't too long
                        if len(descriptor.split()) <= 5:
                            all_descriptors[cluster_id] = descriptor
                        else:
                            # Fall back to basic descriptor
                            info = cluster_info[cluster_id]
                            all_descriptors[cluster_id] = f"{info['a_tag'].title()}-{info['b_tags'][0].title() if info['b_tags'] else 'General'}"
            except:
                # If JSON parsing fails, use basic descriptors for this batch
                for cluster in batch:
                    cluster_id = cluster["id"]
                    a_tag = cluster["a_tag"]
                    b_tags = cluster["b_tags"]
                    all_descriptors[cluster_id] = f"{a_tag.title()}-{b_tags[0].title() if b_tags else 'General'}"
        except Exception as e:
            # If API call fails, use basic descriptors
            for cluster in batch:
                cluster_id = cluster["id"]
                a_tag = cluster["a_tag"]
                b_tags = cluster["b_tags"]
                all_descriptors[cluster_id] = f"{a_tag.title()}-{b_tags[0].title() if b_tags else 'General'}"
    
    # Return the descriptors for all clusters
    return all_descriptors

# Enhanced function to generate descriptive cluster labels
def generate_cluster_descriptors(cluster_info, use_gpt=False, api_key=None):
    """
    Generate descriptive labels for each cluster, with special handling for outlier clusters.
    
    Parameters:
    -----------
    cluster_info : dict
        Dictionary with information about each cluster
    use_gpt : bool
        Whether to use GPT for more sophisticated descriptions
    api_key : str
        OpenAI API key if use_gpt is True
        
    Returns:
    --------
    descriptor_map : dict
        Dictionary mapping cluster IDs to descriptive labels
    """
    # For large datasets, use the batched version
    if use_gpt and api_key and len(cluster_info) > 10:
        return batch_generate_cluster_descriptors(cluster_info, api_key)
    
    descriptor_map = {}
    
    for cluster_id, info in cluster_info.items():
        a_tag = info["a_tag"]
        b_tags = info["b_tags"]
        is_outlier = info.get("is_outlier_cluster", False)
        
        # For outlier clusters, create a special label
        if is_outlier:
            descriptor_map[cluster_id] = f"{a_tag.title()}-Miscellaneous"
            continue
        
        if use_gpt and api_key:
            # Use GPT for more sophisticated labeling
            try:
                openai.api_key = api_key
                sample_keywords = info["keywords"][:min(10, len(info["keywords"]))]
                
                prompt = f"""Generate a brief, specific 2-4 word descriptor for a keyword cluster with these properties:
                - Primary category: {a_tag}
                - Secondary attributes: {', '.join(b_tags[:3]) if b_tags else 'none'}
                - Sample keywords: {', '.join(sample_keywords)}
                
                Your descriptor should be specific, concise (2-4 words), and clear.
                Format: Just return the descriptor with no explanation or punctuation.
                Example good responses: "Vinyl Window Installation", "Interior Door Hardware", "Energy Efficient Windows"
                """
                
                response = openai.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.4,
                    max_tokens=10
                )
                
                descriptor = response.choices[0].message.content.strip().rstrip('.').rstrip(',')
                # Fall back to simple descriptor if GPT returns something too long or empty
                if len(descriptor.split()) > 5 or not descriptor:
                    raise Exception("Invalid descriptor")
                    
                descriptor_map[cluster_id] = descriptor
                
            except Exception as e:
                # Fall back to simple descriptor on error
                if b_tags:
                    descriptor_map[cluster_id] = f"{a_tag.title()}-{b_tags[0].title()}"
                else:
                    descriptor_map[cluster_id] = f"{a_tag.title()}-General"
        else:
            # Generate simple tag-based descriptor
            if b_tags:
                descriptor_map[cluster_id] = f"{a_tag.title()}-{b_tags[0].title()}"
            else:
                descriptor_map[cluster_id] = f"{a_tag.title()}-General"
    
    return descriptor_map

# Create enhanced visualization with outlier highlighting
def create_two_stage_visualization(df_clustered, cluster_info, cluster_descriptors=None):
    """Create a visualization showing the distribution of clusters within A tag groups."""
    # Count keywords per A tag
    a_tag_counts = df_clustered.groupby("A_Group").size()
    
    # Get top A tags for visualization (limit to top 10 for readability)
    top_a_tags = a_tag_counts.sort_values(ascending=False).head(10).index.tolist()
    
    # Prepare data for visualization
    a_tags = []
    cluster_ids = []
    counts = []
    is_outlier = []
    
    for a_tag in top_a_tags:
        # Get clusters for this A tag
        a_clusters = [k for k, v in cluster_info.items() if v["a_tag"] == a_tag]
        
        for cluster_id in a_clusters:
            a_tags.append(a_tag)
            
            # Use descriptive label if available
            if cluster_descriptors and cluster_id in cluster_descriptors:
                cluster_ids.append(cluster_descriptors[cluster_id])
            else:
                cluster_ids.append(f"C{cluster_id}")
                
            counts.append(cluster_info[cluster_id]["size"])
            is_outlier.append(cluster_info[cluster_id].get("is_outlier_cluster", False))
    
    # Create a DataFrame for plotting
    plot_df = pd.DataFrame({
        "A_Tag": a_tags,
        "Cluster": cluster_ids,
        "Count": counts,
        "Is_Outlier": is_outlier
    })
    
    # Create plot - using a simpler approach to avoid hatching issues
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Group by A_Tag
    grouped = plot_df.pivot_table(
        index="A_Tag", columns="Cluster", values="Count", fill_value=0
    )
    
    # Plot the regular clusters (non-outliers)
    regular_clusters = [c for c in plot_df["Cluster"].unique() 
                      if not plot_df[plot_df["Cluster"] == c]["Is_Outlier"].any()]
    outlier_clusters = [c for c in plot_df["Cluster"].unique() 
                       if plot_df[plot_df["Cluster"] == c]["Is_Outlier"].any()]
    
    if regular_clusters:
        regular_data = grouped[regular_clusters]
        regular_data.plot(kind="bar", stacked=True, ax=ax, colormap="tab20", alpha=0.7)
    
    # Add a special bar for outlier clusters if they exist
    if outlier_clusters:
        # Get the count data for outlier clusters
        outlier_data = grouped[outlier_clusters].sum(axis=1)
        
        # Calculate the positions where these bars should start (on top of the regular bars)
        if regular_clusters:
            bottom = regular_data.sum(axis=1)
        else:
            bottom = pd.Series(0, index=grouped.index)
        
        # Add visual representation of outlier clusters
        outlier_bars = ax.bar(
            range(len(grouped)), 
            outlier_data,
            bottom=bottom,
            color='lightgrey',
            alpha=0.7,
            label="Miscellaneous Keywords"
        )
    
    ax.set_title("Keyword Distribution by A:Tag and Cluster")
    ax.set_xlabel("A:Tag")
    ax.set_ylabel("Number of Keywords")
    
    # Add a note about miscellaneous keywords
    if outlier_clusters:
        ax.text(
            0.5, 0.95, 
            "Light grey sections represent miscellaneous keywords that didn't fit well in other clusters",
            transform=ax.transAxes,
            ha='center',
            fontsize=9,
            bbox=dict(facecolor='white', alpha=0.7)
        )
    
    # Customize legend
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, title="Clusters", bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    
    return fig

# NEW: Batch process GPT topic generation function
def batch_generate_topics(cluster_info, api_key, max_clusters_per_batch=5):
    """Process multiple clusters in a single API call to improve performance"""
    openai.api_key = api_key
    all_results = {}
    cluster_batches = []
    
    # Create batches of clusters
    batch = []
    total_keywords = 0
    
    # First sort clusters by size (largest first)
    sorted_clusters = sorted(
        [(k, v) for k, v in cluster_info.items()], 
        key=lambda x: x[1]["size"], 
        reverse=True
    )
    
    for cluster_id, info in sorted_clusters:
        # Skip if this would make batch too large
        if len(batch) >= max_clusters_per_batch or total_keywords > 100:
            if batch:
                cluster_batches.append(batch)
            batch = []
            total_keywords = 0
            
        # Add to current batch
        sample_kws = info["keywords"][:min(15, len(info["keywords"]))]
        batch.append({
            "id": cluster_id,
            "a_tag": info["a_tag"],
            "b_tags": info["b_tags"][:3] if info["b_tags"] else [],
            "keywords": sample_kws,
            "is_outlier": info.get("is_outlier_cluster", False)
        })
        total_keywords += len(sample_kws)
    
    # Add final batch if not empty
    if batch:
        cluster_batches.append(batch)
    
    # Process each batch
    with st.text(f"Processing {len(cluster_batches)} batches of clusters..."):
        progress_bar = st.progress(0)
        
        for i, batch in enumerate(cluster_batches):
            prompt = "Generate content topics for multiple keyword clusters:\n\n"
            
            # Add each cluster to the prompt
            for cluster in batch:
                prompt += f"CLUSTER {cluster['id']}:\n"
                prompt += f"Category: {cluster['a_tag']}\n"
                prompt += f"Attributes: {', '.join(cluster['b_tags'])}\n"
                prompt += f"Keywords: {', '.join(cluster['keywords'])}\n"
                prompt += f"Is Miscellaneous: {'Yes' if cluster['is_outlier'] else 'No'}\n\n"
            
            prompt += """For each cluster, generate 3 content topic ideas in JSON format:
            {
              "results": [
                {
                  "cluster_id": 1,
                  "topics": [
                    {"title": "Topic 1 Title", "format": "Format 1", "value": "Brief value proposition"},
                    {"title": "Topic 2 Title", "format": "Format 2", "value": "Brief value proposition"},
                    {"title": "Topic 3 Title", "format": "Format 3", "value": "Brief value proposition"}
                  ]
                },
                {
                  "cluster_id": 2,
                  "topics": [...]
                }
              ]
            }
            """
            
            try:
                response = openai.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7,
                    response_format={"type": "json_object"},
                    max_tokens=2000
                )
                
                # Parse results
                try:
                    result_json = json.loads(response.choices[0].message.content)
                    for cluster_result in result_json.get("results", []):
                        cluster_id = cluster_result.get("cluster_id")
                        topics = cluster_result.get("topics", [])
                        
                        if cluster_id is not None and topics:
                            # Extract topics for this cluster
                            topic_ideas = []
                            expanded_ideas = []
                            value_props = []
                            
                            for topic in topics:
                                title = topic.get("title", "")
                                format_type = topic.get("format", "")
                                value = topic.get("value", "")
                                
                                if title:
                                    topic_ideas.append(title)
                                    if format_type:
                                        expanded_ideas.append(f"{title} ({format_type})")
                                    else:
                                        expanded_ideas.append(title)
                                    value_props.append(value)
                            
                            # Store results
                            all_results[cluster_id] = {
                                "topic_ideas": topic_ideas,
                                "expanded_ideas": expanded_ideas,
                                "value_props": value_props
                            }
                except Exception as e:
                    # If JSON parsing fails, use basic topics for this batch
                    for cluster in batch:
                        cluster_id = cluster["id"]
                        a_tag = cluster["a_tag"]
                        b_tags = cluster["b_tags"]
                        
                        topic_base = f"{a_tag.title()}"
                        if b_tags:
                            topic_base += f" {b_tags[0].title()}"
                            
                        all_results[cluster_id] = {
                            "topic_ideas": [
                                f"Complete Guide to {topic_base}", 
                                f"{topic_base} Comparison", 
                                f"How to Choose {topic_base}"
                            ],
                            "expanded_ideas": [
                                f"Complete Guide to {topic_base} (Guide)", 
                                f"{topic_base} Comparison (Comparison)", 
                                f"How to Choose {topic_base} (Guide)"
                            ],
                            "value_props": [
                                "Provides comprehensive information on this topic",
                                "Helps readers compare options",
                                "Guides decision-making process"
                            ]
                        }
            except Exception as e:
                # If API call fails, use basic topics
                for cluster in batch:
                    cluster_id = cluster["id"]
                    a_tag = cluster["a_tag"]
                    b_tags = cluster["b_tags"]
                    
                    topic_base = f"{a_tag.title()}"
                    if b_tags:
                        topic_base += f" {b_tags[0].title()}"
                        
                    all_results[cluster_id] = {
                        "topic_ideas": [
                            f"Guide to {topic_base}", 
                            f"{topic_base} Overview", 
                            f"{topic_base} Tips"
                        ],
                        "expanded_ideas": [
                            f"Guide to {topic_base} (Guide)", 
                            f"{topic_base} Overview (Overview)", 
                            f"{topic_base} Tips (Tips)"
                        ],
                        "value_props": [
                            "Provides comprehensive information on this topic",
                            "Gives an overview of key concepts",
                            "Offers practical advice"
                        ]
                    }
            
            # Update progress
            progress_bar.progress((i + 1) / len(cluster_batches))
        
        progress_bar.empty()
    return all_results

# Helper Functions for GPT Integration
def generate_gpt_topics(keywords, a_tags, b_tags, api_key, frequency_data=None):
    """Generate content topics using GPT-4o-mini based on cluster keywords and tags."""
    openai.api_key = api_key
    
    # Prepare a sample of keywords (limited to avoid token limits)
    keyword_sample = keywords[:min(30, len(keywords))]
    
    # Add frequency data if available
    frequency_context = ""
    if frequency_data:
        freq_info = []
        for kw, count in frequency_data.items():
            if kw in keyword_sample:
                freq_info.append(f"{kw} ({count})")
        if freq_info:
            frequency_context = "\nHIGH FREQUENCY KEYWORDS:\n" + ", ".join(freq_info[:10])
    
    # Create prompt with context and instructions
    prompt = f"""You are a content strategist helping to generate content topic ideas based on keyword clusters.

KEYWORD CLUSTER:
{', '.join(keyword_sample)}{frequency_context}

PRIMARY CATEGORY TAGS: {', '.join(a_tags)}
SECONDARY ATTRIBUTE TAGS: {', '.join(b_tags)}

Based on these related keywords and their tags, please generate:
1. Five specific, engaging content topic ideas that would cover the themes in this keyword cluster
2. For each topic, suggest a content format (guide, comparison, how-to, checklist, FAQ, etc.)
3. Briefly explain why this topic would be valuable (1-2 sentences)

The topics should be specific enough to be actionable but broad enough to cover multiple keywords.
Format your response as JSON with this structure:
{{
  "topics": [
    {{
      "title": "How to Choose Energy-Efficient Windows for Coastal Homes",
      "format": "Buyer's Guide",
      "value": "Helps homeowners in coastal areas select windows that withstand harsh conditions while saving energy costs."
    }},
    ...4 more topics
  ]
}}
"""

    # Call the OpenAI API
    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            response_format={"type": "json_object"},
            max_tokens=1000
        )
        
        # Parse response
        content = response.choices[0].message.content
        result = json.loads(content)
        
        # Extract topics and formats
        topic_ideas = []
        expanded_ideas = []
        value_props = []
        
        for item in result.get("topics", []):
            title = item.get("title", "")
            format_type = item.get("format", "")
            value = item.get("value", "")
            
            if title:
                topic_ideas.append(title)
                if format_type:
                    expanded_ideas.append(f"{title} ({format_type})")
                else:
                    expanded_ideas.append(title)
                value_props.append(value)
        
        return topic_ideas, expanded_ideas, value_props
    
    except Exception as e:
        # Fall back if the API call fails
        return (
            [f"Topic for {a_tags[0]} keywords"] if a_tags else ["General Topic"], 
            ["Content Guide"], 
            ["Provides valuable information on this topic"]
        )

def generate_basic_topics(keywords, a_tags, b_tags):
    """Generate basic content topics using KeyBERT when GPT is not available."""
    _, kw_model = get_models()  # Get models when needed
    
    combined_text = " ".join(keywords)
    key_phrases = kw_model.extract_keywords(combined_text, keyphrase_ngram_range=(1, 3), top_n=5)
    topic_phrases = [kp[0] for kp in key_phrases]
    
    # Generate topic ideas based on tags and phrases
    topic_ideas = []
    if a_tags and topic_phrases:
        primary_tag = a_tags[0]
        for phrase in topic_phrases[:3]:
            topic = f"{phrase.title()} {primary_tag.title()}"
            if b_tags:
                topic += f": {b_tags[0].title()}"
            topic_ideas.append(topic)
    elif topic_phrases:
        topic_ideas = [phrase.title() for phrase in topic_phrases[:3]]
    else:
        topic_ideas = ["General Topic"]
    
    # Add guide types
    content_types = ["Ultimate Guide", "Comparison", "How-to", "Buyer's Guide", "Troubleshooting"]
    expanded_ideas = []
    for idea in topic_ideas:
        for content_type in content_types[:2]:  # Just use first two content types
            expanded_ideas.append(f"{idea} ({content_type})")
    
    # Simple value propositions
    value_props = ["Provides comprehensive information on this topic" for _ in topic_ideas]
    
    return topic_ideas, expanded_ideas, value_props

def create_keyword_topic_mapping(df_filtered):
    """Create a mapping between keywords and their associated content topics"""
    # Create a subset of the DataFrame with just the essential columns
    keyword_topic_df = df_filtered[["Keywords", "Cluster", "Subcluster", "Cluster_Label", "Content_Topic", "Cluster_Confidence"]]
    
    # Sort by Cluster and then by Confidence (descending)
    keyword_topic_df = keyword_topic_df.sort_values(
        ["Cluster", "Cluster_Confidence"], 
        ascending=[True, False]
    )
    
    # Keep only necessary columns for the final output
    keyword_topic_df = keyword_topic_df[["Keywords", "Cluster", "Cluster_Label", "Content_Topic"]]
    
    # Rename columns for clarity
    keyword_topic_df = keyword_topic_df.rename(columns={
        "Keywords": "Keyword",
        "Cluster": "Cluster ID", 
        "Cluster_Label": "Cluster Name",
        "Content_Topic": "Content Topic"
    })
    
    return keyword_topic_df

# UPDATED: Cost estimation for text-embedding-3-small
def estimate_gpt4o_mini_costs(num_clusters, use_gpt_descriptors=True, use_gpt_topics=True):
    """
    Estimate costs for using GPT-4o-mini and text-embedding-3-small for cluster analysis
    
    Parameters:
    -----------
    num_clusters : int
        Number of clusters to analyze
    use_gpt_descriptors : bool
        Whether GPT will be used for cluster naming/descriptions
    use_gpt_topics : bool
        Whether GPT will be used for topic generation
    
    Returns:
    --------
    total_cost : float
        Estimated cost in USD
    breakdown : dict
        Breakdown of costs by component
    """
    # GPT-4o-mini pricing (per 1K tokens)
    INPUT_PRICE = 0.00015  # $0.00015 per 1K input tokens
    OUTPUT_PRICE = 0.0006  # $0.0006 per 1K output tokens
    
    # text-embedding-3-small pricing (per 1K tokens)
    EMBEDDING_PRICE = 0.00002  # $0.00002 per 1K tokens
    
    # Initialize cost components
    descriptor_cost = 0
    topic_cost = 0
    analysis_cost = 0
    embedding_cost = 0
    
    # Estimate embedding costs (very rough approximation)
    avg_tokens_per_keyword = 8  # Average token count per keyword
    avg_keywords_per_cluster = 20  # Average cluster size
    total_keywords = num_clusters * avg_keywords_per_cluster
    embedding_tokens = total_keywords * avg_tokens_per_keyword
    embedding_cost = embedding_tokens * EMBEDDING_PRICE / 1000
    
    # Estimate for cluster descriptors (if enabled)
    if use_gpt_descriptors:
        # ~300 tokens input per cluster (prompt + sample keywords + tags)
        descriptor_input_tokens = num_clusters * 300
        # ~10 tokens output per cluster (short descriptive phrase)
        descriptor_output_tokens = num_clusters * 10
        
        descriptor_cost = (descriptor_input_tokens * INPUT_PRICE / 1000) + \
                         (descriptor_output_tokens * OUTPUT_PRICE / 1000)
    
    # Estimate for topic generation (if enabled)
    if use_gpt_topics:
        # ~600 tokens input per cluster (longer prompt with samples, tags, instructions)
        topic_input_tokens = num_clusters * 600
        # ~250 tokens output per cluster (5 topics with descriptions)
        topic_output_tokens = num_clusters * 250
        
        topic_cost = (topic_input_tokens * INPUT_PRICE / 1000) + \
                    (topic_output_tokens * OUTPUT_PRICE / 1000)
    
    # Add cost for overall content strategy (a single analysis)
    analysis_input_tokens = 600  # Strategy prompt with top topics
    analysis_output_tokens = 800  # Strategy recommendations
    analysis_cost = (analysis_input_tokens * INPUT_PRICE / 1000) + \
                   (analysis_output_tokens * OUTPUT_PRICE / 1000)
    
    # Calculate total cost
    total_cost = descriptor_cost + topic_cost + analysis_cost + embedding_cost
    
    # Return breakdown
    return total_cost, {
        "embeddings": embedding_cost,
        "cluster_descriptors": descriptor_cost,
        "topic_generation": topic_cost,
        "content_strategy": analysis_cost
    }

# Main UI
with st.sidebar:
    st.title("Keyword Analysis Tool")
    st.markdown("""
    This tool helps you analyze, tag, and generate content topics from keywords:
    
    1. **Extract Themes** - Find common patterns in your keywords
    2. **Tag Keywords** - Categorize keywords with A, B, C tags
    3. **Generate Topics** - Create content topics from tagged keywords
    """)
    
    # Add OpenAI API key input for GPT-powered features
    st.subheader("OpenAI API Settings")
    api_key = st.text_input("OpenAI API Key (for topic generation & embeddings)", type="password")
    use_gpt = st.checkbox("Use GPT-4o-mini for enhanced analysis", value=True)
    
    # Add info about new embedding model
    st.success("🚀 Now using text-embedding-3-small: Better clustering for all keyword types!")
    
    if use_gpt and not api_key:
        st.warning("⚠️ API key required for GPT features")
    
    # Mode selection
    st.subheader("Select Mode")
    mode = st.radio(
        "Choose a mode:",
        ["Candidate Theme Extraction", "Full Tagging", "Content Topic Clustering"],
        help="Select what you want to do with your keywords"
    )

# Main content area
if mode == "Candidate Theme Extraction":
    st.title("🔍 Extract Keyword Themes")
    
    st.markdown("""
    This mode identifies common themes in your keywords and shows how they would be tagged.
    Useful for understanding what patterns exist in your keyword set.
    """)
    
    # Settings in columns for better use of space
    col1, col2 = st.columns(2)
    with col1:
        nm = st.number_input("Process first N keywords (0 for all)", min_value=0, value=0, key="theme_n")
        topn = st.number_input("Keyphrases per keyword", min_value=1, value=3, key="theme_topn")
    with col2:
        mfreq = st.number_input("Minimum frequency threshold", min_value=1, value=2, key="theme_mfreq")
        clust = st.number_input("Number of clusters (0 to skip)", min_value=0, value=0, key="theme_clust")
    
    # A-Tags input
    user_atags_str = st.text_input("Specify allowed A:Tags (comma-separated)", "door, window", key="theme_a_tags")
    user_a_tags = set(normalize_token(x.strip()) for x in user_atags_str.split(",") if x.strip())
    
    # Reset button to process new file if already processed
    if 'candidate_themes_processed' in st.session_state and st.session_state.candidate_themes_processed:
        if st.button("Process New File", key="reset_themes"):
            st.session_state.candidate_themes_processed = False
            st.experimental_rerun()
    
    # Only show file uploader if not already processed
    if 'candidate_themes_processed' not in st.session_state or not st.session_state.candidate_themes_processed:
        # File upload
        file = st.file_uploader("Upload your keyword file (CSV/Excel)", type=["csv", "xls", "xlsx"], 
                              key="theme_file")
        
        # Process when file is uploaded
        if file:
            try:
                if file.name.endswith(".csv"):
                    df = pd.read_csv(file)
                else:
                    df = pd.read_excel(file)
            except Exception as e:
                st.error(f"Error reading file: {e}")
                st.stop()
        
            if "Keywords" not in df.columns:
                st.error("The file must have a 'Keywords' column.")
            else:
                kw_list = df["Keywords"].tolist()
                if nm > 0:
                    kw_list = kw_list[:nm]
                
                with st.spinner("Extracting themes from keywords..."):
                    progress_bar = st.progress(0)
                    all_phrases = extract_candidate_themes(kw_list, topn, progress_bar=progress_bar)
                    progress_bar.empty()  # Clear the progress bar when done
            
                    c_map = group_candidate_themes(all_phrases, mfreq)
            
                    if c_map:
                        cdf = pd.DataFrame(list(c_map.items()), columns=["Candidate Theme", "Frequency"])
                        cdf = cdf.sort_values(by="Frequency", ascending=False)
                        splitted = []
                        for theme in cdf["Candidate Theme"]:
                            norm = normalize_phrase(theme)
                            canon = canonicalize_phrase(norm)
                            tokens = [t for t in canon.split() if t.strip() != ""]
                            a, b, c = pick_tags_pos_based(tokens, user_a_tags)
                            splitted.append((a, b, c))
                        cdf["A:Tag"], cdf["B:Tag"], cdf["C:Tag"] = zip(*splitted)
                        
                        # Store in session state
                        st.session_state.candidate_themes_processed = True
                        st.session_state.c_map = c_map
                        st.session_state.cdf = cdf
                    else:
                        st.warning("No candidate themes meet the frequency threshold.")
    
    # Display results from session state if already processed
    if 'candidate_themes_processed' in st.session_state and st.session_state.candidate_themes_processed:
        c_map = st.session_state.c_map
        cdf = st.session_state.cdf
        
        st.subheader("Candidate Themes")
        st.dataframe(cdf)

        # Download button with unique key
        st.download_button(
            "Download Candidate Themes CSV", 
            cdf.to_csv(index=False).encode('utf-8'), 
            "candidate_themes.csv", 
            "text/csv",
            key="download_themes"
        )
        
        # GPT-powered insights if enabled
        if use_gpt and api_key:
            st.subheader("💡 Theme Insights")
            if st.button("Generate Theme Insights with GPT", key="theme_insights_button"):
                with st.spinner("Analyzing themes with GPT..."):
                    # Prepare most common themes for analysis
                    top_themes = cdf.head(15)["Candidate Theme"].tolist()
                    top_a_tags = cdf["A:Tag"].value_counts().head(3).index.tolist()
                    
                    try:
                        openai.api_key = api_key
                        response = openai.chat.completions.create(
                            model="gpt-4o-mini",
                            messages=[{"role": "user", "content": f"""
                            You're analyzing keyword themes for content planning.
                            
                            TOP THEMES: {', '.join(top_themes)}
                            
                            MAIN CATEGORIES: {', '.join(top_a_tags)}
                            
                            Please provide:
                            1. A brief analysis of what these themes reveal about user search intent
                            2. Three content strategy recommendations based on these themes
                            
                            Make your insights specific and actionable.
                            """}],
                            temperature=0.5
                        )
                        
                        insights = response.choices[0].message.content
                        st.markdown(insights)
                    except Exception as e:
                        st.error(f"Error generating insights: {e}")

elif mode == "Full Tagging":
    st.title("🏷️ Tag Your Keywords")
    
    st.markdown("""
    This mode processes each keyword to assign it to categories using a three-tag system:
    - **A:Tag** - Primary category (e.g., window, door)
    - **B:Tag** - Secondary attribute or modifier
    - **C:Tag** - Additional attribute
    """)
    
    # Settings in a more compact layout
    col1, col2 = st.columns(2)
    
    with col1:
        seed = st.text_input("Seed Keyword to Remove (optional)", "", key="tag_seed")
        omit_str = st.text_input("Phrases to Omit (comma-separated)", "", key="tag_omit")
    
    with col2:
        user_atags_str = st.text_input("Allowed A:Tags (comma-separated)", "door, window", key="tag_a_tags")
        do_realign = st.checkbox("Enable post-processing re-alignment", value=True,
                               help="Ensures consistent tag placement based on frequency", key="tag_realign")
    
    # Reset button to process new file if already processed
    if 'full_tagging_processed' in st.session_state and st.session_state.full_tagging_processed:
        if st.button("Process New File", key="reset_tagging"):
            st.session_state.full_tagging_processed = False
            st.experimental_rerun()
    
    # Only show file uploader and related UI if not already processed
    if 'full_tagging_processed' not in st.session_state or not st.session_state.full_tagging_processed:
        # Optional: Upload an initial tagging rule CSV
        initial_rule_file = st.file_uploader("Initial Tagging Rule CSV (optional)", type=["csv"], key="tag_rule_file")
        use_initial_rule = st.checkbox("Use Initial Tagging Rule if available", value=False, key="use_tag_rule")
        
        file = st.file_uploader("Upload Keyword File (CSV/Excel)", type=["csv", "xls", "xlsx"], key="tag_file")
        
        # Build the initial rule mapping if provided and requested.
        initial_rule_mapping = {}
        if use_initial_rule and initial_rule_file is not None:
            try:
                rule_df = pd.read_csv(initial_rule_file)
                rule_df = rule_df.fillna('')  # Replace NaN with empty strings.
                # Expect columns: Candidate Theme, A:Tag, B:Tag, C:Tag.
                for index, row in rule_df.iterrows():
                    candidate = str(row["Candidate Theme"])
                    canon_candidate = canonicalize_phrase(normalize_phrase(candidate))
                    initial_rule_mapping[canon_candidate] = (str(row["A:Tag"]), str(row["B:Tag"]), str(row["C:Tag"]))
                
                st.success(f"Loaded {len(initial_rule_mapping)} tagging rules.")
            except Exception as e:
                st.error("Error reading initial tagging rule file: " + str(e))
        
        if file:
            try:
                if file.name.endswith(".csv"):
                    df = pd.read_csv(file)
                else:
                    df = pd.read_excel(file)
            except Exception as e:
                st.error(f"Error reading file: {e}")
                st.stop()
        
            if "Keywords" not in df.columns:
                st.error("The file must have a 'Keywords' column.")
            else:
                omitted_list = [x.strip().lower() for x in omit_str.split(",") if x.strip()]
                user_a_tags = set(normalize_token(x.strip()) for x in user_atags_str.split(",") if x.strip())
        
                with st.spinner("Processing keywords..."):
                    A_list, B_list, C_list = [], [], []
                    keywords = df["Keywords"].tolist()
                    
                    progress_bar = st.progress(0)
                    total = len(keywords)
                    for i, kw in enumerate(keywords):
                        # Canonicalize keyword for potential lookup.
                        canon_kw = canonicalize_phrase(normalize_phrase(kw))
                        if use_initial_rule and initial_rule_mapping and canon_kw in initial_rule_mapping:
                            a, b, c = initial_rule_mapping[canon_kw]
                        else:
                            a, b, c = classify_keyword_three(kw, seed, omitted_list, user_a_tags)
                        A_list.append(a)
                        B_list.append(b)
                        C_list.append(c)
                        progress_bar.progress((i+1)/total)
                    progress_bar.empty()
            
                    df["A:Tag"] = A_list
                    df["B:Tag"] = B_list
                    df["C:Tag"] = C_list
            
                    if do_realign:
                        with st.spinner("Re-aligning tags..."):
                            df = realign_tags_based_on_frequency(df, col_name="B:Tag", other_col="C:Tag")
            
                    # Summary Report: Only A:Tag & B:Tag combination
                    df["A+B"] = df["A:Tag"] + " - " + df["B:Tag"]
                    summary_ab = df.groupby("A+B").size().reset_index(name="Count")
                    summary_ab = summary_ab.sort_values("Count", ascending=False)
                    
                    # Store in session state
                    st.session_state.full_tagging_processed = True
                    st.session_state.tagged_df = df
                    st.session_state.summary_ab = summary_ab
    
    # Display results from session state if already processed
    if 'full_tagging_processed' in st.session_state and st.session_state.full_tagging_processed:
        df = st.session_state.tagged_df
        summary_ab = st.session_state.summary_ab
        
        st.subheader("Tagged Keywords")
        st.dataframe(df[["Keywords", "A:Tag", "B:Tag", "C:Tag"]])
    
        st.subheader("Tag Summary")
        st.dataframe(summary_ab)
    
        # Download buttons with unique keys
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                "Download Full Tagging CSV", 
                df.to_csv(index=False).encode('utf-8'), 
                "tagged_keywords.csv", 
                "text/csv",
                key="download_full_tags"
            )
        with col2:
            st.download_button(
                "Download Tag Summary CSV",
                summary_ab.to_csv(index=False).encode('utf-8'),
                "tag_summary.csv",
                "text/csv",
                key="download_tag_summary"
            )
        
        # GPT tag analysis if enabled
        if use_gpt and api_key:
            st.subheader("💡 Tag Analysis")
            if st.button("Analyze Tag Patterns with GPT", key="tag_analysis_button"):
                with st.spinner("Analyzing tags with GPT..."):
                    # Get top tag combinations for analysis
                    top_combos = summary_ab.head(10)
                    
                    try:
                        openai.api_key = api_key
                        response = openai.chat.completions.create(
                            model="gpt-4o-mini",
                            messages=[{"role": "user", "content": f"""
                            You're analyzing keyword tagging patterns for content planning.
                            
                            TOP TAG COMBINATIONS (A:Tag - B:Tag):
                            {top_combos.to_string(index=False)}
                            
                            Please provide:
                            1. A brief analysis of what these tag patterns reveal about the keyword set
                            2. Suggestions for how to group these tags into content topics
                            
                            Keep your analysis concise and actionable.
                            """}],
                            temperature=0.5
                        )
                        
                        analysis = response.choices[0].message.content
                        st.markdown(analysis)
                    except Exception as e:
                        st.error(f"Error generating analysis: {e}")
                        
        # Add the semantic intent-based clustering section
        st.subheader("🧩 AI-Powered Intent-Based Clustering")
        st.markdown("""
        Discover how your keywords naturally cluster based on semantic intent instead of forcing them into 
        an arbitrary number of clusters. Keywords with similar meaning will group together, with a minimum 
        of 5 common keywords required to form a meaningful cluster.
        """)

        st.info("""
        **Semantic Intent-Based Clustering:**
        - Keywords are first grouped by their A:Tag (primary category)
        - Within each A:Tag group, keywords naturally cluster based on semantic intent
        - A keyword must share intent with at least 5 other keywords to form a cluster
        - Keywords that don't share enough common meaning are placed in "Miscellaneous" groups
        """)

        # Add A-tag selection filter
        st.subheader("A-Tag Selection")
        # Get unique A-tags from the dataframe
        a_tags = sorted(df["A:Tag"].unique().tolist())
        selected_a_tags = st.multiselect(
            "Select A-Tags to analyze (leave empty for all)",
            options=a_tags,
            default=[a_tags[0]] if len(a_tags) > 0 else [],
            help="Select specific A-Tags to prevent pooling across categories"
        )
        
        # Create clustering options
        col1, col2, col3 = st.columns(3)

        with col1:
            # Use OpenAI embeddings option
            use_openai_embeddings = st.checkbox(
                "Use OpenAI embeddings for higher quality clustering", 
                value=True if api_key else False,
                help="Uses text-embedding-3-small for better semantic understanding (requires API key)"
            )
            
            if use_openai_embeddings and not api_key:
                st.warning("API key required for OpenAI embeddings")
                use_openai_embeddings = False
        
        with col2:
            # Controls for intent-based clustering - ADJUSTED DEFAULT VALUES
            min_shared_keywords = st.slider(
                "Min keywords per cluster", 
                min_value=2, 
                max_value=15, 
                value=3,  # Default 3
                help="Minimum number of keywords required to form a cluster"
            )
            
            # Auto-adjust option for similarity threshold
            use_auto_threshold = st.checkbox(
                "Auto-adjust similarity threshold",
                value=True,
                help="Automatically determine optimal threshold based on your keywords"
            )
            
            if not use_auto_threshold:
                similarity_threshold = st.slider(
                    "Similarity threshold", 
                    min_value=0.3,
                    max_value=0.95, 
                    value=0.65,
                    step=0.05,
                    help="How similar keywords must be to cluster together (higher = more similar)"
                )
            else:
                similarity_threshold = 0.65  # Default will be overridden

        # Add clustering method selection
        clustering_method = st.radio(
            "Clustering Approach",
            ["Tag-based", "Semantic", "Hybrid"],
            key="full_tagging_clustering_approach",
            help="Select how keywords should be clustered within each A:Tag group"
        )

        # Add embedding diagnostics feature
        with st.expander("🔍 Embedding Diagnostics"):
            if st.button("Analyze Embedding Quality"):
                with st.spinner("Analyzing embeddings..."):
                    if use_openai_embeddings and api_key:
                        # Sample keywords for analysis (limit to 200 for performance)
                        sample_kws = df["Keywords"].tolist()[:200]
                        
                        # Get sample A and B tags for context
                        sample_a_tags = df["A:Tag"].fillna("").tolist()[:200]
                        sample_b_tags = df["B:Tag"].fillna("").tolist()[:200]
                        
                        # Get enriched embeddings with text-embedding-3-small
                        sample_emb = get_cached_enriched_embeddings(
                            sample_kws, 
                            {'A': sample_a_tags, 'B': sample_b_tags}, 
                            api_key
                        )
                        
                        # Normalize embeddings
                        normalized_emb = sample_emb.copy()
                        for i in range(len(normalized_emb)):
                            norm = np.linalg.norm(normalized_emb[i])
                            if norm > 0:
                                normalized_emb[i] = normalized_emb[i] / norm
                        
                        # Calculate similarity matrix
                        sim_matrix = cosine_similarity(normalized_emb)
                        
                        # Get similarity statistics
                        flat_sim = sim_matrix[np.triu_indices(len(sim_matrix), k=1)]
                        
                        # Display statistics
                        st.write(f"**Similarity Statistics (sample of {len(sample_kws)} keywords):**")
                        st.write(f"Min: {flat_sim.min():.4f}, Max: {flat_sim.max():.4f}")
                        st.write(f"Mean: {flat_sim.mean():.4f}, Median: {np.median(flat_sim):.4f}")
                        
                        # Calculate suggested threshold
                        p50 = np.percentile(flat_sim, 50)
                        p75 = np.percentile(flat_sim, 75)
                        
                        # Dynamic recommendation based on distribution
                        if flat_sim.mean() > 0.7:  # Very similar embeddings
                            suggested = p50  # Use median for high-similarity sets
                        else:  
                            suggested = (p50 + p75) / 2  # Otherwise use between median and 75th percentile
                        
                        # Create histogram
                        fig, ax = plt.subplots(figsize=(10, 6))
                        ax.hist(flat_sim, bins=50)
                        ax.set_title("Embedding Similarity Distribution")
                        ax.set_xlabel("Cosine Similarity")
                        ax.set_ylabel("Count")
                        ax.axvline(x=suggested, color='r', linestyle='--', label=f"Suggested threshold ({suggested:.2f})")
                        ax.axvline(x=p50, color='g', linestyle=':', label=f"Median ({p50:.2f})")
                        ax.axvline(x=p75, color='b', linestyle=':', label=f"75th Percentile ({p75:.2f})")
                        ax.legend()
                        st.pyplot(fig)
                        
                        st.info(f"""
                        **Threshold Recommendations:**
                        - Suggested: {suggested:.2f} ✓ RECOMMENDED
                        - Conservative (fewer, larger clusters): {p75:.2f}
                        - Aggressive (more, smaller clusters): {p50:.2f}
                        
                        With text-embedding-3-small, you should get better clustering results than with previous models!
                        """)
                    else:
                        st.error("OpenAI API key required for embedding diagnostics")

        # Display cost information for OpenAI embeddings
        if use_openai_embeddings:
            num_keywords = len(df) if 'tagged_df' in st.session_state else 0
            est_tokens = num_keywords * 10  # Estimate 10 tokens per keyword on average
            estimated_cost = est_tokens * 0.00002 / 1000
            st.info(f"💰 **Estimated OpenAI embedding cost:** ${estimated_cost:.5f} for {num_keywords} keywords")

        # Display cost information for GPT usage
        if use_gpt and api_key:
            # Estimate number of clusters
            est_num_clusters = len(df["A:Tag"].unique()) * 3  # Rough estimate: ~3 clusters per A:Tag
            total_cost, cost_breakdown = estimate_gpt4o_mini_costs(est_num_clusters, use_gpt_descriptors=True)
    
            st.info(f"""
            💰 **Estimated GPT-4o-mini cost:** ${total_cost:.4f}
            - Embeddings: ${cost_breakdown['embeddings']:.4f} 
            - Cluster naming: ${cost_breakdown['cluster_descriptors']:.4f}
            - Content strategy: ${cost_breakdown['content_strategy']:.4f}
    
            *Note: Actual costs may vary depending on the number of clusters formed*
            """)

        if st.button("Generate Intent-Based Clusters", key="generate_intent_clusters"):
            # Clear memory before processing
            gc.collect()
            
            with st.spinner("Analyzing keyword patterns with semantic intent-based clustering..."):
                # Get models
                embedding_model, _ = get_models()
                
                # Filter by selected A-tags if any were chosen
                if selected_a_tags:
                    df_filtered = df[df["A:Tag"].isin(selected_a_tags)].copy()
                    if len(df_filtered) == 0:
                        st.error("No keywords found with the selected A-tags.")
                        st.stop()
                else:
                    df_filtered = df.copy()
                
                # Use the enhanced two-stage clustering with improved parameter values
                st.text("Processing clusters... this might take a few minutes for large datasets")
                df_clustered, cluster_info = two_stage_clustering(
                    df_filtered, 
                    cluster_method=clustering_method,  # Use the new variable name
                    embedding_model=embedding_model,
                    api_key=api_key,
                    use_openai_embeddings=use_openai_embeddings,
                    min_shared_keywords=min_shared_keywords,
                    similarity_threshold=similarity_threshold
                )
                
                # Generate descriptive labels for each cluster
                use_gpt_descriptors = use_gpt and api_key
                cluster_descriptors = generate_cluster_descriptors(cluster_info, use_gpt_descriptors, api_key)
                
                # Add cluster assignments only to the filtered dataframe (don't mix A-tags)
                df_filtered["Cluster"] = df_clustered["Cluster"]
                df_filtered["A_Group"] = df_clustered["A_Group"]
                df_filtered["Subcluster"] = df_clustered["Subcluster"]
                df_filtered["Cluster_Label"] = df_filtered["Cluster"].map(cluster_descriptors)
                df_filtered["Cluster_Confidence"] = df_clustered["Cluster_Confidence"]
                df_filtered["Is_Outlier"] = df_clustered["Is_Outlier"]
                
                # Use the filtered dataframe for all subsequent operations
                df = df_filtered  # Replace the original df with filtered df
                
                # Show overview of clusters
                st.subheader("Semantic Intent-Based Clustering Results")
                
                # Count clusters and outliers
                total_keywords = len(df_filtered)
                total_clusters = len([k for k, v in cluster_info.items() if not v.get("is_outlier_cluster", False)])
                total_outliers = len([k for k, v in cluster_info.items() if v.get("is_outlier_cluster", False)])
                
                # Fixed line - handle Is_Outlier column properly
                if "Is_Outlier" in df_filtered.columns:
                    df_filtered["Is_Outlier"] = df_filtered["Is_Outlier"].fillna(False)
                    outlier_keywords = df_filtered[df_filtered["Is_Outlier"]].shape[0]
                else:
                    outlier_keywords = 0
                    
                outlier_percent = (outlier_keywords / total_keywords) * 100 if total_keywords > 0 else 0
                
                # Show summary statistics
                st.markdown(f"""
                **Clustering Summary:**
                - Total Keywords: {total_keywords}
                - Semantic Clusters Formed: {total_clusters}
                - Miscellaneous Groups: {total_outliers}
                - Outlier Keywords: {outlier_keywords} ({outlier_percent:.1f}%)
                """)
                
                # Visualization of A tag groups and their clusters
                st.subheader("Keyword Cluster Distribution")
                fig = create_two_stage_visualization(df_clustered, cluster_info, cluster_descriptors)
                st.pyplot(fig)
                
                st.markdown("### Intent-Based Clusters by A:Tag")
                
                # Group clusters by A tag for organized display
                a_tags = sorted(df_filtered["A_Group"].unique())  # Changed from A_Group to A:Tag
                
                for a_tag in a_tags:
                    with st.expander(f"A:Tag: {a_tag}"):
                        # Get clusters for this A tag
                        a_clusters = [k for k, v in cluster_info.items() if v["a_tag"] == a_tag]
                        
                        if not a_clusters:
                            st.info(f"No clusters found for A:Tag '{a_tag}'")
                            continue
                            
                        for cluster_id in a_clusters:
                            info = cluster_info[cluster_id]
                            is_outlier = info.get("is_outlier_cluster", False)
                            
                            # Use descriptive label
                            cluster_label = cluster_descriptors[cluster_id] if cluster_id in cluster_descriptors else f"Cluster {cluster_id}"
                            
                            if is_outlier:
                                st.markdown(f"#### {cluster_label} ({info['size']} miscellaneous keywords)")
                                st.info("This is a miscellaneous group containing keywords that didn't share enough semantic intent with others.")
                            else:
                                st.markdown(f"#### {cluster_label} ({info['size']} keywords)")
                            
                            # Show B and C tags
                            if info["b_tags"]:
                                st.markdown(f"**B:Tags:** {', '.join(info['b_tags'])}")
                            if info["c_tags"]:
                                st.markdown(f"**C:Tags:** {', '.join(info['c_tags'])}")
                            
                            # Get cluster dataframe
                            cluster_df = df[df["Cluster"] == cluster_id]
                            
                            # Get high confidence keywords
                            high_conf_kws = cluster_df[cluster_df["Cluster_Confidence"] >= 0.7]
                            
                            # Show confidence stats
                            avg_confidence = cluster_df['Cluster_Confidence'].mean()
                            st.markdown(f"**Avg. Confidence:** {avg_confidence:.2f}")
                            st.markdown(f"**High confidence keywords:** {len(high_conf_kws)} of {len(cluster_df)} ({len(high_conf_kws)/len(cluster_df)*100:.1f}%)")
                            
                            # Show sample keywords
                            st.markdown("**Sample Keywords:**")
                            sample_df = cluster_df[["Keywords", "Cluster_Confidence"]].sort_values("Cluster_Confidence", ascending=False).head(10).reset_index(drop=True)
                            sample_df["Cluster_Confidence"] = sample_df["Cluster_Confidence"].apply(lambda x: f"{x:.2f}")
                            st.dataframe(sample_df, hide_index=False)
                            
                            st.markdown("---")
                
                # Download button for clustered keywords
                # Add additional columns to indicate quality of the clustering
                df["Cluster_Quality"] = df["Cluster_Confidence"].apply(
                    lambda x: "High" if x >= 0.8 else ("Medium" if x >= 0.6 else "Low")
                )
                
                clustered_csv = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download Intent-Based Clusters",
                    clustered_csv,
                    "semantic_intent_clusters.csv",
                    "text/csv",
                    key="download_intent_clusters"
                )

elif mode == "Content Topic Clustering":
    st.title("📚 Generate Content Topics")
    
    st.markdown("""
    This mode analyzes tagged keywords to:
    1. Group them using semantic intent-based clustering
    2. Generate content topic ideas for each cluster
    3. Provide insights on user intent and content opportunities
    """)
    
    # Reset button to process new file if already processed
    if 'content_topics_processed' in st.session_state and st.session_state.content_topics_processed:
        if st.button("Process New File", key="reset_topics"):
            st.session_state.content_topics_processed = False
            st.experimental_rerun()
    
    # Only show file uploader and settings if not already processed
    if 'content_topics_processed' not in st.session_state or not st.session_state.content_topics_processed:
        # File upload for tagged keywords
        file = st.file_uploader("Upload tagged keywords file (CSV)", type=["csv"],
                               help="This should be a file from the Full Tagging mode",
                               key="topics_file")
        
        if file:
            try:
                df = pd.read_csv(file)
            except Exception as e:
                st.error(f"Error reading file: {e}")
                st.stop()
            
            # Verify the file has required columns
            required_cols = ["Keywords", "A:Tag", "B:Tag", "C:Tag"]
            if not all(col in df.columns for col in required_cols):
                st.error("The file must contain columns: Keywords, A:Tag, B:Tag, and C:Tag.")
                st.stop()
            
            # Clustering parameters
            st.subheader("Semantic Intent-Based Clustering Settings")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Filter by A:Tag
                a_tags = sorted(df["A:Tag"].unique())
                selected_a_tags = st.multiselect("Filter by A:Tag (leave empty for all)", a_tags, key="a_tags_filter")
                
                # Use OpenAI embeddings option
                use_openai_embeddings = st.checkbox(
                    "Use OpenAI embeddings for higher quality clustering", 
                    value=True if api_key else False,
                    help="Uses text-embedding-3-small for better semantic understanding"
                )
                
                if use_openai_embeddings and not api_key:
                    st.warning("API key required for OpenAI embeddings")
                    use_openai_embeddings = False
                
                # Confidence threshold for topics
                topic_confidence_threshold = st.slider(
                    "Topic confidence threshold", 
                    min_value=0.0, 
                    max_value=1.0, 
                    value=0.6,
                    step=0.05, 
                    help="Only include keywords with confidence scores above this threshold in topic generation",
                    key="topic_confidence_threshold"
                )
            
            with col2:
                # Intent-based clustering parameters
                min_shared_keywords = st.slider(
                    "Min keywords per cluster", 
                    min_value=2, 
                    max_value=15, 
                    value=3,  # Default 3
                    help="Minimum number of keywords required to form a cluster"
                )
                
                # Auto-adjust option for similarity threshold
                use_auto_threshold = st.checkbox(
                    "Auto-adjust similarity threshold",
                    value=True,
                    help="Automatically determine optimal threshold based on your keywords"
                )
                
                if not use_auto_threshold:
                    similarity_threshold = st.slider(
                        "Similarity threshold", 
                        min_value=0.3,
                        max_value=0.95, 
                        value=0.65,
                        step=0.05,
                        help="How similar keywords must be to cluster together (higher = more similar)"
                    )
                else:
                    similarity_threshold = 0.65  # Default will be overridden
                
                # Clustering approach
                clustering_approach = st.radio(
                    "Clustering Approach",
                    ["Tag-based", "Semantic", "Hybrid"],
                    help="""
                    Tag-based: Cluster by B and C tags
                    Semantic: Cluster by keyword meaning
                    Hybrid: Combine tags and meaning
                    """,
                    key="topic_clustering_approach"
                )
            
            # Performance optimization options
            st.subheader("Performance Options for Large Datasets")
            use_fast_mode = st.checkbox(
                "Use fast mode for large datasets (recommended for >500 keywords)", 
                value=True if len(df) > 500 else False,
                help="Processes multiple clusters in batches and uses more efficient algorithms"
            )
            
            # GPT option
            if use_gpt:
                use_gpt_for_topics = st.checkbox("Use GPT for topic generation", value=True if api_key else False,
                                              help="Generate more creative topic ideas with GPT", key="use_gpt_topics")
                if use_gpt_for_topics and not api_key:
                    st.warning("API key required for GPT topic generation")

            # Add embedding diagnostics feature
            with st.expander("🔍 Embedding Diagnostics"):
                if st.button("Analyze Embedding Quality"):
                    with st.spinner("Analyzing embeddings..."):
                        if use_openai_embeddings and api_key:
                            # Sample keywords for analysis (limit for performance)
                            filtered_df = df
                            if selected_a_tags:
                                filtered_df = df[df["A:Tag"].isin(selected_a_tags)]
                            
                            sample_kws = filtered_df["Keywords"].tolist()[:200]
                            
                            # Get sample A and B tags for context
                            sample_a_tags = filtered_df["A:Tag"].fillna("").tolist()[:200]
                            sample_b_tags = filtered_df["B:Tag"].fillna("").tolist()[:200]
                            
                            # Get enriched embeddings with text-embedding-3-small
                            sample_emb = get_cached_enriched_embeddings(
                                sample_kws, 
                                {'A': sample_a_tags, 'B': sample_b_tags}, 
                                api_key
                            )
                            
                            # Normalize embeddings
                            normalized_emb = sample_emb.copy()
                            for i in range(len(normalized_emb)):
                                norm = np.linalg.norm(normalized_emb[i])
                                if norm > 0:
                                    normalized_emb[i] = normalized_emb[i] / norm
                            
                            # Calculate similarity matrix
                            sim_matrix = cosine_similarity(normalized_emb)
                            
                            # Get similarity statistics
                            flat_sim = sim_matrix[np.triu_indices(len(sim_matrix), k=1)]
                            
                            # Display statistics
                            st.write(f"**Similarity Statistics (sample of {len(sample_kws)} keywords):**")
                            st.write(f"Min: {flat_sim.min():.4f}, Max: {flat_sim.max():.4f}")
                            st.write(f"Mean: {flat_sim.mean():.4f}, Median: {np.median(flat_sim):.4f}")
                            
                            # Calculate suggested threshold
                            p50 = np.percentile(flat_sim, 50)
                            p75 = np.percentile(flat_sim, 75)
                            
                            # Dynamic recommendation based on distribution
                            if flat_sim.mean() > 0.7:  # Very similar embeddings
                                suggested = p50  # Use median for high-similarity sets
                            else:  
                                suggested = (p50 + p75) / 2  # Otherwise use between median and 75th percentile
                                
                            # Create histogram
                            fig, ax = plt.subplots(figsize=(10, 6))
                            ax.hist(flat_sim, bins=50)
                            ax.set_title("Embedding Similarity Distribution")
                            ax.set_xlabel("Cosine Similarity")
                            ax.set_ylabel("Count")
                            ax.axvline(x=suggested, color='r', linestyle='--', label=f"Suggested threshold ({suggested:.2f})")
                            ax.axvline(x=p50, color='g', linestyle=':', label=f"Median ({p50:.2f})")
                            ax.axvline(x=p75, color='b', linestyle=':', label=f"75th Percentile ({p75:.2f})")
                            ax.legend()
                            st.pyplot(fig)
                            
                            st.info(f"""
                            **Threshold Recommendations:**
                            - Suggested: {suggested:.2f} ✓ RECOMMENDED
                            - Conservative (fewer, larger clusters): {p75:.2f}
                            - Aggressive (more, smaller clusters): {p50:.2f}
                            
                            With text-embedding-3-small, you should get better clustering results than with previous models!
                            """)
                        else:
                            st.error("OpenAI API key required for embedding diagnostics")
            
            # Display cost information for OpenAI embeddings
            if use_openai_embeddings:
                filtered_df = df
                if selected_a_tags:
                    filtered_df = df[df["A:Tag"].isin(selected_a_tags)]
                
                num_keywords = len(filtered_df)
                est_tokens = num_keywords * 10  # Estimate 10 tokens per keyword on average
                estimated_cost = est_tokens * 0.00002 / 1000
                st.info(f"💰 **Estimated OpenAI embedding cost:** ${estimated_cost:.5f} for {num_keywords} keywords")

            # Display cost information for GPT usage
            if use_gpt_for_topics and api_key:
                # Filter dataset if needed
                filtered_df = df
                if selected_a_tags:
                    filtered_df = df[df["A:Tag"].isin(selected_a_tags)]
                
                # Estimate number of clusters
                est_num_clusters = len(filtered_df["A:Tag"].unique()) * 3  # Rough estimate: ~3 clusters per A:Tag
                total_cost, cost_breakdown = estimate_gpt4o_mini_costs(est_num_clusters, use_gpt_descriptors=True, use_gpt_topics=True)
                
                st.info(f"""
                💰 **Estimated GPT-4o-mini cost:** ${total_cost:.4f}
                - Embeddings: ${cost_breakdown['embeddings']:.4f}
                - Cluster naming: ${cost_breakdown['cluster_descriptors']:.4f}
                - Topic generation: ${cost_breakdown['topic_generation']:.4f}
                - Content strategy: ${cost_breakdown['content_strategy']:.4f}
                
                *Note: Actual costs may vary depending on the number of clusters formed*
                """)
            
            # Run clustering
            if st.button("Generate Content Topics", key="generate_topics_button"):
                if use_gpt_for_topics and not api_key:
                    st.error("Please provide an OpenAI API key to use GPT for topic generation")
                    st.stop()
                
                # Clear memory before processing
                gc.collect()
                
                with st.spinner("Processing keywords and generating topics..."):
                    progress_bar = st.progress(0)
                    
                    # Get models
                    embedding_model, kw_model = get_models()
                    
                    # First, filter by selected A tags if specified
                    if selected_a_tags:
                        df_filtered = df[df["A:Tag"].isin(selected_a_tags)]
                        if len(df_filtered) == 0:
                            st.warning("No keywords match the selected filters")
                            st.stop()
                    else:
                        df_filtered = df.copy()
                    
                    # Reduce dataframe size by selecting only needed columns
                    df_filtered = df_filtered[["Keywords", "A:Tag", "B:Tag", "C:Tag"]]
                    
                    # Apply semantic intent-based clustering
                    progress_bar.progress(0.1)
                    st.text("Applying semantic intent-based clustering by A:Tag...")
                    
                    # Use fast mode for large datasets
                    if use_fast_mode and len(df_filtered) > 500:
                        st.text("Using optimized processing for large dataset...")
                        
                        # Use more aggressive min_shared_keywords to create fewer clusters
                        adaptive_min_shared = min(10, max(5, int(len(df_filtered) / 100)))
                        
                        st.text("Processing clusters... this might take a few minutes for large datasets")
                        df_clustered, cluster_info = two_stage_clustering(
                            df_filtered, 
                            cluster_method=clustering_approach,
                            embedding_model=embedding_model,
                            api_key=api_key,
                            use_openai_embeddings=use_openai_embeddings,
                            min_shared_keywords=min_shared_keywords,
                            similarity_threshold=similarity_threshold
                        )
                            
                        # Generate cluster labels with batching
                        use_gpt_descriptors = use_gpt and api_key
                        cluster_descriptors = generate_cluster_descriptors(cluster_info, use_gpt_descriptors, api_key)
                        
                        # Combine the clustered data with the filtered dataframe
                        df_filtered = df_filtered.reset_index(drop=True)
                        df_filtered["Cluster"] = df_clustered["Cluster"] 
                        df_filtered["A_Group"] = df_clustered["A_Group"]
                        df_filtered["Subcluster"] = df_clustered["Subcluster"]
                        df_filtered["Cluster_Label"] = df_filtered["Cluster"].map(cluster_descriptors)
                        df_filtered["Cluster_Confidence"] = df_clustered["Cluster_Confidence"]
                        df_filtered["Is_Outlier"] = df_clustered["Is_Outlier"]
                        
                        # Initialize data structures
                        cluster_insights = []
                        topic_map = {}
                        
                        # Generate topics in batches
                        if use_gpt_for_topics and api_key:
                            batch_results = batch_generate_topics(cluster_info, api_key, max_clusters_per_batch=5)
                        else:
                            # Create fallback batch results using basic approach
                            batch_results = {}
                            for cluster_id, info in cluster_info.items():
                                a_tag = info["a_tag"]
                                b_tags = info["b_tags"][:3] if info["b_tags"] else []
                                sample_keywords = info["keywords"][:min(15, len(info["keywords"]))]
                                
                                # Generate basic topics
                                topic_ideas, expanded_ideas, value_props = generate_basic_topics(
                                    sample_keywords, [a_tag], b_tags
                                )
                                
                                batch_results[cluster_id] = {
                                    "topic_ideas": topic_ideas,
                                    "expanded_ideas": expanded_ideas,
                                    "value_props": value_props
                                }
                        
                        # Process results into format expected by the rest of the code
                        for cluster_id, info in cluster_info.items():
                            results = batch_results.get(cluster_id, {})
                            
                            topic_ideas = results.get("topic_ideas", [])
                            if not topic_ideas:
                                topic_ideas = [f"Content about {info['a_tag']}"]
                                
                            expanded_ideas = results.get("expanded_ideas", [])
                            if not expanded_ideas or len(expanded_ideas) < len(topic_ideas):
                                expanded_ideas = [f"{idea} (Guide)" for idea in topic_ideas]
                                
                            value_props = results.get("value_props", [])
                            if not value_props or len(value_props) < len(topic_ideas):
                                value_props = ["Provides valuable information on this topic"] * len(topic_ideas)
                            
                            # Filter for high confidence keywords
                            high_conf_df = df_filtered[
                                (df_filtered["Cluster"] == cluster_id) & 
                                (df_filtered["Cluster_Confidence"] >= topic_confidence_threshold)
                            ]
                            
                            # Add to cluster insights
                            cluster_insights.append({
                                "cluster_id": cluster_id,
                                "cluster_label": cluster_descriptors.get(cluster_id, f"Cluster {cluster_id}"),
                                "size": info["size"],
                                "high_confidence_count": len(high_conf_df),
                                "a_tags": [info["a_tag"]],
                                "b_tags": info["b_tags"],
                                "c_tags": info["c_tags"],
                                "keywords": info["keywords"],
                                "high_confidence_keywords": high_conf_df["Keywords"].tolist(),
                                "sample_keywords": info["keywords"][:min(15, len(info["keywords"]))],
                                "topic_phrases": topic_ideas,  # Using topic ideas as topic phrases
                                "topic_ideas": topic_ideas,
                                "expanded_ideas": expanded_ideas,
                                "value_props": value_props,
                                "analysis": f"Cluster of {info['a_tag']} keywords",
                                "is_outlier_cluster": info.get("is_outlier_cluster", False)
                            })
                            
                            # Create topic map
                            if info.get("is_outlier_cluster", False):
                                topic_map[cluster_id] = f"{cluster_descriptors.get(cluster_id, '')}: Miscellaneous"
                            elif topic_ideas:
                                topic_map[cluster_id] = f"{cluster_descriptors.get(cluster_id, '')}: {topic_ideas[0]}"
                            else:
                                topic_map[cluster_id] = cluster_descriptors.get(cluster_id, f"Cluster {cluster_id}")
                    else:
                        # Standard processing for smaller datasets
                        try:
                            with timeout(600):  # 10 minute timeout
                                df_clustered, cluster_info = two_stage_clustering(
                                    df_filtered, 
                                    cluster_method=clustering_approach,
                                    embedding_model=embedding_model,
                                    api_key=api_key,
                                    use_openai_embeddings=use_openai_embeddings,
                                    min_shared_keywords=min_shared_keywords,
                                    similarity_threshold=similarity_threshold  # Will be auto-adjusted if needed
                                )
                        except TimeoutError:
                            st.error("Clustering timed out. Try using fewer keywords or more aggressive clustering settings.")
                            st.stop()
                        
                        # Generate descriptive labels for each cluster
                        use_gpt_descriptors = use_gpt and api_key
                        cluster_descriptors = generate_cluster_descriptors(cluster_info, use_gpt_descriptors, api_key)
                        
                        # Combine the clustered data with the filtered dataframe
                        df_filtered = df_filtered.reset_index(drop=True)
                        df_filtered["Cluster"] = df_clustered["Cluster"] 
                        df_filtered["A_Group"] = df_clustered["A_Group"]
                        df_filtered["Subcluster"] = df_clustered["Subcluster"]
                        df_filtered["Cluster_Label"] = df_filtered["Cluster"].map(cluster_descriptors)
                        df_filtered["Cluster_Confidence"] = df_clustered["Cluster_Confidence"]
                        df_filtered["Is_Outlier"] = df_clustered["Is_Outlier"]
                        
                        progress_bar.progress(0.5)
                        
                        # Generate topic insights for each cluster
                        st.text("Generating content topics for each cluster...")
                        
                        # Reset progress tracking for topic generation
                        cluster_insights = []
                        topic_map = {}  # Map of cluster IDs to topic names
                        
                        # Process each cluster
                        cluster_ids = sorted(cluster_info.keys())
                        for i, cluster_id in enumerate(cluster_ids):
                            info = cluster_info[cluster_id]
                            a_tag = info["a_tag"]
                            is_outlier_cluster = info.get("is_outlier_cluster", False)
                            
                            # Get cluster dataframe
                            cluster_df = df_filtered[df_filtered["Cluster"] == cluster_id]
                            
                            # Filter for high confidence keywords for better quality topics
                            high_conf_df = cluster_df[cluster_df["Cluster_Confidence"] >= topic_confidence_threshold]
                            
                            # Get the keywords from the cluster - use high confidence for topics
                            if not is_outlier_cluster and len(high_conf_df) >= 5:
                                # Use high confidence keywords for regular clusters
                                keywords_for_topics = high_conf_df["Keywords"].tolist()
                                high_confidence_keywords = keywords_for_topics
                            else:
                                # For outlier clusters or small clusters, use all keywords
                                keywords_for_topics = cluster_df["Keywords"].tolist()
                                high_confidence_keywords = cluster_df[cluster_df["Cluster_Confidence"] >= topic_confidence_threshold]["Keywords"].tolist()
                            
                            # Sample keywords for topic generation
                            sample_keywords = keywords_for_topics[:min(30, len(keywords_for_topics))]
                            
                            # Extract top tags
                            top_b_tags = info["b_tags"]
                            
                            # Optional: Get frequency data 
                            kw_freq = None
                            if "Count" in df_filtered.columns:
                                kw_freq = dict(zip(cluster_df["Keywords"], cluster_df["Count"]))
                            
                            # Generate topics with GPT or basic approach
                            if use_gpt_for_topics and api_key:
                                topic_ideas, expanded_ideas, value_props = generate_gpt_topics(
                                    sample_keywords, [a_tag], top_b_tags, api_key, kw_freq
                                )
                            else:
                                # Basic topic generation
                                topic_ideas, expanded_ideas, value_props = generate_basic_topics(
                                    sample_keywords, [a_tag], top_b_tags
                                )
                            
                            # Get cluster label
                            cluster_label = cluster_descriptors[cluster_id] if cluster_id in cluster_descriptors else f"Cluster {cluster_id}"
                            
                            # Store the primary topic name in the mapping (use descriptive label if available)
                            if is_outlier_cluster:
                                topic_map[cluster_id] = f"{cluster_label} (Miscellaneous)"
                            elif topic_ideas:
                                topic_map[cluster_id] = f"{cluster_label}: {topic_ideas[0]}"
                            else:
                                topic_map[cluster_id] = cluster_label
                            
                            # Extract topic phrases for visualization
                            if keywords_for_topics:
                                combined_text = " ".join(sample_keywords)
                                key_phrases = kw_model.extract_keywords(combined_text, keyphrase_ngram_range=(1, 3), top_n=5)
                                topic_phrases = [kp[0] for kp in key_phrases]
                            else:
                                topic_phrases = []
                            
                            # Create a simple analysis description
                            if is_outlier_cluster:
                                cluster_analysis = f"Miscellaneous {a_tag} keywords that didn't share enough semantic intent with others."
                            else:
                                cluster_analysis = f"Cluster of {a_tag} keywords with shared semantic intent focused on {', '.join(top_b_tags[:2] if top_b_tags else ['general attributes'])}"
                            
                            cluster_insights.append({
                                "cluster_id": cluster_id,
                                "cluster_label": cluster_label,
                                "size": len(cluster_df),
                                "high_confidence_count": len(high_conf_df),
                                "a_tags": [a_tag],  # Now always a single A tag per cluster
                                "b_tags": top_b_tags,
                                "c_tags": info["c_tags"],
                                "keywords": cluster_df["Keywords"].tolist(),
                                "high_confidence_keywords": high_confidence_keywords,
                                "sample_keywords": sample_keywords,
                                "topic_phrases": topic_phrases,
                                "topic_ideas": topic_ideas,
                                "expanded_ideas": expanded_ideas,
                                "value_props": value_props,
                                "analysis": cluster_analysis,
                                "is_outlier_cluster": is_outlier_cluster
                            })
                            
                            # Update progress
                            progress_bar.progress(0.5 + (0.4 * (i + 1) / len(cluster_ids)))
                    
                    # Add topic labels to the DataFrame
                    df_filtered["Content_Topic"] = df_filtered["Cluster"].map(topic_map)
                    
                    # Create quality indicator based on confidence
                    df_filtered["Content_Quality"] = df_filtered["Cluster_Confidence"].apply(
                        lambda x: "High" if x >= 0.8 else ("Medium" if x >= 0.6 else "Low")
                    )
                    
                    # Create topic summary table
                    topic_summary = []
                    for insight in cluster_insights:
                        for i, idea in enumerate(insight['topic_ideas']):
                            if i >= len(insight['value_props']):
                                value = ""
                            else:
                                value = insight['value_props'][i]
                                
                            topic_summary.append({
                                "Cluster Label": insight["cluster_label"],
                                "Is Miscellaneous": "Yes" if insight["is_outlier_cluster"] else "No",
                                "Topic": idea,
                                "Format": insight['expanded_ideas'][i].replace(idea, "").strip("() ") if i < len(insight['expanded_ideas']) else "",
                                "Value": value,
                                "Cluster ID": insight['cluster_id'],
                                "A Tag": insight['a_tags'][0],
                                "Keywords": insight['size'],
                                "High Confidence Keywords": insight['high_confidence_count'],
                                "Confidence Ratio": f"{insight['high_confidence_count']/insight['size']*100:.1f}%" if insight['size'] > 0 else "0%",
                                "Main Tags": ", ".join(insight['b_tags'][:3] if insight['b_tags'] else [])
                            })
                    
                    topic_df = pd.DataFrame(topic_summary) if topic_summary else None
                    
                    # Complete progress
                    progress_bar.progress(1.0)
                    progress_bar.empty()
                    
                    # Store in session state
                    st.session_state.content_topics_processed = True
                    st.session_state.df_filtered = df_filtered
                    st.session_state.cluster_insights = cluster_insights
                    st.session_state.topic_df = topic_df
                    st.session_state.topic_map = topic_map
                    st.session_state.cluster_info = cluster_info  # Store cluster info for visualization
                    st.session_state.cluster_descriptors = cluster_descriptors  # Store cluster descriptors
                    st.session_state['stored_topic_conf_threshold'] = topic_confidence_threshold # Store confidence threshold
    
    # Display results from session state if already processed
    if 'content_topics_processed' in st.session_state and st.session_state.content_topics_processed:
        df_filtered = st.session_state.df_filtered
        cluster_insights = st.session_state.cluster_insights
        topic_df = st.session_state.topic_df
        
        # Get confidence threshold from session state
        topic_confidence_threshold = st.session_state.get('stored_topic_conf_threshold', st.session_state.get('topic_confidence_threshold', 0.6))
        
        # Define topic_map from session state or recreate it if needed
        if 'topic_map' in st.session_state:
            topic_map = st.session_state.topic_map
        else:
            # Recreate the mapping as a fallback
            topic_map = {}
            for i, insight in enumerate(cluster_insights):
                if insight["topic_ideas"] and len(insight["topic_ideas"]) > 0:
                    topic_map[i] = insight["topic_ideas"][0]
                else:
                    topic_map[i] = f"Cluster {i}"
        
        # Retrieve cluster_info from session state
        if 'cluster_info' in st.session_state:
            cluster_info = st.session_state.cluster_info
            
            # Show semantic intent-based clustering visualization
            st.subheader("Semantic Intent-Based Cluster Distribution")
            
            # Get cluster descriptors
            cluster_descriptors = st.session_state.cluster_descriptors if 'cluster_descriptors' in st.session_state else None
            
            # Count clusters and outliers
            total_keywords = len(df_filtered)
            total_clusters = len([k for k, v in cluster_info.items() if not v.get("is_outlier_cluster", False)])
            total_outliers = len([k for k, v in cluster_info.items() if v.get("is_outlier_cluster", False)])
            outlier_keywords = df_filtered[df_filtered["Is_Outlier"]].shape[0]
            outlier_percent = (outlier_keywords / total_keywords) * 100 if total_keywords > 0 else 0
            
            # Show summary statistics
            st.markdown(f"""
            **Clustering Summary:**
            - Total Keywords: {total_keywords}
            - Semantic Clusters Formed: {total_clusters}
            - Miscellaneous Groups: {total_outliers}
            - Outlier Keywords: {outlier_keywords} ({outlier_percent:.1f}%)
            """)
            
            # Visualization of A tag groups and their clusters
            fig = create_two_stage_visualization(df_filtered, cluster_info, cluster_descriptors)
            st.pyplot(fig)
            
            # Show topic summary and download with unique key
            st.subheader("Content Topic Ideas")
            st.dataframe(topic_df)
            
            csv_topics = topic_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "Download Content Topics CSV",
                csv_topics,
                "topic_ideas.csv",
                "text/csv",
                key="download_topic_ideas"
            )
            
            # Create download buttons with unique keys
            st.subheader("Export Results")

            # Add dedicated keyword-topic mapping
            st.subheader("Keyword-Topic Mapping")
            st.info("This file shows which keywords belong to each content topic, helping you understand which keywords each topic should target.")
            
            # Create the keyword-topic mapping DataFrame
            keyword_topic_df = create_keyword_topic_mapping(df_filtered)
            
            # Create download button
            keyword_topic_csv = keyword_topic_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "Download Keyword-Topic Mapping CSV",
                keyword_topic_csv,
                "keyword_topic_mapping.csv",
                "text/csv",
                key="download_keyword_topic_mapping"
            )

            # Also create a topic-centered version that groups by topic
            st.info("For easier understanding of which keywords go with which topic:")
            
            # Create grouped version
            keyword_groups = []
            for cluster_id, topic in topic_map.items():
                cluster_keywords = df_filtered[df_filtered["Cluster"] == cluster_id]["Keywords"].tolist()
                cluster_name = cluster_descriptors.get(cluster_id, f"Cluster {cluster_id}")
                
                # Skip if no keywords
                if not cluster_keywords:
                    continue
                    
                # Add row for each keyword in this cluster
                for keyword in cluster_keywords:
                    keyword_groups.append({
                        "Content Topic": topic,
                        "Cluster Name": cluster_name,
                        "Keyword": keyword
                    })
            
            if keyword_groups:
                topic_keywords_df = pd.DataFrame(keyword_groups)
                topic_keywords_df = topic_keywords_df.sort_values(["Content Topic", "Cluster Name"])
                
                # Create download button for topic-centered version
                topic_csv = topic_keywords_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "Download Topic-Centered Keyword Mapping",
                    topic_csv,
                    "topic_keyword_mapping.csv",
                    "text/csv",
                    key="download_topic_centered_mapping"
                )
            
            csv_result = df_filtered.to_csv(index=False).encode('utf-8')
            st.download_button(
                "Download Clustered Keywords CSV",
                csv_result,
                "content_topics.csv",
                "text/csv",
                key="download_clustered_keywords"
            )
            
            # Also offer a high confidence version
            high_conf_df = df_filtered[df_filtered["Cluster_Confidence"] >= topic_confidence_threshold]
            high_conf_csv = high_conf_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "Download High Confidence Keywords Only",
                high_conf_csv,
                "high_confidence_keywords.csv",
                "text/csv",
                key="download_high_conf_keywords"
            )
            
            # Show a content strategy overview if using GPT
            if use_gpt and api_key:
                st.subheader("📋 Content Strategy Overview")
                if st.button("Generate Content Strategy Recommendations", key="content_strategy_button"):
                    with st.spinner("Creating content strategy recommendations..."):
                        try:
                            # Filter to non-miscellaneous topics with reasonable confidence
                            quality_topics = topic_df[topic_df["Is Miscellaneous"] == "No"].copy()
                            quality_topics["Confidence"] = quality_topics["High Confidence Keywords"] / quality_topics["Keywords"]
                            quality_topics = quality_topics.sort_values("Confidence", ascending=False)
                            
                            # Extract topic information for the prompt
                            top_topics = [
                                f"{row['Topic']} ({row['Format']}): {row['High Confidence Keywords']} high confidence keywords" 
                                for _, row in quality_topics.head(8).iterrows()
                            ]
                            
                            # Get the distribution of A tags
                            a_tag_counts = df_filtered["A:Tag"].value_counts()
                            a_tag_info = [f"{tag}: {count}" for tag, count in a_tag_counts.items()]
                            
                            openai.api_key = api_key
                            response = openai.chat.completions.create(
                                model="gpt-4o-mini",
                                messages=[{"role": "user", "content": f"""
                                As a content strategist, create a brief content plan based on these keyword clusters.
                                
                                HIGHEST QUALITY CONTENT TOPICS:
                                {', '.join(top_topics)}
                                
                                KEYWORD CATEGORY DISTRIBUTION:
                                {', '.join(a_tag_info)}
                                
                                Please provide:
                                1. Top 3 priority content topics and why they should be prioritized
                                2. How these topics could be connected in a content hub structure
                                3. A suggestion for seasonal or evergreen content based on these topics
                                
                                Focus on the highest quality clusters that have the most high-confidence keywords.
                                Keep recommendations specific and actionable.
                                """}],
                                temperature=0.5,
                                max_tokens=800
                            )
                            
                            strategy = response.choices[0].message.content
                            st.markdown(strategy)
                        except Exception as e:
                            st.error(f"Error generating content strategy: {e}")
        else:
            st.error("Cluster information is missing. Please reprocess the data.")
            st.stop()
