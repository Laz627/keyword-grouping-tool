# main.py
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
from sklearn.cluster import KMeans

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

# Enhanced Two-Stage Clustering with Outlier Detection
def two_stage_clustering(df, num_clusters, cluster_method="Tag-based", embedding_model=None, outlier_threshold=1.5):
    """
    Perform two-stage clustering with outlier detection:
    1. Group by A tag
    2. Within each A tag group, cluster by B and C tags
    3. Detect and mark outliers based on distance from cluster center
    
    Parameters:
    -----------
    df : DataFrame
        DataFrame containing Keywords, A:Tag, B:Tag, C:Tag columns
    num_clusters : int
        Maximum number of clusters to create within each A tag group
    cluster_method : str
        Method for clustering within A tag groups: "Tag-based", "Semantic", "Hybrid"
    embedding_model : SentenceTransformer model
        Model for generating embeddings (required for semantic and hybrid methods)
    outlier_threshold : float
        Threshold for outlier detection (standard deviations from mean distance)
        
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
            
        # Determine number of clusters for this group
        # Adaptive based on group size - smaller groups get fewer clusters
        if group_size < 5:
            # Very small groups don't need clustering
            n_clusters = 1
        else:
            # Scale clusters based on group size, with a maximum of num_clusters
            n_clusters = min(num_clusters, max(1, group_size // 5))
        
        # Create a copy of the group with index reset
        group_df = group_df.copy().reset_index(drop=True)
        
        # Add A_Group column
        group_df["A_Group"] = a_tag
        
        # For confidence scoring
        group_df["Cluster_Confidence"] = 1.0  # Default confidence
        group_df["Is_Outlier"] = False  # Default not an outlier
        
        # Handle clustering within this A tag group
        if group_size <= n_clusters:
            # Not enough samples to cluster meaningfully
            group_df["Subcluster"] = 0
        else:
            # Enough samples to cluster - generate features based on selected method
            if cluster_method == "Tag-based":
                # Use only B and C tags for clustering within A tag groups
                b_dummies = pd.get_dummies(group_df["B:Tag"], prefix="B")
                c_dummies = pd.get_dummies(group_df["C:Tag"], prefix="C")
                
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
                if "Subcluster" not in group_df.columns:
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                    group_df["Subcluster"] = kmeans.fit_predict(features)
                    
                    # Calculate distance to cluster center for confidence scoring
                    for cluster_idx in range(n_clusters):
                        mask = group_df["Subcluster"] == cluster_idx
                        if mask.sum() > 0:
                            # Get cluster center (centroid)
                            centroid = kmeans.cluster_centers_[cluster_idx]
                            
                            # Calculate distances for this subcluster
                            subcluster_features = features[mask]
                            distances = np.linalg.norm(subcluster_features - centroid, axis=1)
                            
                            # Calculate confidence scores (inverse of normalized distance)
                            max_dist = distances.max() if len(distances) > 0 else 1.0
                            if max_dist > 0:
                                confidences = 1.0 - (distances / max_dist)
                            else:
                                confidences = np.ones(len(distances))
                            
                            # Detect outliers (using standard deviation)
                            if len(distances) > 1:
                                threshold = distances.mean() + outlier_threshold * distances.std()
                                outliers = distances > threshold
                            else:
                                outliers = np.zeros(len(distances), dtype=bool)
                            
                            # Assign confidence and outlier status
                            group_df.loc[mask, "Cluster_Confidence"] = confidences
                            group_df.loc[mask, "Is_Outlier"] = outliers
                    
            elif cluster_method == "Semantic":
                # Use keyword embeddings
                keywords = group_df["Keywords"].tolist()
                
                if not keywords or embedding_model is None:
                    group_df["Subcluster"] = 0
                else:
                    embeddings = embedding_model.encode(keywords)
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                    group_df["Subcluster"] = kmeans.fit_predict(embeddings)
                    
                    # Calculate distance to cluster center for confidence scoring
                    for cluster_idx in range(n_clusters):
                        mask = group_df["Subcluster"] == cluster_idx
                        if mask.sum() > 0:
                            # Get cluster center (centroid)
                            cluster_embeddings = embeddings[mask]
                            centroid = kmeans.cluster_centers_[cluster_idx]
                            
                            # Calculate distances
                            distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)
                            
                            # Calculate confidence scores
                            max_dist = distances.max() if len(distances) > 0 else 1.0
                            if max_dist > 0:
                                confidences = 1.0 - (distances / max_dist)
                            else:
                                confidences = np.ones(len(distances))
                            
                            # Detect outliers
                            if len(distances) > 1:
                                threshold = distances.mean() + outlier_threshold * distances.std()
                                outliers = distances > threshold
                            else:
                                outliers = np.zeros(len(distances), dtype=bool)
                            
                            # Assign confidence and outlier status
                            group_df.loc[mask, "Cluster_Confidence"] = confidences
                            group_df.loc[mask, "Is_Outlier"] = outliers
                    
            else:  # Hybrid
                # Combine keywords with B and C tags only
                combined_texts = group_df.apply(
                    lambda row: f"{row['Keywords']} {row['B:Tag']} {row['C:Tag']}",
                    axis=1
                ).tolist()
                
                if not combined_texts or embedding_model is None:
                    group_df["Subcluster"] = 0
                else:
                    embeddings = embedding_model.encode(combined_texts)
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                    group_df["Subcluster"] = kmeans.fit_predict(embeddings)
                    
                    # Calculate distance to cluster center for confidence scoring
                    for cluster_idx in range(n_clusters):
                        mask = group_df["Subcluster"] == cluster_idx
                        if mask.sum() > 0:
                            # Get cluster center (centroid)
                            cluster_embeddings = embeddings[mask]
                            centroid = kmeans.cluster_centers_[cluster_idx]
                            
                            # Calculate distances
                            distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)
                            
                            # Calculate confidence scores
                            max_dist = distances.max() if len(distances) > 0 else 1.0
                            if max_dist > 0:
                                confidences = 1.0 - (distances / max_dist)
                            else:
                                confidences = np.ones(len(distances))
                            
                            # Detect outliers
                            if len(distances) > 1:
                                threshold = distances.mean() + outlier_threshold * distances.std()
                                outliers = distances > threshold
                            else:
                                outliers = np.zeros(len(distances), dtype=bool)
                            
                            # Assign confidence and outlier status
                            group_df.loc[mask, "Cluster_Confidence"] = confidences
                            group_df.loc[mask, "Is_Outlier"] = outliers
        
        # Create a unique global cluster ID for each subcluster
        subcluster_map = {}
        subclusters = group_df["Subcluster"].unique()
        
        # Create an outlier cluster for this A tag
        outlier_cluster_id = global_cluster_id + len(subclusters)
        
        # Process regular subclusters
        for subcluster in subclusters:
            subcluster_map[subcluster] = global_cluster_id
            
            # Get the subcluster dataframe excluding outliers
            subcluster_df = group_df[(group_df["Subcluster"] == subcluster) & (~group_df["Is_Outlier"])]
            
            # Get the most common B and C tags in this cluster
            top_b_tags = subcluster_df["B:Tag"].value_counts().head(3).index.tolist()
            top_c_tags = subcluster_df["C:Tag"].value_counts().head(3).index.tolist()
            
            # Get keywords in this cluster
            keywords_in_cluster = subcluster_df["Keywords"].tolist()
            
            # Only create the cluster if it has non-outlier keywords
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
        outlier_df = group_df[group_df["Is_Outlier"]]
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
            outlier_indices = outlier_df.index
            group_df.loc[outlier_indices, "Subcluster"] = -1  # Use -1 to mark outliers
            subcluster_map[-1] = outlier_cluster_id
        
        # Map subclusters to global cluster IDs
        group_df["Cluster"] = group_df["Subcluster"].map(subcluster_map)
        
        # Add to the result dataframe
        clustered_df = pd.concat([clustered_df, group_df], ignore_index=True)
    
    return clustered_df, cluster_info

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

# Helper Functions for GPT Integration and Clustering
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

def get_gpt_cluster_insights(cluster_df, api_key):
    """Generate high-level insights about a cluster using GPT."""
    if len(cluster_df) < 5:
        return "Not enough keywords for meaningful analysis."
        
    # Sample keywords for the prompt
    sample_kws = cluster_df["Keywords"].sample(min(20, len(cluster_df))).tolist()
    
    # Get the most common tags
    a_tags = cluster_df["A:Tag"].value_counts().head(3).index.tolist()
    b_tags = cluster_df["B:Tag"].value_counts().head(3).index.tolist()
    
    # Create the prompt
    prompt = f"""Analyze this set of related keywords and identify key patterns or themes:

KEYWORDS:
{', '.join(sample_kws)}

COMMON CATEGORY TAGS: {', '.join(a_tags)}
COMMON ATTRIBUTE TAGS: {', '.join(b_tags)}

Write a concise 2-3 sentence insight about what these keywords represent in terms of:
1. User intent (what are users trying to accomplish?)
2. Content needs (what information would best serve these searches?)

Keep your analysis brief but insightful."""

    try:
        openai.api_key = api_key
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=150
        )
        
        insight = response.choices[0].message.content.strip()
        return insight
    except Exception as e:
        return f"Unable to generate insights. Please check your API key and try again."

def create_cluster_analysis_word_doc(cluster_insights):
    """Create a formatted Word document with cluster analysis."""
    if not DOCX_AVAILABLE:
        raise ImportError("python-docx is required for Word document export")
    
    doc = Document()
    
    # Add title
    title = doc.add_heading('Keyword Cluster Analysis Report', level=0)
    
    # Add TOC instructions
    p = doc.add_paragraph()
    p.add_run('TABLE OF CONTENTS').bold = True
    doc.add_paragraph('To generate a table of contents, go to the "References" tab in Word and click "Table of Contents"')
    
    # Add a page break after instructions
    doc.add_page_break()
    
    # Loop through each cluster to add content
    for i, insight in enumerate(cluster_insights):
        cluster_id = insight["cluster_id"]
        size = insight["size"]
        
        # Create cluster heading
        if "cluster_label" in insight:
            cluster_heading = f"{insight['cluster_label']} ({size} keywords)"
        elif insight["a_tags"]:
            main_tag = insight["a_tags"][0]
            cluster_heading = f"Cluster {cluster_id}: {main_tag.title()} ({size} keywords)"
        else:
            cluster_heading = f"Cluster {cluster_id}: {size} keywords"
        
        doc.add_heading(cluster_heading, level=1)
        
        # Add cluster analysis
        doc.add_heading('Cluster Analysis', level=2)
        doc.add_paragraph(insight['analysis'])
        
        # Add tag information
        if insight['a_tags']:
            doc.add_heading('Main Tags', level=2)
            doc.add_paragraph(', '.join(insight['a_tags']))
        
        if insight['b_tags']:
            doc.add_heading('Secondary Tags', level=2)
            doc.add_paragraph(', '.join(insight['b_tags']))
        
        # Add topic phrases
        if insight['topic_phrases']:
            doc.add_heading('Key Phrases', level=2)
            doc.add_paragraph(', '.join(insight['topic_phrases']))
        
        # Add content topic suggestions
        if insight['topic_ideas']:
            doc.add_heading('Suggested Content Topics', level=2)
            
            for j, (idea, value) in enumerate(zip(insight['expanded_ideas'], insight['value_props'])):
                topic_num = j + 1
                run = doc.add_paragraph().add_run(f"Topic {topic_num}: {idea}")
                run.bold = True
                doc.add_paragraph(value)
                doc.add_paragraph() # Add some spacing
        
        # Add keywords in a table - only include high confidence ones if available
        doc.add_heading('Keywords', level=2)
        
        # Filter for non-outlier keywords with high confidence
        if 'is_outlier' in insight and 'high_confidence_keywords' in insight:
            keywords = insight['high_confidence_keywords']
            doc.add_paragraph("Note: Showing only high-confidence keywords that closely match this cluster.")
        else:
            keywords = insight['keywords']
        
        # Calculate number of columns and rows for the keyword table
        cols = 3  # We'll use 3 columns
        
        if keywords:
            # Determine number of rows needed
            rows = (len(keywords) + cols - 1) // cols
            
            table = doc.add_table(rows=rows, cols=cols)
            table.style = 'Table Grid'
            
            # Fill the table with keywords
            keyword_idx = 0
            for row in range(rows):
                for col in range(cols):
                    if keyword_idx < len(keywords):
                        cell = table.cell(row, col)
                        cell.text = keywords[keyword_idx]
                        keyword_idx += 1
        
        # Add a page break between clusters (except the last one)
        if i < len(cluster_insights) - 1:
            doc.add_page_break()
    
    # Save to a BytesIO object
    doc_buffer = BytesIO()
    doc.save(doc_buffer)
    doc_buffer.seek(0)
    
    return doc_buffer

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
    api_key = st.text_input("OpenAI API Key (for topic generation)", type="password")
    use_gpt = st.checkbox("Use GPT-4o-mini for enhanced analysis", value=True)
    
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

        if clust > 0 and len(c_map) >= clust:
            st.subheader("Theme Clusters")
            with st.spinner("Clustering themes..."):
                embedding_model, _ = get_models()
                
                reps = list(c_map.keys())
                emb = embedding_model.encode(reps)
                
                # Apply KMeans
                km = KMeans(n_clusters=clust, random_state=42)
                labs = km.fit_predict(emb)
                
                # Calculate distance to cluster center for outlier detection
                distances = []
                for i, lab in enumerate(labs):
                    centroid = km.cluster_centers_[lab]
                    distance = np.linalg.norm(emb[i] - centroid)
                    distances.append(distance)
                
                # Convert to numpy array
                distances = np.array(distances)
                
                # Calculate confidence scores
                max_dist = np.max(distances)
                confidences = 1.0 - (distances / max_dist)
                
                # Detect outliers using standard deviation
                threshold = np.mean(distances) + 1.5 * np.std(distances)
                is_outlier = distances > threshold
                
                # Create a dataframe for easier viewing
                cluster_df = pd.DataFrame({
                    "Theme": reps,
                    "Cluster": labs,
                    "Frequency": [c_map[rep] for rep in reps],
                    "Distance": distances,
                    "Confidence": confidences,
                    "Is_Outlier": is_outlier
                })
                
                # Create descriptive labels
                if use_gpt and api_key:
                    # Create cluster info in the format expected by generate_cluster_descriptors
                    theme_cluster_info = {}
                    
                    # First create regular clusters
                    for i in range(clust):
                        # Get non-outlier themes in this cluster
                        cluster_themes = cluster_df[(cluster_df["Cluster"] == i) & (~cluster_df["Is_Outlier"])]["Theme"].tolist()
                        
                        if cluster_themes:
                            a_tags = cdf[cdf["Candidate Theme"].isin(cluster_themes)]["A:Tag"].value_counts().head(1).index.tolist()
                            b_tags = cdf[cdf["Candidate Theme"].isin(cluster_themes)]["B:Tag"].value_counts().head(3).index.tolist()
                            
                            theme_cluster_info[i] = {
                                "a_tag": a_tags[0] if a_tags else "general",
                                "b_tags": b_tags,
                                "c_tags": [],
                                "keywords": cluster_themes,
                                "is_outlier_cluster": False
                            }
                    
                    # Now create one outlier cluster per unique A:Tag
                    outlier_themes = cluster_df[cluster_df["Is_Outlier"]]["Theme"].tolist()
                    if outlier_themes:
                        outlier_a_tags = cdf[cdf["Candidate Theme"].isin(outlier_themes)]["A:Tag"].unique()
                        
                        for i, a_tag in enumerate(outlier_a_tags):
                            # Get outlier themes with this A:Tag
                            a_tag_themes = cdf[(cdf["Candidate Theme"].isin(outlier_themes)) & (cdf["A:Tag"] == a_tag)]["Candidate Theme"].tolist()
                            if a_tag_themes:
                                b_tags = cdf[cdf["Candidate Theme"].isin(a_tag_themes)]["B:Tag"].value_counts().head(3).index.tolist()
                                
                                outlier_id = clust + i
                                theme_cluster_info[outlier_id] = {
                                    "a_tag": a_tag,
                                    "b_tags": b_tags,
                                    "c_tags": [],
                                    "keywords": a_tag_themes,
                                    "is_outlier_cluster": True
                                }
                    
                    # Generate descriptive labels
                    cluster_labels = generate_cluster_descriptors(theme_cluster_info, True, api_key)
                    
                    # Map regular clusters to their labels
                    label_map = {}
                    for i in range(clust):
                        if i in cluster_labels:
                            label_map[i] = cluster_labels[i]
                        else:
                            label_map[i] = f"Cluster {i}"
                    
                    # Map outliers to their A:Tag's outlier cluster label
                    for idx, row in cluster_df[cluster_df["Is_Outlier"]].iterrows():
                        theme = row["Theme"]
                        a_tag = cdf[cdf["Candidate Theme"] == theme]["A:Tag"].iloc[0]
                        
                        # Find matching outlier cluster
                        for cluster_id, info in theme_cluster_info.items():
                            if info["is_outlier_cluster"] and info["a_tag"] == a_tag:
                                label_map[row["Cluster"]] = cluster_labels[cluster_id]
                                break
                        else:
                            # If no matching outlier cluster found
                            label_map[row["Cluster"]] = f"{a_tag.title()}-Miscellaneous"
                    
                    # Add labels to dataframe
                    cluster_df["Cluster_Label"] = cluster_df["Cluster"].map(label_map)
                    
                    # For outliers, mark their label with "(Misc.)"
                    cluster_df.loc[cluster_df["Is_Outlier"], "Cluster_Label"] = \
                        cluster_df.loc[cluster_df["Is_Outlier"], "Cluster_Label"].apply(
                            lambda x: f"{x} (Misc.)" if not x.endswith("Miscellaneous") else x
                        )
                
                # Show clusters in tabs with outlier information
                st.markdown("### Cluster Overview")
                
                # Create a confidence filter
                confidence_threshold = st.slider(
                    "Filter by confidence score", 
                    min_value=0.0, 
                    max_value=1.0, 
                    value=0.5,
                    step=0.05,
                    help="Higher values show only keywords that more closely match their cluster"
                )
                
                # Create tabs for each cluster
                tabs = st.tabs([f"Cluster {i}" for i in range(clust)])
                for i, tab in enumerate(tabs):
                    with tab:
                        # Get all data for this cluster
                        cluster_data = cluster_df[cluster_df["Cluster"] == i]
                        
                        # Get filtered data (high confidence only)
                        high_conf_data = cluster_data[cluster_data["Confidence"] >= confidence_threshold]
                        
                        # Display cluster label if available
                        if "Cluster_Label" in cluster_data.columns:
                            cluster_label = cluster_data["Cluster_Label"].iloc[0].split(" (Misc.)")[0]
                            st.subheader(f"Cluster {i}: {cluster_label}")
                        
                        # Show counts
                        st.markdown(f"**Total themes:** {len(cluster_data)}")
                        st.markdown(f"**High confidence themes:** {len(high_conf_data)} (confidence ≥ {confidence_threshold:.2f})")
                        st.markdown(f"**Outlier themes:** {cluster_data['Is_Outlier'].sum()}")
                        
                        # Show high confidence themes
                        st.markdown("#### High Confidence Themes")
                        st.dataframe(high_conf_data.sort_values(["Is_Outlier", "Confidence"], ascending=[True, False]))
                
                # Create a separate tab for outliers
                with st.expander("View Outlier Themes"):
                    outlier_data = cluster_df[cluster_df["Is_Outlier"]]
                    if len(outlier_data) > 0:
                        st.dataframe(outlier_data.sort_values("Confidence"))
                    else:
                        st.write("No outlier themes detected.")

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
                        
        # Add the AI clustering section with two-stage approach and outlier detection
        st.subheader("🧩 AI-Powered Tag Clustering")
        st.markdown("""
        Discover how your keywords naturally cluster beyond the assigned tags using a two-stage approach that first groups by A:Tag,
        then clusters by B and C tags within each A:Tag group.
        """)

        st.info("""
        **Enhanced Clustering with Outlier Detection:**
        - Keywords are first grouped by their A:Tag (primary category)
        - Within each A:Tag group, keywords are clustered based on B and C tags (or semantic meaning)
        - Keywords that don't fit well in any cluster are placed in "Miscellaneous" groups
        - A confidence score (0-1) indicates how well each keyword matches its cluster
        """)

        # Create cluster options
        clustering_col1, clustering_col2 = st.columns(2)

        with clustering_col1:
            num_clusters = st.slider("Max clusters per A:Tag group", 
                                   min_value=2, 
                                   max_value=10, 
                                   value=min(5, len(df) // 20),
                                   key="tag_num_clusters")
            
            outlier_threshold = st.slider("Outlier sensitivity", 
                                        min_value=1.0, 
                                        max_value=3.0, 
                                        value=1.5, 
                                        step=0.1,
                                        help="Lower values detect more outliers (1.0-3.0)",
                                        key="outlier_threshold")

        with clustering_col2:
            cluster_method = st.radio(
                "Clustering method:",
                ["Tag-based", "Semantic", "Hybrid"],
                key="tag_cluster_method",
                help="""
                Tag-based: Cluster by B and C tags
                Semantic: Cluster by keyword meaning
                Hybrid: Combine tags and meaning
                """
            )
            
            confidence_filter = st.slider(
                "Confidence threshold", 
                min_value=0.0, 
                max_value=1.0, 
                value=0.6,
                step=0.05, 
                help="Only include keywords with confidence scores above this threshold",
                key="confidence_threshold"
            )

        if st.button("Generate Tag Clusters", key="generate_tag_clusters"):
            with st.spinner("Analyzing keyword patterns with outlier detection..."):
                # Get models
                embedding_model, _ = get_models()
                
                # Use the enhanced two-stage clustering approach with outlier detection
                df_clustered, cluster_info = two_stage_clustering(
                    df, 
                    num_clusters=num_clusters, 
                    cluster_method=cluster_method,
                    embedding_model=embedding_model,
                    outlier_threshold=outlier_threshold
                )
                
                # Generate descriptive labels for each cluster
                use_gpt_descriptors = use_gpt and api_key
                cluster_descriptors = generate_cluster_descriptors(cluster_info, use_gpt_descriptors, api_key)
                
                # Add cluster assignments to the main dataframe
                df["Cluster"] = df_clustered["Cluster"]
                df["A_Group"] = df_clustered["A_Group"]
                df["Subcluster"] = df_clustered["Subcluster"]
                df["Cluster_Label"] = df["Cluster"].map(cluster_descriptors)
                df["Cluster_Confidence"] = df_clustered["Cluster_Confidence"]
                df["Is_Outlier"] = df_clustered["Is_Outlier"]
                
                # Show overview of clusters
                st.subheader("Two-Stage Clustering Results")
                
                # Visualization of A tag groups and their clusters
                fig = create_two_stage_visualization(df_clustered, cluster_info, cluster_descriptors)
                st.pyplot(fig)
                
                # Add confidence score summary statistics
                st.subheader("Confidence Score Distribution")
                
                # Create confidence score statistics
                conf_stats = {
                    "Mean Confidence": df["Cluster_Confidence"].mean(),
                    "Median Confidence": df["Cluster_Confidence"].median(),
                    "Min Confidence": df["Cluster_Confidence"].min(),
                    "Max Confidence": df["Cluster_Confidence"].max(),
                    "Standard Deviation": df["Cluster_Confidence"].std(),
                    "Total Keywords": len(df),
                    "High Confidence Keywords": len(df[df["Cluster_Confidence"] >= confidence_filter]),
                    "Outlier Keywords": df["Is_Outlier"].sum()
                }
                
                # Display confidence statistics
                conf_df = pd.DataFrame([conf_stats])
                st.dataframe(conf_df)
                
                # Create a histogram of confidence scores
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.hist(df["Cluster_Confidence"], bins=20, alpha=0.7, color="steelblue")
                ax.axvline(x=confidence_filter, color='red', linestyle='--', 
                          label=f'Threshold = {confidence_filter}')
                ax.set_title("Distribution of Cluster Confidence Scores")
                ax.set_xlabel("Confidence Score")
                ax.set_ylabel("Count")
                ax.legend()
                st.pyplot(fig)
                
                # Prepare filtered dataframe for better visualizations (excluding outliers or low confidence)
                high_conf_df = df[df["Cluster_Confidence"] >= confidence_filter]
                
                # Show clusters with their properties
                st.markdown("### Clusters by A:Tag")
                
                # Group clusters by A tag for organized display
                a_tags = sorted(df["A_Group"].unique())
                
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
                                st.info("This is a miscellaneous group containing keywords that didn't fit well in other clusters.")
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
                            high_conf_kws = cluster_df[cluster_df["Cluster_Confidence"] >= confidence_filter]
                            
                            # Show confidence stats
                            st.markdown(f"**Avg. Confidence:** {cluster_df['Cluster_Confidence'].mean():.2f}")
                            st.markdown(f"**High confidence keywords:** {len(high_conf_kws)} of {len(cluster_df)} ({len(high_conf_kws)/len(cluster_df)*100:.1f}%)")
                            
                            # Show sample keywords
                            st.markdown("**Sample Keywords:**")
                            
                            # Use tabs to separate high confidence vs all keywords
                            kw_tabs = st.tabs(["High Confidence Keywords", "All Keywords"])
                            
                            with kw_tabs[0]:
                                # Show high confidence keywords
                                high_conf_sample = high_conf_kws.sort_values("Cluster_Confidence", ascending=False).head(15)
                                if len(high_conf_sample) > 0:
                                    sample_df = high_conf_sample[["Keywords", "Cluster_Confidence"]].reset_index(drop=True)
                                    sample_df["Cluster_Confidence"] = sample_df["Cluster_Confidence"].apply(lambda x: f"{x:.2f}")
                                    st.dataframe(sample_df, hide_index=False)
                                else:
                                    st.info(f"No keywords with confidence ≥ {confidence_filter}")
                            
                            with kw_tabs[1]:
                                # Show all keywords
                                sample_df = cluster_df[["Keywords", "Cluster_Confidence", "Is_Outlier"]].sort_values(
                                    ["Is_Outlier", "Cluster_Confidence"], 
                                    ascending=[True, False]
                                ).head(15).reset_index(drop=True)
                                sample_df["Cluster_Confidence"] = sample_df["Cluster_Confidence"].apply(lambda x: f"{x:.2f}")
                                st.dataframe(sample_df, hide_index=False)
                            
                            st.markdown("---")
                
                # Add GPT insights if enabled
                if use_gpt and api_key:
                    st.subheader("💡 Cluster Insights")
                    if st.button("Generate Cluster Insights with GPT", key="tag_cluster_insights"):
                        with st.spinner("Analyzing clusters with GPT..."):
                            try:
                                openai.api_key = api_key
                                
                                # Process each A tag
                                for a_tag in a_tags:
                                    # Get regular (non-outlier) clusters for this A tag
                                    a_clusters = [k for k, v in cluster_info.items() 
                                                if v["a_tag"] == a_tag and not v.get("is_outlier_cluster", False)]
                                    
                                    if not a_clusters:
                                        continue
                                        
                                    with st.expander(f"Insights for A:Tag: {a_tag}"):
                                        # Prepare cluster data for GPT
                                        cluster_data = []
                                        for cluster_id in a_clusters:
                                            info = cluster_info[cluster_id]
                                            label = cluster_descriptors[cluster_id] if cluster_id in cluster_descriptors else f"Cluster {cluster_id}"
                                            
                                            # Get only high confidence keywords
                                            high_conf_kws = df[(df["Cluster"] == cluster_id) & 
                                                             (df["Cluster_Confidence"] >= confidence_filter)]["Keywords"].tolist()
                                            
                                            cluster_data.append({
                                                "id": label,
                                                "size": len(high_conf_kws),
                                                "b_tags": info["b_tags"],
                                                "c_tags": info["c_tags"],
                                                "sample_keywords": high_conf_kws[:10]
                                            })
                                        
                                        # Create the prompt
                                        prompt = f"""Analyze these keyword clusters for A:Tag '{a_tag}':

{json.dumps(cluster_data, indent=2)}

Please provide:
1. How these clusters differ from each other based on B and C tags
2. What user intent is revealed by each cluster
3. Content recommendations for each cluster

Note: These clusters only contain keywords with high confidence scores that match well with their cluster.
Keep your analysis concise and focused on B and C tag patterns."""
                                        
                                        response = openai.chat.completions.create(
                                            model="gpt-4o-mini",
                                            messages=[{"role": "user", "content": prompt}],
                                            temperature=0.5,
                                            max_tokens=500
                                        )
                                        
                                        insights = response.choices[0].message.content
                                        st.markdown(insights)
                            except Exception as e:
                                st.error(f"Error generating insights: {e}")
                
                # Add download button for clusters with confidence scores
                # Add additional columns to indicate quality of the clustering
                df["Cluster_Quality"] = df["Cluster_Confidence"].apply(
                    lambda x: "High" if x >= 0.8 else ("Medium" if x >= 0.6 else "Low")
                )
                
                clustered_csv = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download Clustered Tags",
                    clustered_csv,
                    "tagged_keyword_clusters.csv",
                    "text/csv",
                    key="download_tag_clusters"
                )
                
                # Also offer a filtered version with only high confidence keywords
                high_conf_csv = high_conf_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download High-Confidence Keywords Only",
                    high_conf_csv,
                    "high_confidence_keywords.csv",
                    "text/csv",
                    key="download_high_conf"
                )

elif mode == "Content Topic Clustering":
    st.title("📚 Generate Content Topics")
    
    st.markdown("""
    This mode analyzes tagged keywords to:
    1. Group them into meaningful clusters with confidence scoring
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
            st.subheader("Clustering Settings")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Filter by A:Tag
                a_tags = sorted(df["A:Tag"].unique())
                selected_a_tags = st.multiselect("Filter by A:Tag (leave empty for all)", a_tags, key="a_tags_filter")
                
                # Number of clusters and outlier threshold
                num_clusters = st.slider("Max clusters per A:Tag group", 
                                        min_value=2, 
                                        max_value=10, 
                                        value=min(5, len(df) // 20),
                                        help="Maximum clusters to create within each A:Tag group",
                                        key="topic_num_clusters")
                
                outlier_threshold = st.slider("Outlier sensitivity", 
                                            min_value=1.0, 
                                            max_value=3.0, 
                                            value=1.5, 
                                            step=0.1,
                                            help="Lower values detect more outliers (1.0-3.0)",
                                            key="topic_outlier_threshold")
            
            with col2:
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
                
                # Confidence threshold
                confidence_threshold = st.slider(
                    "Confidence threshold", 
                    min_value=0.0, 
                    max_value=1.0, 
                    value=0.6,
                    step=0.05, 
                    help="Only include keywords with confidence scores above this threshold in topic generation",
                    key="topic_confidence_threshold"
                )
                
                # GPT option
                if use_gpt:
                    use_gpt_for_topics = st.checkbox("Use GPT for topic generation", value=True if api_key else False,
                                                  help="Generate more creative topic ideas with GPT", key="use_gpt_topics")
                    if use_gpt_for_topics and not api_key:
                        st.warning("API key required for GPT topic generation")
            
            # Run clustering
            if st.button("Generate Content Topics", key="generate_topics_button"):
                if use_gpt_for_topics and not api_key:
                    st.error("Please provide an OpenAI API key to use GPT for topic generation")
                    st.stop()
                
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
                    
                    # Apply two-stage clustering approach with outlier detection
                    progress_bar.progress(0.1)
                    st.text("Applying two-stage clustering by A:Tag with outlier detection...")
                    
                    df_clustered, cluster_info = two_stage_clustering(
                        df_filtered, 
                        num_clusters=num_clusters,
                        cluster_method=clustering_approach,
                        embedding_model=embedding_model,
                        outlier_threshold=outlier_threshold
                    )
                    
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
                    
                    progress_bar.progress(0.3)
                    
                    # Create visualization for semantic or hybrid clustering
                    viz_df = None
                    if clustering_approach in ["Semantic", "Hybrid"]:
                        st.text("Creating visualization...")
                        
                        # Use t-SNE on the whole dataset first
                        all_keywords = df_filtered["Keywords"].tolist()
                        
                        # Sample if dataset is large
                        max_viz_points = 1000
                        if len(all_keywords) > max_viz_points:
                            viz_indices = np.random.choice(len(all_keywords), max_viz_points, replace=False)
                            viz_keywords = [all_keywords[i] for i in viz_indices]
                            viz_clusters = df_filtered.iloc[viz_indices]["Cluster"].values
                            viz_a_tags = df_filtered.iloc[viz_indices]["A:Tag"].values
                            viz_labels = df_filtered.iloc[viz_indices]["Cluster_Label"].values
                            viz_confidence = df_filtered.iloc[viz_indices]["Cluster_Confidence"].values
                            viz_outliers = df_filtered.iloc[viz_indices]["Is_Outlier"].values
                            
                            # Generate embeddings for the sampled keywords
                            viz_embeddings = embedding_model.encode(viz_keywords)
                        else:
                            viz_keywords = all_keywords
                            viz_clusters = df_filtered["Cluster"].values
                            viz_a_tags = df_filtered["A:Tag"].values
                            viz_labels = df_filtered["Cluster_Label"].values
                            viz_confidence = df_filtered["Cluster_Confidence"].values
                            viz_outliers = df_filtered["Is_Outlier"].values
                            
                            # Generate embeddings for all keywords
                            viz_embeddings = embedding_model.encode(viz_keywords)
                        
                        # Apply t-SNE
                        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(viz_embeddings) // 10))
                        embeddings_2d = tsne.fit_transform(viz_embeddings)
                        
                        # Create visualization dataframe
                        viz_df = pd.DataFrame({
                            "x": embeddings_2d[:, 0],
                            "y": embeddings_2d[:, 1],
                            "keyword": viz_keywords,
                            "cluster": viz_clusters,
                            "a_tag": viz_a_tags,
                            "cluster_label": viz_labels,
                            "confidence": viz_confidence,
                            "is_outlier": viz_outliers
                        })
                    
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
                        high_conf_df = cluster_df[cluster_df["Cluster_Confidence"] >= confidence_threshold]
                        
                        # Get the keywords from the cluster - use high confidence for topics
                        if not is_outlier_cluster and len(high_conf_df) >= 5:
                            # Use high confidence keywords for regular clusters
                            keywords_for_topics = high_conf_df["Keywords"].tolist()
                            high_confidence_keywords = keywords_for_topics
                        else:
                            # For outlier clusters or small clusters, use all keywords
                            keywords_for_topics = cluster_df["Keywords"].tolist()
                            high_confidence_keywords = cluster_df[cluster_df["Cluster_Confidence"] >= confidence_threshold]["Keywords"].tolist()
                        
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
                            if is_outlier_cluster:
                                # For miscellaneous clusters, use a special prompt
                                misc_prompt = f"""Generate content topic ideas for a group of miscellaneous {a_tag} keywords that didn't fit well in other clusters.
                                Sample keywords: {', '.join(sample_keywords[:10])}
                                
                                Create 2-3 broad topic ideas that could cover these diverse keywords."""
                                
                                try:
                                    response = openai.chat.completions.create(
                                        model="gpt-4o-mini",
                                        messages=[{"role": "user", "content": misc_prompt}],
                                        temperature=0.7,
                                        max_tokens=200
                                    )
                                    misc_topics = response.choices[0].message.content.strip().split("\n")
                                    topic_ideas = [t.strip("- 123.") for t in misc_topics if t.strip()][:3]
                                    expanded_ideas = [f"{t} (Reference Guide)" for t in topic_ideas]
                                    value_props = ["Provides helpful information on miscellaneous topics" for _ in topic_ideas]
                                    cluster_analysis = f"Miscellaneous {a_tag} keywords that didn't fit well in other clusters."
                                except Exception as e:
                                    topic_ideas = [f"Miscellaneous {a_tag} Topics"]
                                    expanded_ideas = [f"Miscellaneous {a_tag} Topics (Reference Guide)"]
                                    value_props = ["Covers various topics that didn't fit in other categories."]
                                    cluster_analysis = f"Miscellaneous {a_tag} keywords that didn't fit well in other clusters."
                            else:
                                # For regular clusters
                                topic_ideas, expanded_ideas, value_props = generate_gpt_topics(
                                    sample_keywords, [a_tag], top_b_tags, api_key, kw_freq
                                )
                                # Also get cluster insights
                                cluster_analysis = get_gpt_cluster_insights(high_conf_df if len(high_conf_df) >= 5 else cluster_df, api_key)
                        else:
                            # Basic topic generation
                            if is_outlier_cluster:
                                topic_ideas = [f"Miscellaneous {a_tag} Topics"]
                                expanded_ideas = [f"Miscellaneous {a_tag} Topics (Reference Guide)"]
                                value_props = ["Covers various topics that didn't fit in other categories."]
                                cluster_analysis = f"Miscellaneous {a_tag} keywords that didn't fit well in other clusters."
                            else:
                                topic_ideas, expanded_ideas, value_props = generate_basic_topics(
                                    sample_keywords, [a_tag], top_b_tags
                                )
                                # Basic cluster analysis
                                cluster_analysis = f"Cluster focused on {a_tag} with emphasis on {', '.join(top_b_tags[:2] if top_b_tags else ['general attributes'])}"
                        
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
                    st.session_state.viz_df = viz_df
                    st.session_state.topic_map = topic_map
                    st.session_state.cluster_info = cluster_info  # Store cluster info for visualization
                    st.session_state.cluster_descriptors = cluster_descriptors  # Store cluster descriptors
                    st.session_state.confidence_threshold = confidence_threshold  # Store confidence threshold
    
    # Display results from session state if already processed
    if 'content_topics_processed' in st.session_state and st.session_state.content_topics_processed:
        df_filtered = st.session_state.df_filtered
        cluster_insights = st.session_state.cluster_insights
        topic_df = st.session_state.topic_df
        
        # Get confidence threshold from session state
        confidence_threshold = st.session_state.confidence_threshold if 'confidence_threshold' in st.session_state else 0.6
        
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
            
            # Show two-stage clustering visualization
            st.subheader("A:Tag Cluster Distribution")
            
            # Get cluster descriptors
            cluster_descriptors = st.session_state.cluster_descriptors if 'cluster_descriptors' in st.session_state else None
            
            # Visualization of A tag groups and their clusters
            fig = create_two_stage_visualization(df_filtered, cluster_info, cluster_descriptors)
            st.pyplot(fig)
            
            # Show confidence score summary
            st.subheader("Cluster Confidence Score Summary")
            
            # Create confidence stats per cluster
            conf_stats = []
            for cluster_id, info in cluster_info.items():
                cluster_df = df_filtered[df_filtered["Cluster"] == cluster_id]
                
                conf_stats.append({
                    "Cluster": cluster_descriptors.get(cluster_id, f"Cluster {cluster_id}") if cluster_descriptors else f"Cluster {cluster_id}",
                    "Is Miscellaneous": "Yes" if info.get("is_outlier_cluster", False) else "No",
                    "Keywords": len(cluster_df),
                    "High Confidence Keywords": len(cluster_df[cluster_df["Cluster_Confidence"] >= confidence_threshold]),
                    "Avg Confidence": f"{cluster_df['Cluster_Confidence'].mean():.2f}",
                    "Min Confidence": f"{cluster_df['Cluster_Confidence'].min():.2f}",
                    "Max Confidence": f"{cluster_df['Cluster_Confidence'].max():.2f}"
                })
            
            conf_df = pd.DataFrame(conf_stats)
            st.dataframe(conf_df)
            
            # Show visualization if available
            if st.session_state.viz_df is not None:
                st.subheader("Keyword Cluster Visualization")
                
                viz_df = st.session_state.viz_df
                
                # Create plot with confidence-based sizing and outlier marking
                fig, ax = plt.subplots(figsize=(12, 10))
                
                # Create separate scatter plots for regular and outlier points
                regular_points = viz_df[~viz_df["is_outlier"]]
                outlier_points = viz_df[viz_df["is_outlier"]]
                
                # Plot regular points with confidence-based sizing
                scatter_reg = ax.scatter(
                    regular_points["x"], regular_points["y"], 
                    c=regular_points["cluster"], 
                    cmap="tab20", 
                    alpha=0.7,
                    s=regular_points["confidence"] * 100,  # Size based on confidence
                    edgecolors='none'
                )
                
                # Plot outliers with distinctive styling
                if len(outlier_points) > 0:
                    scatter_out = ax.scatter(
                        outlier_points["x"], outlier_points["y"], 
                        c=outlier_points["cluster"],
                        cmap="tab20",
                        alpha=0.4,
                        s=30,
                        marker='x',
                        edgecolors='black',
                        linewidths=1
                    )
                
                # Add labels for some points
                high_conf_points = viz_df[viz_df["confidence"] > 0.9].sample(min(20, len(viz_df[viz_df["confidence"] > 0.9])))
                for i, row in high_conf_points.iterrows():
                    ax.annotate(row["keyword"], 
                                (row["x"], row["y"]),
                                xytext=(5, 5),
                                textcoords="offset points",
                                fontsize=8,
                                alpha=0.8)
                
                ax.set_title("Keyword Clusters (point size = confidence, x = outliers)")
                ax.set_xticks([])
                ax.set_yticks([])
                legend = ax.legend(*scatter_reg.legend_elements(), title="Clusters")
                ax.add_artist(legend)
                
                st.pyplot(fig)
            
            # Display cluster insights by A tag
            st.subheader("Content Topics by A:Tag")
            
            # Group clusters by A:Tag for easier exploration
            a_tags = sorted(set(info["a_tag"] for info in cluster_info.values()))
            
            for a_tag in a_tags:
                # Get clusters for this A:Tag
                a_clusters = [k for k, v in cluster_info.items() if v["a_tag"] == a_tag]
                
                with st.expander(f"A:Tag: {a_tag} ({len(a_clusters)} content topics)"):
                    for cluster_id in a_clusters:
                        # Get cluster info
                        info = cluster_info[cluster_id]
                        is_outlier_cluster = info.get("is_outlier_cluster", False)
                        
                        # Get topic name and label
                        topic_name = topic_map.get(cluster_id, f"Cluster {cluster_id}")
                        cluster_label = cluster_descriptors[cluster_id] if cluster_descriptors and cluster_id in cluster_descriptors else f"Cluster {cluster_id}"
                        
                        if is_outlier_cluster:
                            st.markdown(f"#### {cluster_label} (Miscellaneous)")
                            st.info("This is a miscellaneous group containing keywords that didn't fit well in other clusters.")
                        else:
                            st.markdown(f"#### {cluster_label}")
                        
                        st.markdown(f"**Size:** {info['size']} keywords")
                        
                        # Find the insight for this cluster
                        insight = next((i for i in cluster_insights if i["cluster_id"] == cluster_id), None)
                        if insight:
                            st.markdown(f"**Analysis:** {insight['analysis']}")
                            
                            # Show confidence stats
                            st.markdown(f"**High confidence keywords:** {insight['high_confidence_count']} of {insight['size']} " + 
                                       f"({insight['high_confidence_count']/insight['size']*100:.1f}%)" if insight['size'] > 0 else "0%")
                        
                        # Show B tags
                        if info["b_tags"]:
                            st.markdown(f"**Main B:Tags:** {', '.join(info['b_tags'])}")
                        
                        # Show sample keywords - use tabs to separate high confidence from all
                        kw_tabs = st.tabs(["High Confidence Keywords", "All Keywords"])
                        
                        with kw_tabs[0]:
                            # Show high confidence keywords
                            if insight and insight["high_confidence_keywords"]:
                                sample_size = min(5, len(insight["high_confidence_keywords"]))
                                sample_df = pd.DataFrame({"Keywords": insight["high_confidence_keywords"][:sample_size]})
                                st.dataframe(sample_df)
                            else:
                                st.info(f"No high confidence keywords (threshold = {confidence_threshold})")
                        
                        with kw_tabs[1]:
                            # Show all keywords
                            sample_size = min(5, len(info["keywords"]))
                            if sample_size > 0:
                                sample_df = pd.DataFrame({"Keywords": info["keywords"][:sample_size]})
                                st.dataframe(sample_df)
                        
                        # Show matching topics
                        matching_topics = [row for idx, row in topic_df.iterrows() if row["Cluster ID"] == cluster_id]
                        if matching_topics:
                            st.markdown("**Topic Ideas:**")
                            for topic in matching_topics[:3]:  # Show up to 3 topics
                                if topic['Is Miscellaneous'] == "Yes":
                                    st.markdown(f"- {topic['Topic']} ({topic['Format']}) - _Miscellaneous topics_")
                                else:
                                    st.markdown(f"- {topic['Topic']} ({topic['Format']})")
                        
                        st.markdown("---")
            
            # 5. Display cluster insights
            st.subheader("All Content Topic Clusters")
            
            for insight in cluster_insights:
                cluster_id = insight["cluster_id"]
                size = insight["size"]
                is_outlier = insight.get("is_outlier_cluster", False)
                
                # Create a title for the expander showing key info
                cluster_label = insight["cluster_label"] if "cluster_label" in insight else f"Cluster {cluster_id}"
                if is_outlier:
                    expander_title = f"{cluster_label} (Miscellaneous: {size} keywords)"
                elif insight["a_tags"]:
                    main_tag = insight["a_tags"][0]
                    expander_title = f"{cluster_label}: {main_tag.title()} ({size} keywords, {insight['high_confidence_count']} high confidence)"
                else:
                    expander_title = f"{cluster_label} ({size} keywords, {insight['high_confidence_count']} high confidence)"
                    
                with st.expander(expander_title):
                    # Show confidence stats
                    conf_ratio = insight['high_confidence_count']/insight['size']*100 if insight['size'] > 0 else 0
                    quality_indicator = "🟢 High" if conf_ratio >= 80 else ("🟡 Medium" if conf_ratio >= 50 else "🔴 Low")
                    
                    st.markdown(f"**Cluster Quality:** {quality_indicator} ({conf_ratio:.1f}% high confidence keywords)")
                    
                    # Show cluster analysis
                    st.markdown(f"**Cluster Analysis:** {insight['analysis']}")
                    
                    # Show tag information
                    if insight['a_tags']:
                        st.markdown(f"**Main Tags:** {', '.join(insight['a_tags'])}")
                    if insight['b_tags']:
                        st.markdown(f"**Secondary Tags:** {', '.join(insight['b_tags'])}")
                    
                    # Show topic phrases
                    if insight['topic_phrases']:
                        st.markdown("**Key Phrases:** " + ", ".join(insight['topic_phrases']))
                    
                    # Content topic suggestions
                    if insight['topic_ideas']:
                        st.markdown("### Suggested Content Topics")
                        
                        # Create a more visual representation of topics
                        for i, (idea, value) in enumerate(zip(insight['expanded_ideas'], insight['value_props'])):
                            st.markdown(f"**Topic {i+1}: {idea}**")
                            st.markdown(f"_{value}_")
                            st.markdown("---")
                    
                    # Sample keywords with confidence tabs
                    st.markdown("### Sample Keywords")
                    
                    # Use tabs to separate high confidence vs all keywords
                    kw_tabs = st.tabs(["High Confidence Keywords", "All Keywords"])
                    
                    with kw_tabs[0]:
                        # Show high confidence keywords
                        high_conf_sample = min(10, len(insight['high_confidence_keywords']))
                        if high_conf_sample > 0:
                            sample_df = pd.DataFrame({"Keywords": insight['high_confidence_keywords'][:high_conf_sample]})
                            st.table(sample_df)
                        else:
                            st.info(f"No keywords with confidence ≥ {confidence_threshold}")
                    
                    with kw_tabs[1]:
                        # Show all keywords
                        sample_size = min(10, len(insight['keywords']))
                        sample_df = pd.DataFrame({"Keywords": insight['keywords'][:sample_size]})
                        st.table(sample_df)
            
            # Add export functionality for detailed cluster observations
            st.subheader("Export Detailed Cluster Analysis")
    
            if st.button("Generate Detailed Cluster Report", key="gen_cluster_report"):
                with st.spinner("Creating reports in different formats..."):
                    # Create a formatted report of all cluster insights
                    report_content = "# KEYWORD CLUSTER ANALYSIS REPORT\n\n"
                    
                    for insight in cluster_insights:
                        cluster_id = insight["cluster_id"]
                        size = insight["size"]
                        is_outlier = insight.get("is_outlier_cluster", False)
                        
                        # Format cluster header with descriptive label
                        cluster_label = insight["cluster_label"] if "cluster_label" in insight else f"Cluster {cluster_id}"
                        
                        if is_outlier:
                            report_content += f"## {cluster_label} (Miscellaneous) ({size} keywords)\n\n"
                            report_content += f"_Note: This cluster contains keywords that didn't closely match other clusters._\n\n"
                        elif insight["a_tags"]:
                            main_tag = insight["a_tags"][0]
                            report_content += f"## {cluster_label}: {main_tag.title()} ({size} keywords)\n\n"
                        else:
                            report_content += f"## {cluster_label} ({size} keywords)\n\n"
                        
                        # Add confidence stats
                        conf_ratio = insight['high_confidence_count']/insight['size']*100 if insight['size'] > 0 else 0
                        report_content += f"**Cluster Quality:** {conf_ratio:.1f}% high confidence keywords\n\n"
                        
                        # Add cluster analysis
                        report_content += f"**Cluster Analysis:** {insight['analysis']}\n\n"
                        
                        # Add tag information
                        if insight['a_tags']:
                            report_content += f"**Main Tags:** {', '.join(insight['a_tags'])}\n\n"
                        if insight['b_tags']:
                            report_content += f"**Secondary Tags:** {', '.join(insight['b_tags'])}\n\n"
                        
                        # Add topic phrases
                        if insight['topic_phrases']:
                            report_content += "**Key Phrases:** " + ", ".join(insight['topic_phrases']) + "\n\n"
                        
                        # Add content topic suggestions
                        if insight['topic_ideas']:
                            report_content += "### Suggested Content Topics\n\n"
                            
                            for i, (idea, value) in enumerate(zip(insight['expanded_ideas'], insight['value_props'])):
                                report_content += f"**Topic {i+1}: {idea}**\n\n"
                                report_content += f"_{value}_\n\n"
                                report_content += "---\n\n"
                        
                        # Add sample keywords (high confidence only for report clarity)
                        report_content += "### High Confidence Sample Keywords\n\n"
                        sample_size = min(20, len(insight['high_confidence_keywords']))
                        if sample_size > 0:
                            for i, kw in enumerate(insight['high_confidence_keywords'][:sample_size]):
                                report_content += f"{i+1}. {kw}\n"
                        else:
                            report_content += "No high confidence keywords in this cluster.\n"
                        
                        report_content += "\n\n---\n\n"
                    
                    # Generate Word document if available
                    word_doc_ready = False
                    if DOCX_AVAILABLE:
                        try:
                            word_buffer = create_cluster_analysis_word_doc(cluster_insights)
                            word_doc_ready = True
                        except Exception as e:
                            st.error(f"Error generating Word document: {e}")
                    
                    # Also create a CSV version with structured data
                    detailed_rows = []
                    for insight in cluster_insights:
                        # Basic cluster info
                        cluster_label = insight["cluster_label"] if "cluster_label" in insight else f"Cluster {insight['cluster_id']}"
                        is_outlier = insight.get("is_outlier_cluster", False)
                        
                        row = {
                            "cluster_id": insight["cluster_id"],
                            "cluster_label": cluster_label,
                            "is_miscellaneous": "Yes" if is_outlier else "No",
                            "size": insight["size"],
                            "high_confidence_count": insight["high_confidence_count"],
                            "confidence_ratio": f"{insight['high_confidence_count']/insight['size']*100:.1f}%" if insight['size'] > 0 else "0%",
                            "analysis": insight["analysis"],
                            "main_tags": ", ".join(insight["a_tags"]) if insight["a_tags"] else "",
                            "secondary_tags": ", ".join(insight["b_tags"]) if insight["b_tags"] else "",
                            "key_phrases": ", ".join(insight["topic_phrases"]) if insight["topic_phrases"] else "",
                            "high_confidence_keywords": ", ".join(insight["high_confidence_keywords"][:10]) if insight["high_confidence_keywords"] else ""
                        }
                        
                        # Add topics (up to 5)
                        for i, topic in enumerate(insight["topic_ideas"][:5]):
                            row[f"topic_{i+1}"] = topic
                            if i < len(insight["value_props"]):
                                row[f"value_{i+1}"] = insight["value_props"][i]
                        
                        detailed_rows.append(row)
                
                # Display download options in columns
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.download_button(
                        "📄 Download as Markdown",
                        report_content,
                        "cluster_analysis_report.md",
                        "text/markdown",
                        key="download_cluster_report_md"
                    )
                
                with col2:
                    if DOCX_AVAILABLE and word_doc_ready:
                        st.download_button(
                            "📝 Download as Word Document",
                            word_buffer,
                            "cluster_analysis_report.docx",
                            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                            key="download_cluster_report_docx"
                        )
                    else:
                        st.info("Word export requires python-docx library")
                
                with col3:
                    if detailed_rows:
                        detailed_df = pd.DataFrame(detailed_rows)
                        csv_detailed = detailed_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            "📊 Download as CSV",
                            csv_detailed,
                            "cluster_analysis_data.csv",
                            "text/csv",
                            key="download_cluster_data_csv"
                        )
            
            # Create download buttons with unique keys
            st.subheader("Export Results")
            
            csv_result = df_filtered.to_csv(index=False).encode('utf-8')
            st.download_button(
                "Download Clustered Keywords CSV",
                csv_result,
                "content_topics.csv",
                "text/csv",
                key="download_clustered_keywords"
            )
            
            # Also offer a high confidence version
            high_conf_df = df_filtered[df_filtered["Cluster_Confidence"] >= confidence_threshold]
            high_conf_csv = high_conf_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "Download High Confidence Keywords Only",
                high_conf_csv,
                "high_confidence_keywords.csv",
                "text/csv",
                key="download_high_conf_keywords"
            )
            
            # Show topic summary and download with unique key
            if topic_df is not None:
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
                
                # Keywords Grouped by Content Topic section
                st.subheader("Keywords Grouped by Content Topic")
                
                # Display keywords grouped by Content Topic
                for topic_id, topic_name in topic_map.items():
                    cluster_df = df_filtered[df_filtered["Cluster"] == topic_id]
                    keyword_count = len(cluster_df)
                    high_conf_count = len(cluster_df[cluster_df["Cluster_Confidence"] >= confidence_threshold])
                    
                    with st.expander(f"{topic_name} ({keyword_count} keywords, {high_conf_count} high confidence)"):
                        # Use tabs to separate high confidence vs all keywords
                        kw_tabs = st.tabs(["High Confidence Keywords", "All Keywords"])
                        
                        with kw_tabs[0]:
                            # Show high confidence keywords
                            high_conf_kws = cluster_df[cluster_df["Cluster_Confidence"] >= confidence_threshold]
                            if len(high_conf_kws) > 0:
                                # Create a sample for display (limit to 20 for cleaner UI)
                                sample_size = min(20, len(high_conf_kws))
                                sample_df = high_conf_kws[["Keywords", "A:Tag", "B:Tag", "Cluster_Confidence"]].head(sample_size)
                                sample_df["Cluster_Confidence"] = sample_df["Cluster_Confidence"].apply(lambda x: f"{x:.2f}")
                                st.dataframe(sample_df)
                                
                                if len(high_conf_kws) > sample_size:
                                    st.info(f"Showing {sample_size} of {len(high_conf_kws)} high confidence keywords. Download full mapping below.")
                            else:
                                st.info(f"No high confidence keywords (threshold = {confidence_threshold})")
                        
                        with kw_tabs[1]:
                            # Show all keywords
                            if not cluster_df.empty:
                                # Create a sample for display (limit to 20 for cleaner UI)
                                sample_size = min(20, len(cluster_df))
                                sample_df = cluster_df[["Keywords", "A:Tag", "B:Tag", "Cluster_Confidence", "Is_Outlier"]].head(sample_size)
                                sample_df["Cluster_Confidence"] = sample_df["Cluster_Confidence"].apply(lambda x: f"{x:.2f}")
                                st.dataframe(sample_df)
                                
                                if len(cluster_df) > sample_size:
                                    st.info(f"Showing {sample_size} of {len(cluster_df)} keywords. Download full mapping below.")
                
                # Download options for keyword-topic mapping
                st.subheader("Export Keyword-Topic Mapping")
                col1, col2 = st.columns(2)
            
                with col1:
                    # Option 1: Download the full dataframe with topics assigned
                    csv_with_topics = df_filtered.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "Download Full Dataset with Topics",
                        csv_with_topics,
                        "keywords_with_topics.csv",
                        "text/csv",
                        key="download_full_with_topics"
                    )
            
                with col2:
                    # Option 2: Download a simplified topic-keyword mapping
                    simple_mapping = []
                    for _, row in df_filtered.iterrows():
                        simple_mapping.append({
                            "Content_Topic": row["Content_Topic"],
                            "Cluster_Label": row["Cluster_Label"] if "Cluster_Label" in row else f"Cluster {row['Cluster']}",
                            "Keyword": row["Keywords"],
                            "A:Tag": row["A:Tag"],
                            "B:Tag": row["B:Tag"],
                            "Confidence": row["Cluster_Confidence"],
                            "Quality": row["Content_Quality"],
                            "Is_Outlier": "Yes" if row["Is_Outlier"] else "No"
                        })
                    
                    simple_mapping_df = pd.DataFrame(simple_mapping)
                    csv_simple_mapping = simple_mapping_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "Download Topic-Keyword Mapping",
                        csv_simple_mapping,
                        "topic_keyword_map.csv",
                        "text/csv",
                        key="download_topic_mapping"
                    )
                
                # Provide an overall content strategy if using GPT
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
