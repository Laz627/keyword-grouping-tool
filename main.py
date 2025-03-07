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
                # Load models into session state
                st.session_state.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                st.session_state.kw_model = KeyBERT(model=st.session_state.embedding_model)
                st.session_state.models_loaded = True
            except Exception as e:
                st.error(f"Error loading models: {e}")
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
        if insight["a_tags"]:
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
        
        # Add keywords in a table
        doc.add_heading('Keywords', level=2)
        
        # Calculate number of columns and rows for the keyword table
        keywords = insight['keywords']
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
                km = KMeans(n_clusters=clust, random_state=42)
                labs = km.fit_predict(emb)
                
                # Create a dataframe for easier viewing
                cluster_df = pd.DataFrame({
                    "Theme": reps,
                    "Cluster": labs,
                    "Frequency": [c_map[rep] for rep in reps]
                })
                
                # Show clusters in tabs
                tabs = st.tabs([f"Cluster {i}" for i in range(clust)])
                for i, tab in enumerate(tabs):
                    with tab:
                        cluster_data = cluster_df[cluster_df["Cluster"] == i]
                        st.dataframe(cluster_data.sort_values("Frequency", ascending=False))

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

elif mode == "Content Topic Clustering":
    st.title("📚 Generate Content Topics")
    
    st.markdown("""
    This mode analyzes tagged keywords to:
    1. Group them into meaningful clusters
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
                
                # Number of clusters
                num_clusters = st.slider("Number of clusters", min_value=2, max_value=20, value=min(8, len(df) // 10),
                                        help="Number of content topic clusters to generate", key="num_clusters")
            
            with col2:
                # Clustering approach
                clustering_approach = st.radio(
                    "Clustering Approach",
                    ["Semantic (keyword meaning)", "Tag-based (A/B Tags)", "Hybrid (tags + meaning)"],
                    help="How to group your keywords into clusters",
                    key="clustering_approach"
                )
                
                # GPT option
                if use_gpt:
                    use_gpt_for_topics = st.checkbox("Use GPT for topic generation", value=True if api_key else False,
                                                  help="Generate more creative topic ideas with GPT", key="use_gpt_topics")
                    if use_gpt_for_topics and not api_key:
                        st.warning("API key required for GPT topic generation")
            
            # Apply filters
            if selected_a_tags:
                df_filtered = df[df["A:Tag"].isin(selected_a_tags)]
                if len(df_filtered) == 0:
                    st.warning("No keywords match the selected filters")
                    st.stop()
            else:
                df_filtered = df.copy()
            
            # Run clustering
            if st.button("Generate Content Topics", key="generate_topics_button"):
                if use_gpt_for_topics and not api_key:
                    st.error("Please provide an OpenAI API key to use GPT for topic generation")
                    st.stop()
                
                with st.spinner("Processing keywords and generating topics..."):
                    progress_bar = st.progress(0)
                    
                    # Get models
                    embedding_model, kw_model = get_models()
                    
                    # 1. Create feature vectors based on the chosen approach
                    if clustering_approach == "Tag-based (A/B Tags)":
                        # Create one-hot encoding of tags
                        a_dummies = pd.get_dummies(df_filtered["A:Tag"], prefix="A")
                        b_dummies = pd.get_dummies(df_filtered["B:Tag"], prefix="B")
                        
                        features = pd.concat([a_dummies, b_dummies], axis=1)
                        
                    elif clustering_approach == "Semantic (keyword meaning)":
                        # Use embeddings for semantic meaning
                        st.text("Generating keyword embeddings...")
                        keywords = df_filtered["Keywords"].tolist()
                        # Process in batches to avoid memory issues with large datasets
                        batch_size = 500
                        features = []
                        for i in range(0, len(keywords), batch_size):
                            batch = keywords[i:i+batch_size]
                            features.append(embedding_model.encode(batch))
                        features = np.vstack(features)
                        
                    else:  # Hybrid approach
                        # Combine tag information with semantic embeddings
                        st.text("Generating hybrid features...")
                        
                        # Create enriched keywords by combining with tags
                        enriched_keywords = df_filtered.apply(
                            lambda row: f"{row['Keywords']} {row['A:Tag']} {row['B:Tag']}",
                            axis=1
                        ).tolist()
                        
                        # Process in batches
                        batch_size = 500
                        features = []
                        for i in range(0, len(enriched_keywords), batch_size):
                            batch = enriched_keywords[i:i+batch_size]
                            features.append(embedding_model.encode(batch))
                        features = np.vstack(features)
                    
                    progress_bar.progress(0.3)
                    
                    # 2. Apply KMeans clustering
                    st.text("Clustering keywords...")
                    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
                    clusters = kmeans.fit_predict(features)
                    df_filtered["Cluster"] = clusters
                    
                    progress_bar.progress(0.5)
                    
                    # 3. Generate topic insights for each cluster
                    st.text("Generating topic insights...")
                    cluster_insights = []
                    
                    # Create a mapping of cluster IDs to topic names for later use
                    topic_map = {}
                    
                    for cluster_id in range(num_clusters):
                        cluster_df = df_filtered[df_filtered["Cluster"] == cluster_id]
                        
                        # Get the most common tags
                        top_a_tags = cluster_df["A:Tag"].value_counts().head(3).index.tolist()
                        top_b_tags = cluster_df["B:Tag"].value_counts().head(5).index.tolist()
                        
                        # Get keywords in cluster
                        keywords_in_cluster = cluster_df["Keywords"].tolist()
                        
                        # Sample keywords for analysis
                        sample_keywords = keywords_in_cluster[:min(30, len(keywords_in_cluster))]
                        
                        # Optional: Get frequency data 
                        kw_freq = None
                        if "Count" in cluster_df.columns:
                            kw_freq = dict(zip(cluster_df["Keywords"], cluster_df["Count"]))
                        
                        # Generate topics with GPT or basic approach
                        if use_gpt_for_topics and api_key:
                            topic_ideas, expanded_ideas, value_props = generate_gpt_topics(
                                sample_keywords, top_a_tags, top_b_tags, api_key, kw_freq
                            )
                            # Also get cluster insights
                            cluster_analysis = get_gpt_cluster_insights(cluster_df, api_key)
                        else:
                            topic_ideas, expanded_ideas, value_props = generate_basic_topics(
                                sample_keywords, top_a_tags, top_b_tags
                            )
                            # Basic cluster analysis
                            cluster_analysis = f"Cluster focused on {', '.join(top_a_tags)} with emphasis on {', '.join(top_b_tags[:2] if top_b_tags else ['general attributes'])}"
                        
                        # Store the primary topic name in the mapping
                        if topic_ideas:
                            topic_map[cluster_id] = topic_ideas[0]
                        else:
                            topic_map[cluster_id] = f"Cluster {cluster_id}"
                        
                        # Extract topic phrases for visualization
                        if keywords_in_cluster:
                            combined_text = " ".join(sample_keywords)
                            key_phrases = kw_model.extract_keywords(combined_text, keyphrase_ngram_range=(1, 3), top_n=5)
                            topic_phrases = [kp[0] for kp in key_phrases]
                        else:
                            topic_phrases = []
                        
                        cluster_insights.append({
                            "cluster_id": cluster_id,
                            "size": len(cluster_df),
                            "a_tags": top_a_tags,
                            "b_tags": top_b_tags,
                            "keywords": keywords_in_cluster,
                            "sample_keywords": sample_keywords,
                            "topic_phrases": topic_phrases,
                            "topic_ideas": topic_ideas,
                            "expanded_ideas": expanded_ideas,
                            "value_props": value_props,
                            "analysis": cluster_analysis
                        })
                        
                        # Update progress
                        progress_bar.progress(0.5 + (0.4 * (cluster_id + 1) / num_clusters))
                    
                    # 4. Create visualization
                    viz_df = None
                    if clustering_approach in ["Semantic (keyword meaning)", "Hybrid (tags + meaning)"]:
                        st.text("Creating visualization...")
                        # Use t-SNE for visualization
                        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(df_filtered) // 10))
                        
                        # Sample if dataset is large to speed up TSNE
                        max_viz_points = 1000
                        if len(df_filtered) > max_viz_points:
                            viz_indices = np.random.choice(len(df_filtered), max_viz_points, replace=False)
                            viz_features = features[viz_indices]
                            viz_clusters = clusters[viz_indices]
                            viz_keywords = df_filtered.iloc[viz_indices]["Keywords"].values
                            viz_a_tags = df_filtered.iloc[viz_indices]["A:Tag"].values
                        else:
                            viz_features = features
                            viz_clusters = clusters
                            viz_keywords = df_filtered["Keywords"].values
                            viz_a_tags = df_filtered["A:Tag"].values
                        
                        # Apply t-SNE
                        embeddings_2d = tsne.fit_transform(viz_features)
                        
                        # Create visualization dataframe
                        viz_df = pd.DataFrame({
                            "x": embeddings_2d[:, 0],
                            "y": embeddings_2d[:, 1],
                            "keyword": viz_keywords,
                            "cluster": viz_clusters,
                            "a_tag": viz_a_tags
                        })
                    
                    # Create topic summary table
                    topic_summary = []
                    for insight in cluster_insights:
                        for i, idea in enumerate(insight['topic_ideas']):
                            if i >= len(insight['value_props']):
                                value = ""
                            else:
                                value = insight['value_props'][i]
                                
                            topic_summary.append({
                                "Topic": idea,
                                "Format": insight['expanded_ideas'][i].replace(idea, "").strip("() ") if i < len(insight['expanded_ideas']) else "",
                                "Value": value,
                                "Cluster": insight['cluster_id'],
                                "Keywords": insight['size'],
                                "Main Tags": ", ".join(insight['a_tags'] + insight['b_tags'][:2] if insight['b_tags'] else insight['a_tags'])
                            })
                    
                    topic_df = pd.DataFrame(topic_summary) if topic_summary else None
                    
                    # Complete progress
                    progress_bar.progress(1.0)
                    progress_bar.empty()
                    
                    # Add topic mapping to the filtered dataframe
                    df_filtered["Content_Topic"] = df_filtered["Cluster"].map(topic_map)
                    
                    # Store in session state
                    st.session_state.content_topics_processed = True
                    st.session_state.df_filtered = df_filtered
                    st.session_state.cluster_insights = cluster_insights
                    st.session_state.topic_df = topic_df
                    st.session_state.viz_df = viz_df
                    st.session_state.topic_map = topic_map  # Store topic_map in session state
    
    # Display results from session state if already processed
    if 'content_topics_processed' in st.session_state and st.session_state.content_topics_processed:
        df_filtered = st.session_state.df_filtered
        cluster_insights = st.session_state.cluster_insights
        topic_df = st.session_state.topic_df
        
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
        
        # Show visualization if available
        if st.session_state.viz_df is not None:
            st.subheader("Keyword Cluster Visualization")
            
            viz_df = st.session_state.viz_df
            fig, ax = plt.subplots(figsize=(10, 8))
            scatter = ax.scatter(
                viz_df["x"], viz_df["y"], 
                c=viz_df["cluster"], 
                cmap="tab20", 
                alpha=0.7,
                s=50
            )
            ax.set_title("Keyword Clusters")
            ax.set_xticks([])
            ax.set_yticks([])
            legend = ax.legend(*scatter.legend_elements(), title="Clusters")
            ax.add_artist(legend)
            
            st.pyplot(fig)
        
        # 5. Display cluster insights
        st.subheader("Content Topic Clusters")
        
        for insight in cluster_insights:
            cluster_id = insight["cluster_id"]
            size = insight["size"]
            
            # Create a title for the expander showing key info
            if insight["a_tags"]:
                main_tag = insight["a_tags"][0]
                expander_title = f"Cluster {cluster_id}: {main_tag.title()} ({size} keywords)"
            else:
                expander_title = f"Cluster {cluster_id}: {size} keywords"
                
            with st.expander(expander_title):
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
                
                # Sample keywords in a table
                st.markdown("### Sample Keywords")
                sample_size = min(10, len(insight['keywords']))
                sample_df = pd.DataFrame({
                    "Keywords": insight['keywords'][:sample_size]
                })
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
                    
                    # Format cluster header
                    if insight["a_tags"]:
                        main_tag = insight["a_tags"][0]
                        report_content += f"## Cluster {cluster_id}: {main_tag.title()} ({size} keywords)\n\n"
                    else:
                        report_content += f"## Cluster {cluster_id}: {size} keywords\n\n"
                    
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
                    
                    # Add sample keywords
                    report_content += "### Sample Keywords\n\n"
                    sample_size = min(20, len(insight['keywords']))
                    for i, kw in enumerate(insight['keywords'][:sample_size]):
                        report_content += f"{i+1}. {kw}\n"
                    
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
                    row = {
                        "cluster_id": insight["cluster_id"],
                        "size": insight["size"],
                        "analysis": insight["analysis"],
                        "main_tags": ", ".join(insight["a_tags"]) if insight["a_tags"] else "",
                        "secondary_tags": ", ".join(insight["b_tags"]) if insight["b_tags"] else "",
                        "key_phrases": ", ".join(insight["topic_phrases"]) if insight["topic_phrases"] else "",
                        "sample_keywords": ", ".join(insight["keywords"][:10]) if insight["keywords"] else ""
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
                
                with st.expander(f"{topic_name} ({keyword_count} keywords)"):
                    # Show keywords in this topic
                    if not cluster_df.empty:
                        # Create a sample for display (limit to 20 for cleaner UI)
                        sample_size = min(20, len(cluster_df))
                        sample_df = cluster_df[["Keywords", "A:Tag", "B:Tag"]].head(sample_size)
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
                        "Keyword": row["Keywords"],
                        "A:Tag": row["A:Tag"],
                        "B:Tag": row["B:Tag"]
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
                            # Extract topic information for the prompt
                            top_topics = [
                                f"{row['Topic']} ({row['Format']}): {row['Keywords']} keywords" 
                                for _, row in topic_df.head(8).iterrows()
                            ]
                            
                            # Get the distribution of A tags
                            a_tag_counts = df_filtered["A:Tag"].value_counts()
                            a_tag_info = [f"{tag}: {count}" for tag, count in a_tag_counts.items()]
                            
                            openai.api_key = api_key
                            response = openai.chat.completions.create(
                                model="gpt-4o-mini",
                                messages=[{"role": "user", "content": f"""
                                As a content strategist, create a brief content plan based on these keyword clusters.
                                
                                GENERATED CONTENT TOPICS:
                                {', '.join(top_topics)}
                                
                                KEYWORD CATEGORY DISTRIBUTION:
                                {', '.join(a_tag_info)}
                                
                                Please provide:
                                1. Top 3 priority content topics and why they should be prioritized
                                2. How these topics could be connected in a content hub structure
                                3. A suggestion for seasonal or evergreen content based on these topics
                                
                                Keep recommendations specific and actionable.
                                """}],
                                temperature=0.5,
                                max_tokens=800
                            )
                            
                            strategy = response.choices[0].message.content
                            st.markdown(strategy)
                        except Exception as e:
                            st.error(f"Error generating content strategy: {e}")
