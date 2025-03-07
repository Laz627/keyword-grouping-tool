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
from nltk import word_tokenize, pos_tag
from nltk.stem import WordNetLemmatizer
import openai

# Download NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

lemmatizer = WordNetLemmatizer()

# Load models on demand using session state
def get_models():
    """Load models on demand to avoid Streamlit/PyTorch conflicts"""
    if 'models_loaded' not in st.session_state:
        with st.spinner("Loading NLP models... (this might take a moment)"):
            try:
                from sentence_transformers import SentenceTransformer
                from sklearn.cluster import KMeans
                
                # Load models into session state
                st.session_state.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                st.session_state.kw_model = KeyBERT(model=st.session_state.embedding_model)
                st.session_state.models_loaded = True
            except Exception as e:
                st.error(f"Error loading models: {e}")
                st.session_state.models_loaded = False
                st.stop()
    
    return st.session_state.embedding_model, st.session_state.kw_model

# All your helper functions that use the models should now access them via get_models()
def extract_candidate_themes(keywords_list, top_n, progress_bar=None):
    """Extract candidate themes with dynamic model loading"""
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

# Continue with other function definitions, modified to use get_models() instead of global model variables
def classify_keyword_three(keyword, seed, omitted_list, user_a_tags):
    """Process a keyword string with dynamic model loading"""
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

# Define the rest of your helper functions
# [...]

# Main UI - add this to the bottom of the script
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
    
    # [Rest of the UI code for this mode]

elif mode == "Full Tagging":
    st.title("🏷️ Tag Your Keywords")
    
    # [Rest of the UI code for this mode]

elif mode == "Content Topic Clustering":
    st.title("📚 Generate Content Topics")
    
    # [Rest of the UI code for this mode]

###
### Main Application Function
###

def run_app():
    # Sidebar with application description and mode selection
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

    # Main content area - customize based on selected mode
    if mode == "Candidate Theme Extraction":
        st.title("🔍 Extract Keyword Themes")
        
        st.markdown("""
        This mode identifies common themes in your keywords and shows how they would be tagged.
        Useful for understanding what patterns exist in your keyword set.
        """)
        
        # File upload
        file = st.file_uploader("Upload your keyword file (CSV/Excel)", type=["csv", "xls", "xlsx"])
        
        # Settings in columns for better use of space
        col1, col2 = st.columns(2)
        with col1:
            nm = st.number_input("Process first N keywords (0 for all)", min_value=0, value=0)
            topn = st.number_input("Keyphrases per keyword", min_value=1, value=3)
        with col2:
            mfreq = st.number_input("Minimum frequency threshold", min_value=1, value=2)
            clust = st.number_input("Number of clusters (0 to skip)", min_value=0, value=0)
        
        # A-Tags input
        user_atags_str = st.text_input("Specify allowed A:Tags (comma-separated)", "door, window")
        user_a_tags = set(normalize_token(x.strip()) for x in user_atags_str.split(",") if x.strip())
        
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
            
                        st.subheader("Candidate Themes")
                        st.dataframe(cdf)
            
                        if clust > 0 and len(c_map) >= clust:
                            st.subheader("Theme Clusters")
                            with st.spinner("Clustering themes..."):
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
            
                        # Download button
                        st.download_button(
                            "Download Candidate Themes CSV", 
                            cdf.to_csv(index=False).encode('utf-8'), 
                            "candidate_themes.csv", 
                            "text/csv"
                        )
                        
                        # GPT-powered insights if enabled
                        if use_gpt and api_key:
                            st.subheader("💡 Theme Insights")
                            if st.button("Generate Theme Insights with GPT"):
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
                    else:
                        st.warning("No candidate themes meet the frequency threshold.")

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
            seed = st.text_input("Seed Keyword to Remove (optional)", "")
            omit_str = st.text_input("Phrases to Omit (comma-separated)", "")
        
        with col2:
            user_atags_str = st.text_input("Allowed A:Tags (comma-separated)", "door, window")
            do_realign = st.checkbox("Enable post-processing re-alignment", value=True,
                                   help="Ensures consistent tag placement based on frequency")
        
        # Optional: Upload an initial tagging rule CSV
        initial_rule_file = st.file_uploader("Initial Tagging Rule CSV (optional)", type=["csv"])
        use_initial_rule = st.checkbox("Use Initial Tagging Rule if available", value=False)
        
        file = st.file_uploader("Upload Keyword File (CSV/Excel)", type=["csv", "xls", "xlsx"])
        
        # Build the initial rule mapping if provided and requested.
        initial_rule_mapping = {}
        if use_initial_rule and initial_rule_file is not None:
            try:
                if initial_rule_file.name.endswith(".csv"):
                    rule_df = pd.read_csv(initial_rule_file)
                else:
                    rule_df = pd.read_excel(initial_rule_file)
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
            
                    st.subheader("Tagged Keywords")
                    st.dataframe(df[["Keywords", "A:Tag", "B:Tag", "C:Tag"]])
            
                    # Summary Report: Only A:Tag & B:Tag combination
                    df["A+B"] = df["A:Tag"] + " - " + df["B:Tag"]
                    summary_ab = df.groupby("A+B").size().reset_index(name="Count")
                    summary_ab = summary_ab.sort_values("Count", ascending=False)
                    
                    st.subheader("Tag Summary")
                    st.dataframe(summary_ab)
            
                    # Download buttons
                    col1, col2 = st.columns(2)
                    with col1:
                        st.download_button(
                            "Download Full Tagging CSV", 
                            df.to_csv(index=False).encode('utf-8'), 
                            "tagged_keywords.csv", 
                            "text/csv"
                        )
                    with col2:
                        st.download_button(
                            "Download Tag Summary CSV",
                            summary_ab.to_csv(index=False).encode('utf-8'),
                            "tag_summary.csv",
                            "text/csv"
                        )
                    
                    # GPT tag analysis if enabled
                    if use_gpt and api_key:
                        st.subheader("💡 Tag Analysis")
                        if st.button("Analyze Tag Patterns with GPT"):
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
        
        # File upload for tagged keywords
        file = st.file_uploader("Upload tagged keywords file (CSV)", type=["csv"],
                               help="This should be a file from the Full Tagging mode")
        
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
                selected_a_tags = st.multiselect("Filter by A:Tag (leave empty for all)", a_tags)
                
                # Number of clusters
                num_clusters = st.slider("Number of clusters", min_value=2, max_value=20, value=min(8, len(df) // 10),
                                        help="Number of content topic clusters to generate")
            
            with col2:
                # Clustering approach
                clustering_approach = st.radio(
                    "Clustering Approach",
                    ["Semantic (keyword meaning)", "Tag-based (A/B Tags)", "Hybrid (tags + meaning)"],
                    help="How to group your keywords into clusters"
                )
                
                # GPT option
                if use_gpt:
                    use_gpt_for_topics = st.checkbox("Use GPT for topic generation", value=True if api_key else False,
                                                  help="Generate more creative topic ideas with GPT")
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
            if st.button("Generate Content Topics"):
                if use_gpt_for_topics and not api_key:
                    st.error("Please provide an OpenAI API key to use GPT for topic generation")
                    st.stop()
                
                with st.spinner("Processing keywords and generating topics..."):
                    progress_bar = st.progress(0)
                    
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
                        
                        # Complete progress
                        progress_bar.progress(1.0)
                        progress_bar.empty()
                        
                        # Create scatter plot
                        st.subheader("Keyword Cluster Visualization")
                        
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
                    else:
                        # Complete progress for non-visualization approach
                        progress_bar.progress(1.0)
                        progress_bar.empty()
                    
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
                    
                    # 6. Create downloadable output
                    # Add cluster and topic info to the dataframe
                    topic_map = {}
                    for insight in cluster_insights:
                        cluster_id = insight["cluster_id"]
                        if insight["topic_ideas"]:
                            topic_map[cluster_id] = insight["topic_ideas"][0]
                        else:
                            topic_map[cluster_id] = f"Cluster {cluster_id}"
                    
                    df_filtered["Content_Topic"] = df_filtered["Cluster"].map(topic_map)
                    
                    # Create CSV for download
                    st.subheader("Export Results")
                    
                    csv_result = df_filtered.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "Download Clustered Keywords CSV",
                        csv_result,
                        "content_topics.csv",
                        "text/csv"
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
                                "Topic": idea,
                                "Format": insight['expanded_ideas'][i].replace(idea, "").strip("() ") if i < len(insight['expanded_ideas']) else "",
                                "Value": value,
                                "Cluster": insight['cluster_id'],
                                "Keywords": insight['size'],
                                "Main Tags": ", ".join(insight['a_tags'] + insight['b_tags'][:2] if insight['b_tags'] else insight['a_tags'])
                            })
                    
                    if topic_summary:
                        topic_df = pd.DataFrame(topic_summary)
                        st.subheader("Content Topic Ideas")
                        st.dataframe(topic_df)
                        
                        # Create CSV for download
                        csv_topics = topic_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            "Download Content Topics CSV",
                            csv_topics,
                            "topic_ideas.csv",
                            "text/csv"
                        )
                        
                        # Provide an overall content strategy if using GPT
                        if use_gpt_for_topics and api_key:
                            st.subheader("📋 Content Strategy Overview")
                            if st.button("Generate Content Strategy Recommendations"):
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
