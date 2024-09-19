import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
import nltk
from nltk.stem import PorterStemmer
import io
import re

# Download necessary NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Initialize stemmer and stopwords
stemmer = PorterStemmer()
stop_words = set(nltk.corpus.stopwords.words('english'))

# Initialize the embedding model (MiniLM or another lightweight model for embeddings)
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  # Adjust as needed

def preprocess_text(text):
    # Convert to lowercase and tokenize
    words = nltk.word_tokenize(text.lower())
    # Remove stopwords and stem
    return ' '.join([stemmer.stem(word) for word in words if word.isalnum() and word not in stop_words])

def get_cluster_name(cluster_keywords, cluster_embedding):
    """
    Find the most representative keyword from the cluster based on cosine similarity
    between each keyword's embedding and the cluster centroid embedding.
    """
    # Generate embeddings for each keyword in the cluster
    keyword_embeddings = embedding_model.encode(cluster_keywords)
    
    # Calculate cosine similarities between the centroid and each keyword
    similarities = np.dot(keyword_embeddings, cluster_embedding) / (
        np.linalg.norm(keyword_embeddings, axis=1) * np.linalg.norm(cluster_embedding)
    )
    
    # Find the keyword with the highest similarity to the cluster centroid
    best_keyword = cluster_keywords[np.argmax(similarities)]
    return best_keyword

# Title and Instructions
st.title("Keyword Clustering Tool with Embeddings")
st.markdown("""
    **Instructions:**
    1. Upload a CSV file with your keywords and optional fields (Search Volume, CPC, Ranked Position, URL).
    2. Select the number of clusters using the slider before initiating the clustering process.
    3. The tool will classify and cluster the keywords using semantic embeddings and generate a downloadable CSV file with the results.
""")

# Template download
def generate_template():
    template = pd.DataFrame({
        "Keywords": ["Example keyword 1", "Example keyword 2"],
        "Search Volume": [1000, 500],
        "CPC": [0.5, 0.7],
        "Ranked Position": [1, 2],
        "URL": ["http://example.com/1", "http://example.com/2"]
    })
    return template.to_csv(index=False).encode('utf-8')

st.download_button(
    label="Download Template CSV",
    data=generate_template(),
    file_name="keyword_template.csv",
    mime="text/csv"
)

# File upload
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

# Initialize session state for storing embeddings and reduced embeddings
if 'embeddings' not in st.session_state:
    st.session_state['embeddings'] = None
    st.session_state['reduced_embeddings'] = None
    st.session_state['df'] = None

# User input for number of clusters
num_clusters = st.slider("Select Number of Clusters", min_value=2, max_value=150, value=10, step=1)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    required_columns = ["Keywords"]
    optional_columns = ["Search Volume", "CPC", "Ranked Position", "URL"]
    
    if "Keywords" not in df.columns:
        st.error("CSV must contain 'Keywords' column.")
    else:
        # Ensure optional columns are present and fill NaN with empty strings
        for col in optional_columns:
            if col not in df.columns:
                df[col] = ''
            else:
                df[col] = df[col].fillna('')
        
        df['Keywords'] = df['Keywords'].astype(str)

        # Preprocess keywords
        df['Processed_Keywords'] = df['Keywords'].apply(preprocess_text)
        st.session_state['df'] = df

if st.session_state['df'] is not None and st.button("Classify and Cluster Keywords"):
    df = st.session_state['df']

    # Step 1: Generate embeddings for the processed keywords
    st.info("Generating embeddings, this might take a few minutes...")
    embeddings = embedding_model.encode(df['Processed_Keywords'].tolist(), show_progress_bar=True)
    st.session_state['embeddings'] = np.array(embeddings)
    st.success("Embeddings generated successfully!")

    # Step 2: Reduce dimensions before clustering using PCA
    st.info("Reducing dimensionality with PCA to speed up clustering...")
    pca = PCA(n_components=50)  # Adjust number of components as needed
    reduced_embeddings = pca.fit_transform(st.session_state['embeddings'])
    st.session_state['reduced_embeddings'] = reduced_embeddings
    st.success("Dimensionality reduction completed!")

    # Step 3: Perform clustering
    with st.spinner("Clustering keywords..."):
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        df['Cluster'] = kmeans.fit_predict(st.session_state['reduced_embeddings'])
    
    st.success("Clustering completed!")

    # Step 4: Generate cluster names using the cluster centroids
    st.info("Generating cluster names, this might take a moment...")
    cluster_names = []
    progress_bar = st.progress(0)
    for cluster in range(num_clusters):
        cluster_keywords = df[df['Cluster'] == cluster]['Keywords'].tolist()
        cluster_embedding = kmeans.cluster_centers_[cluster]
        cluster_name = get_cluster_name(cluster_keywords, cluster_embedding)
        cluster_names.append({'Cluster': cluster, 'Cluster Name': cluster_name})
        progress_bar.progress((cluster + 1) / num_clusters)
    
    cluster_names_df = pd.DataFrame(cluster_names)
    
    # Merge cluster names with the main dataframe
    final_df = pd.merge(df, cluster_names_df, on='Cluster', how='left')
    
    # Select and order the final columns
    final_columns = ['Keywords', 'Search Volume', 'CPC', 'Ranked Position', 'URL', 'Cluster', 'Cluster Name']
    final_df = final_df[final_columns]

    # Step 5: Prepare CSV for download
    st.info("Preparing the output file...")
    output = io.BytesIO()
    final_df.to_csv(output, index=False)
    output.seek(0)
    
    st.download_button(
        label="Download Clustered Keywords CSV",
        data=output,
        file_name="clustered_keywords.csv",
        mime="text/csv"
    )

    st.dataframe(final_df)
