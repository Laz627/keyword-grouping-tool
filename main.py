import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
import nltk
from nltk.stem import PorterStemmer
import io
from scipy.spatial.distance import cosine
import openai

# Download necessary NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Initialize stemmer and stopwords
stemmer = PorterStemmer()
stop_words = set(nltk.corpus.stopwords.words('english'))

# Initialize the embedding model (MiniLM for speed)
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Set your OpenAI API key
openai.api_key = "your_openai_api_key"  # Replace with your actual API key

def preprocess_text(text):
    # Convert to lowercase and tokenize
    words = nltk.word_tokenize(text.lower())
    # Remove stopwords and stem
    return ' '.join([stemmer.stem(word) for word in words if word.isalnum() and word not in stop_words])

def average_embedding(cluster_keywords, pca):
    """
    Calculate the average embedding for a list of keywords after applying PCA to reduce dimensions.
    """
    keyword_embeddings = embedding_model.encode(cluster_keywords)
    reduced_embeddings = pca.transform(keyword_embeddings)
    avg_embedding = np.mean(reduced_embeddings, axis=0)
    return avg_embedding, reduced_embeddings

def find_closest_keyword(cluster_keywords, avg_embedding, reduced_embeddings):
    """
    Find the keyword closest to the average embedding of the cluster.
    """
    closest_keyword = cluster_keywords[0]
    closest_distance = float('inf')
    for keyword, embedding in zip(cluster_keywords, reduced_embeddings):
        distance = cosine(avg_embedding, embedding)
        if distance < closest_distance:
            closest_distance = distance
            closest_keyword = keyword
    return closest_keyword

def refine_cluster_name(cluster_name, keywords):
    """
    Refine the cluster name using OpenAI model.
    """
    prompt = f"""
    The following cluster of keywords has been assigned the initial name '{cluster_name}'. Based on the keywords listed, suggest a more descriptive and relevant cluster name that accurately represents the keywords:
    
    Keywords: {', '.join(keywords)}
    
    Suggested Cluster Name:
    """
    response = openai.Completion.create(
        engine="text-davinci-003",  # You can also use GPT-4 if available
        prompt=prompt,
        max_tokens=50,
        temperature=0.7
    )
    return response.choices[0].text.strip()

# Title and Instructions
st.title("Keyword Clustering Tool with Enhanced Naming")
st.markdown("""
    **Instructions:**
    1. Upload a CSV file with your keywords and optional fields (Search Volume, CPC, Ranked Position, URL).
    2. Select the number of clusters using the slider before initiating the clustering process.
    3. The tool will classify and cluster the keywords using semantic embeddings and generate a downloadable CSV file with the results.
    4. OpenAI will refine the cluster names for better contextual relevance.
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

        if st.button("Classify and Cluster Keywords"):
            # Step 1: Generate embeddings for the processed keywords
            st.info("Generating embeddings, this might take a few minutes...")
            embeddings = embedding_model.encode(df['Processed_Keywords'].tolist(), show_progress_bar=True)
            embeddings = np.array(embeddings)
            st.success("Embeddings generated successfully!")

            # Step 2: Reduce dimensions before clustering using PCA
            st.info("Reducing dimensionality with PCA to speed up clustering...")
            pca = PCA(n_components=50)
            reduced_embeddings = pca.fit_transform(embeddings)
            st.success("Dimensionality reduction completed!")

            # Step 3: Perform clustering
            with st.spinner("Clustering keywords..."):
                kmeans = KMeans(n_clusters=num_clusters, random_state=42)
                df['Cluster'] = kmeans.fit_predict(reduced_embeddings)
            
            st.success("Clustering completed!")

            # Step 4: Generate initial cluster names using the average embedding approach
            st.info("Generating initial cluster names...")
            cluster_names = []
            progress_bar = st.progress(0)
            for cluster in range(num_clusters):
                cluster_keywords = df[df['Cluster'] == cluster]['Keywords'].tolist()
                avg_embedding, reduced_cluster_embeddings = average_embedding(cluster_keywords, pca)
                initial_name = find_closest_keyword(cluster_keywords, avg_embedding, reduced_cluster_embeddings)
                refined_name = refine_cluster_name(initial_name, cluster_keywords)  # Refine with OpenAI
                cluster_names.append({'Cluster': cluster, 'Cluster Name': refined_name})
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
