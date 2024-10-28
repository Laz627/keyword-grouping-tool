import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
import nltk
from nltk.stem import PorterStemmer
from scipy.spatial.distance import cosine
import openai
import io

# Download necessary NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Initialize stemmer and stopwords
stemmer = PorterStemmer()
stop_words = set(nltk.corpus.stopwords.words('english'))

# Initialize the embedding model (MiniLM for speed)
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

def preprocess_text(text):
    words = nltk.word_tokenize(text.lower())
    return ' '.join([stemmer.stem(word) for word in words if word.isalnum() and word not in stop_words])

def average_embedding(cluster_keywords, pca):
    keyword_embeddings = embedding_model.encode(cluster_keywords)
    reduced_embeddings = pca.transform(keyword_embeddings)
    avg_embedding = np.mean(reduced_embeddings, axis=0)
    return avg_embedding, reduced_embeddings

def find_closest_keyword(cluster_keywords, avg_embedding, reduced_embeddings):
    closest_keyword = cluster_keywords[0]
    closest_distance = float('inf')
    for keyword, embedding in zip(cluster_keywords, reduced_embeddings):
        distance = cosine(avg_embedding, embedding)
        if distance < closest_distance and keyword != closest_keyword:
            closest_distance = distance
            closest_keyword = keyword
    return closest_keyword

def summarize_cluster_gpt4o(cluster_keywords, granularity, api_key):
    """
    Uses GPT-4o mini to generate a descriptive cluster name.
    """
    openai.api_key = api_key
    prompt = f"Create a {granularity} summary name for the following keywords, capturing the main theme: {', '.join(cluster_keywords)}"
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=50,
        temperature=0.5
    )
    return response.choices[0].message['content'].strip()

st.title("Enhanced Keyword Clustering Tool with GPT-4o Mini")
st.markdown("Upload a CSV with a 'Keywords' column. The tool will categorize based on semantic similarity.")

api_key = st.text_input("Enter OpenAI API Key", type="password")
granularity = st.selectbox("Select Cluster Name Granularity", ["short", "medium", "detailed"])

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
num_clusters = st.slider("Select Number of Clusters", min_value=2, max_value=50, value=10, step=1)

if uploaded_file is not None and api_key:
    df = pd.read_csv(uploaded_file)
    
    if "Keywords" not in df.columns:
        st.error("CSV must contain 'Keywords' column.")
    else:
        df['Processed_Keywords'] = df['Keywords'].apply(preprocess_text)

        embeddings = embedding_model.encode(df['Processed_Keywords'].tolist())
        embeddings = np.array(embeddings)

        pca = PCA(n_components=50)
        reduced_embeddings = pca.fit_transform(embeddings)

        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        df['Cluster'] = kmeans.fit_predict(reduced_embeddings)

        cluster_names = []
        for cluster in range(num_clusters):
            cluster_keywords = df[df['Cluster'] == cluster]['Keywords'].tolist()
            avg_embedding, reduced_cluster_embeddings = average_embedding(cluster_keywords, pca)
            initial_name = find_closest_keyword(cluster_keywords, avg_embedding, reduced_cluster_embeddings)
            refined_name = summarize_cluster_gpt4o(cluster_keywords, granularity, api_key)
            cluster_names.append({'Cluster': cluster, 'Cluster Name': refined_name})
        
        cluster_names_df = pd.DataFrame(cluster_names)
        final_df = pd.merge(df, cluster_names_df, on='Cluster', how='left')

        output = io.BytesIO()
        final_df.to_csv(output, index=False)
        output.seek(0)

        st.download_button(
            label="Download Categorized Keywords CSV",
            data=output,
            file_name="categorized_keywords.csv",
            mime="text/csv"
        )
        
        st.dataframe(final_df)
