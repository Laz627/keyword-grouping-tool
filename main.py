import streamlit as st
import pandas as pd
from keybert import KeyBERT
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer

# Initialize the embedding model and KeyBERT model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
kw_model = KeyBERT()

st.title("Keyword Clustering and Classification with KeyBERT")
st.markdown("Upload a CSV with a 'Keywords' column. The tool will categorize based on semantic similarity, optionally using a seed keyword for context.")

# Optional input for seed keyword
seed_keyword = st.text_input("Enter Seed Keyword (Optional)", value="")

uploaded_file = st.file_uploader("Upload your CSV file with a 'Keywords' column", type=["csv"])

# Slider for number of clusters
num_clusters = st.slider("Select Number of Clusters", min_value=2, max_value=50, value=10, step=1)

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    if "Keywords" not in df.columns:
        st.error("CSV must contain a 'Keywords' column.")
    else:
        # Generate embeddings for clustering
        embeddings = embedding_model.encode(df['Keywords'].tolist())
        
        # Perform clustering
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        df['Cluster'] = kmeans.fit_predict(embeddings)

        # Extract cluster labels using KeyBERT for unigram, bigram, and trigram
        cluster_labels = []
        for cluster_num in range(num_clusters):
            cluster_keywords = df[df['Cluster'] == cluster_num]['Keywords'].tolist()
            
            # Combine cluster keywords and add seed keyword if provided
            combined_text = " ".join(cluster_keywords)
            if seed_keyword:  # Only add seed keyword if provided
                combined_text += " " + seed_keyword
            
            # Generate labels for unigram, bigram, and trigram
            unigrams = kw_model.extract_keywords(combined_text, keyphrase_ngram_range=(1, 1), stop_words='english', top_n=1)
            bigrams = kw_model.extract_keywords(combined_text, keyphrase_ngram_range=(2, 2), stop_words='english', top_n=1)
            trigrams = kw_model.extract_keywords(combined_text, keyphrase_ngram_range=(3, 3), stop_words='english', top_n=1)

            cluster_labels.append({
                'Cluster': cluster_num,
                'Unigram Label': unigrams[0][0] if unigrams else "N/A",
                'Bigram Label': bigrams[0][0] if bigrams else "N/A",
                'Trigram Label': trigrams[0][0] if trigrams else "N/A"
            })

        cluster_labels_df = pd.DataFrame(cluster_labels)
        df = pd.merge(df, cluster_labels_df, on="Cluster", how="left")

        # Show DataFrame with cluster names
        st.write("Clustered Keywords with Labels:")
        st.dataframe(df)
        
        # Option to download the labeled clusters
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(label="Download Clustered Keywords CSV", data=csv, file_name="clustered_keywords.csv", mime="text/csv")
