import streamlit as st
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import nltk
from nltk.stem.snowball import SnowballStemmer
import io
from collections import Counter

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Initialize stemmer and stopwords
snow_stemmer = SnowballStemmer(language='english')
stop_words = set(nltk.corpus.stopwords.words('english'))

# Function to stem and preprocess list of keywords
def preprocess_keywords(keywords):
    preprocessed_keywords = []
    for keyword in keywords:
        words = nltk.word_tokenize(keyword.lower())
        stem_words = [snow_stemmer.stem(word) for word in words if word.isalnum() and word not in stop_words]
        preprocessed_keywords.append(" ".join(stem_words))
    return preprocessed_keywords

# Title and Instructions
st.title("Free Keyword Clustering Tool")
st.markdown("""
    **Instructions:**
    1. Upload a CSV file with your keywords and optional fields (Search Volume, CPC, Ranked Position, URL).
    2. Adjust the number of clusters for categorization.
    3. The tool will classify and cluster the keywords, and generate a downloadable CSV file with the results.
    
    **Created by:** Brandon Lazovic
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

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    required_columns = ["Keywords"]
    optional_columns = ["Search Volume", "CPC", "Ranked Position", "URL"]
    
    if not all(col in df.columns for col in required_columns):
        st.error("CSV must contain 'Keywords' column.")
    else:
        # Ensure optional columns are present and fill NaN with empty strings
        for col in optional_columns:
            if col not in df.columns:
                df[col] = ''
            else:
                df[col] = df[col].fillna('')
        
        df['Keywords'] = df['Keywords'].astype(str)

        textlist = df['Keywords'].tolist()
        preprocessed_textlist = preprocess_keywords(textlist)
        
        # Check if there are enough unique terms for clustering
        if len(set(preprocessed_textlist)) < 2:
            st.error("Not enough unique terms to perform clustering. Please provide more diverse keywords.")
        else:
            # User input for number of clusters
            num_clusters = st.slider("Select Number of Clusters", min_value=2, max_value=50, value=10, step=1)

            if st.button("Classify and Cluster Keywords"):
                tfidf_vectorizer = TfidfVectorizer(max_df=0.85, max_features=10000, min_df=1, stop_words='english', use_idf=True, ngram_range=(1,3))
                try:
                    tfidf_matrix = tfidf_vectorizer.fit_transform(preprocessed_textlist)
                    
                    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
                    kmeans.fit(tfidf_matrix)
                    clusters = kmeans.labels_.tolist()
                    
                    cluster_df = pd.DataFrame(clusters, columns=['Cluster'])
                    keywords_df = pd.DataFrame(textlist, columns=['Keyword'])
                    result = pd.concat([cluster_df, keywords_df], axis=1)
                    
                    # Generate cluster names based on most common words
                    def get_cluster_name(cluster_keywords):
                        words = ' '.join(cluster_keywords).split()
                        word_counts = Counter(words)
                        top_words = [word for word, _ in word_counts.most_common(3)]
                        return ' '.join(top_words)

                    cluster_names = result.groupby('Cluster')['Keyword'].apply(get_cluster_name).reset_index()
                    cluster_names.columns = ['Cluster', 'Cluster Name']
                    
                    final_df = pd.merge(result, cluster_names, on='Cluster', how='left')
                    
                    # Merge back optional columns
                    final_df = pd.merge(final_df, df, left_on='Keyword', right_on='Keywords', how='left')
                    
                    # Define the final columns order
                    final_columns = ['Keyword', 'Search Volume', 'CPC', 'Ranked Position', 'URL', 'Cluster', 'Cluster Name']
                    
                    # Ensure all required columns are present
                    for col in final_columns:
                        if col not in final_df.columns:
                            final_df[col] = None
                    
                    # Select and order the final columns
                    final_df = final_df[final_columns]
                    
                    # Remove empty rows
                    final_df = final_df.dropna(subset=['Keyword']).reset_index(drop=True)
                    
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
                except ValueError as e:
                    st.error(f"An error occurred during clustering: {e}")
                    st.text("Make sure your data is properly formatted and contains enough unique terms to cluster.")
                except KeyError as e:
                    st.error(f"KeyError: {e}. Please ensure your CSV file has the correct columns.")

# To run this app, save the script and run streamlit run script_name.py
