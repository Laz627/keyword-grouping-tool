import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import nltk
from nltk.stem.snowball import SnowballStemmer
import io
from collections import Counter

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

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

# Function to extract noun phrases
def extract_noun_phrases(text):
    words = nltk.word_tokenize(text)
    tagged = nltk.pos_tag(words)
    noun_phrases = []
    current_phrase = []
    for word, tag in tagged:
        if tag.startswith('NN'):
            current_phrase.append(word)
        elif current_phrase:
            noun_phrases.append(' '.join(current_phrase))
            current_phrase = []
    if current_phrase:
        noun_phrases.append(' '.join(current_phrase))
    return noun_phrases

# Function to get cluster name
def get_cluster_name(cluster_keywords, tfidf_vectorizer, tfidf_matrix, cluster_idx):
    # Get the centroid of the cluster
    centroid = tfidf_matrix[cluster_idx].mean(axis=0).A1
    
    # Get the top 10 terms for this cluster by TF-IDF score
    top_term_indices = centroid.argsort()[-10:][::-1]
    top_terms = [tfidf_vectorizer.get_feature_names_out()[i] for i in top_term_indices]
    
    # Extract noun phrases from the cluster keywords
    all_noun_phrases = []
    for keyword in cluster_keywords:
        all_noun_phrases.extend(extract_noun_phrases(keyword.lower()))
    
    # Count the noun phrases
    phrase_counts = Counter(all_noun_phrases)
    
    # Combine top TF-IDF terms and frequent noun phrases, ensuring they appear in original keywords
    combined_terms = []
    for term in top_terms + [phrase for phrase, _ in phrase_counts.most_common(10)]:
        if any(term.lower() in keyword.lower() for keyword in cluster_keywords):
            combined_terms.append(term)
        if len(combined_terms) == 3:
            break
    
    # Create cluster name
    cluster_name = ' '.join(combined_terms)
    return cluster_name

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

            # ... (previous code remains the same)

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
        
        # Generate cluster names
        cluster_names = []
        for cluster in range(num_clusters):
            cluster_keywords = result[result['Cluster'] == cluster]['Keyword'].tolist()
            cluster_name = get_cluster_name(cluster_keywords, tfidf_vectorizer, tfidf_matrix, cluster)
            cluster_names.append({'Cluster': cluster, 'Cluster Name': cluster_name})
        
        cluster_names_df = pd.DataFrame(cluster_names)
        
        final_df = pd.merge(result, cluster_names_df, on='Cluster', how='left')
        
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

# ... (rest of the code remains the same)

# To run this app, save the script and run streamlit run script_name.py
