import streamlit as st
import pandas as pd
import re
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
import nltk
from nltk.stem.snowball import SnowballStemmer
import io

# Download necessary NLTK data
nltk.download('punkt')

# Initialize stemmer
snow_stemmer = SnowballStemmer(language='english')

# Function to stem list of keywords
def stemmList(list):
    stemmed_list = []
    for l in list:
        words = nltk.word_tokenize(l)
        stem_words = [snow_stemmer.stem(word) for word in words if word.isalnum()]
        stemmed_list.append(" ".join(stem_words))
    return stemmed_list

# Title and Instructions
st.title("Free Keyword Clustering Tool")
st.markdown("""
    **Instructions:**
    1. Upload a CSV file with your keywords and optional fields (Search Volume, CPC, Ranked Position, URL).
    2. Optionally, enter L1 and L2 classifications to help reduce "miscellaneous" keywords.
    3. Adjust the sensitivity for clustering.
    4. The tool will classify and cluster the keywords, and generate a downloadable CSV file with the results.
    
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
    if not all(col in df.columns for col in required_columns):
        st.error("CSV must contain 'Keywords' column.")
    else:
        # Ensure optional columns are present and fill NaN with empty strings
        for col in ["Search Volume", "CPC", "Ranked Position", "URL"]:
            if col not in df.columns:
                df[col] = ''
            else:
                df[col] = df[col].fillna('')
        
        df['Keywords'] = df['Keywords'].astype(str)

        # User input for L1 and L2 classifications
        l1_input = st.text_area("Enter L1 Classifications (comma-separated)", "")
        l2_input = st.text_area("Enter L2 Classifications (comma-separated)", "")

        l1_classifications = [cls.strip() for cls in l1_input.split(',') if cls.strip()]
        l2_classifications = [cls.strip() for cls in l2_input.split(',') if cls.strip()]
        
        if l1_classifications and l2_classifications:
            def classify_keywords(keyword):
                for l1 in l1_classifications:
                    if any(re.search(rf'\b{re.escape(l2)}\b', keyword, re.IGNORECASE) for l2 in l2_classifications):
                        return l1
                return None

            df['Classification'] = df['Keywords'].apply(classify_keywords)
            
            # Separate pre-classified and unclassified keywords
            classified_df = df.dropna(subset=['Classification'])
            unclassified_df = df[df['Classification'].isna()]
        else:
            df['Classification'] = None
            classified_df = df[[]]  # Empty DataFrame with the same columns
            unclassified_df = df
        
        textlist = unclassified_df['Keywords'].to_list()
        labellist = textlist
        textlist = stemmList(textlist)
        
        # Check if there are enough unique terms for clustering
        if len(set(textlist)) < 2:
            st.error("Not enough unique terms to perform clustering. Please provide more diverse keywords.")
        else:
            # User input for clustering sensitivity
            sensitivity = st.slider("Select Clustering Sensitivity", min_value=0.01, max_value=1.0, value=0.2, step=0.01)
            min_clustersize = st.number_input("Minimum Cluster Size", min_value=1, value=2)

            if st.button("Classify and Cluster Keywords"):
                tfidf_vectorizer = TfidfVectorizer(max_df=0.85, max_features=10000, min_df=1, stop_words='english', use_idf=True, ngram_range=(1,2))
                try:
                    tfidf_matrix = tfidf_vectorizer.fit_transform(textlist)
                    
                    ds = DBSCAN(eps=sensitivity, min_samples=min_clustersize).fit(tfidf_matrix)
                    clusters = ds.labels_.tolist()
                    
                    cluster_df = pd.DataFrame(clusters, columns=['Cluster'])
                    keywords_df = pd.DataFrame(labellist, columns=['Keyword'])
                    result = pd.merge(cluster_df, keywords_df, left_index=True, right_index=True)
                    
                    # Assign remaining keywords to "Miscellaneous"
                    result['Cluster'] = result['Cluster'].apply(lambda x: 'Miscellaneous' if x == -1 else x)
                    
                    # Combine pre-classified and clustered keywords
                    if not classified_df.empty:
                        classified_df['Cluster'] = classified_df['Classification']
                    final_df = pd.concat([classified_df, result[['Cluster', 'Keyword']]])
                    
                    # Merge back optional columns
                    final_df = pd.merge(final_df, df, left_on='Keyword', right_on='Keywords', how='left')
                    
                    # Ensure all columns exist before aggregation
                    for col in ["Search Volume", "CPC", "Ranked Position", "URL"]:
                        if col not in final_df.columns:
                            final_df[col] = ''
                    
                    # Separate rows for each keyword and their attributes
                    output_df = final_df.explode('Keyword')
                    
                    output = io.BytesIO()
                    output_df.to_csv(output, index=False)
                    output.seek(0)
                    
                    st.download_button(
                        label="Download Clustered Keywords CSV",
                        data=output,
                        file_name="clustered_keywords.csv",
                        mime="text/csv"
                    )

                    st.dataframe(output_df)
                except ValueError as e:
                    st.error(f"An error occurred during clustering: {e}")
                    st.text("Make sure your data is properly formatted and contains enough unique terms to cluster.")

# To run this app, save the script and run `streamlit run script_name.py`
