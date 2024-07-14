import streamlit as st
import pandas as pd
import re
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
import nltk
from nltk.stem.snowball import SnowballStemmer
import io

# Initialize stemmer
snow_stemmer = SnowballStemmer(language='english')

# Function to stem list of keywords
def stemmList(list):
    stemmed_list = []
    for l in list:
        words = l.split(" ")
        stem_words = [snow_stemmer.stem(word) for word in words]
        stemmed_list.append(" ".join(stem_words))
    return stemmed_list

# Title and Instructions
st.title("Free Keyword Clustering Tool")
st.markdown("""
    **Instructions:**
    1. Upload a CSV file with your queries in the first column.
    2. Enter L1 and L2 classifications to help reduce "miscellaneous" keywords.
    3. Adjust the sensitivity for clustering.
    4. The tool will classify and cluster the keywords, and generate a downloadable CSV file with the results.
    
    **Created by:** Brandon Lazovic
""")

# File upload
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df['Keywords'] = df.iloc[:, 0].astype(str)
    
    # User input for L1 and L2 classifications
    l1_input = st.text_area("Enter L1 Classifications (comma-separated)", "Plush")
    l2_input = st.text_area("Enter L2 Classifications (comma-separated)", "Plush Elves, Plush Reindeer, Plush Santa, Plush Gnomes")

    l1_classifications = [cls.strip() for cls in l1_input.split(',')]
    l2_classifications = [cls.strip() for cls in l2_input.split(',')]
    
    def classify_keywords(keyword):
        for l1 in l1_classifications:
            if any(re.search(rf'\b{re.escape(l2)}\b', keyword, re.IGNORECASE) for l2 in l2_classifications):
                return l1
        return None

    df['Classification'] = df['Keywords'].apply(classify_keywords)
    
    # Separate pre-classified and unclassified keywords
    classified_df = df.dropna(subset=['Classification'])
    unclassified_df = df[df['Classification'].isna()]

    textlist = unclassified_df['Keywords'].to_list()
    labellist = textlist
    textlist = stemmList(textlist)
    
    # User input for clustering sensitivity
    sensitivity = st.slider("Select Clustering Sensitivity", min_value=0.01, max_value=1.0, value=0.2, step=0.01)
    min_clustersize = st.number_input("Minimum Cluster Size", min_value=1, value=2)

    if st.button("Classify and Cluster Keywords"):
        tfidf_vectorizer = TfidfVectorizer(max_df=0.2, max_features=10000, min_df=0.01, stop_words='english', use_idf=True, ngram_range=(1,2))
        tfidf_matrix = tfidf_vectorizer.fit_transform(textlist)
        
        ds = DBSCAN(eps=sensitivity, min_samples=min_clustersize).fit(tfidf_matrix)
        clusters = ds.labels_.tolist()
        
        cluster_df = pd.DataFrame(clusters, columns=['Cluster'])
        keywords_df = pd.DataFrame(labellist, columns=['Keyword'])
        result = pd.merge(cluster_df, keywords_df, left_index=True, right_index=True)
        
        # Assign remaining keywords to "Miscellaneous"
        result['Cluster'] = result['Cluster'].apply(lambda x: 'Miscellaneous' if x == -1 else x)
        
        # Combine pre-classified and clustered keywords
        classified_df['Cluster'] = classified_df['Classification']
        final_df = pd.concat([classified_df, result[['Cluster', 'Keyword']]])
        
        # Group and save the results to a CSV file
        grouping = final_df.groupby(['Cluster'])['Keyword'].apply(' | '.join).reset_index()
        
        output = io.BytesIO()
        grouping.to_csv(output, index=False)
        output.seek(0)
        
        st.download_button(
            label="Download Clustered Keywords CSV",
            data=output,
            file_name="clustered_keywords.csv",
            mime="text/csv"
        )

        st.dataframe(grouping)

# To run this app, save the script and run `streamlit run script_name.py`
