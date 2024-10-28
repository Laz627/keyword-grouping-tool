import streamlit as st
import pandas as pd
from keybert import KeyBERT

# Initialize the KeyBERT model
kw_model = KeyBERT()

st.title("Keyword Theme Extraction with KeyBERT")
st.markdown("Upload a CSV file with a 'Keywords' column. The tool will extract relevant themes using KeyBERT.")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv", "xls", "xlsx"])

# Function to apply KeyBERT and extract unigrams, bigrams, and trigrams
def apply_keybert(df):
    if 'Keywords' not in df.columns:
        st.error("Error: The dataframe must contain a column named 'Keywords'.")
        return None

    # Function to extract n-grams using KeyBERT
    def extract_ngram(text, ngram_range):
        keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=ngram_range, stop_words='english')
        return keywords[0][0] if keywords else ""  # Return the keyword or an empty string if none found

    # Apply KeyBERT for unigrams, bigrams, and trigrams
    df['Core (1-gram)'] = df['Keywords'].apply(lambda x: extract_ngram(x, (1, 1)) if len(x) > 0 else "")
    df['Core (2-gram)'] = df['Keywords'].apply(lambda x: extract_ngram(x, (2, 2)) if len(x) > 0 else "")
    df['Core (3-gram)'] = df['Keywords'].apply(lambda x: extract_ngram(x, (3, 3)) if len(x) > 0 else "")

    return df

if uploaded_file:
    # Load the uploaded file into a DataFrame
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    
    # Apply KeyBERT and display results
    df_with_keybert = apply_keybert(df)
    
    if df_with_keybert is not None:
        st.write("Extracted Themes for Keywords:")
        st.dataframe(df_with_keybert)

        # Option to download the modified DataFrame
        csv = df_with_keybert.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Extracted Themes CSV",
            data=csv,
            file_name="keywords_with_themes.csv",
            mime="text/csv"
        )
