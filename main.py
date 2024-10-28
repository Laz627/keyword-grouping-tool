import pandas as pd
from keybert import KeyBERT
from collections import Counter
import re

def cluster_keywords(keywords, seed_keyword=''):
    """
    Clusters keywords into meaningful groups based on keyphrase extraction.

    Parameters:
    - keywords: list of strings or pandas DataFrame with a 'Keywords' column.
    - seed_keyword: optional string to provide context for theme extraction.

    Returns:
    - pandas DataFrame with original keywords and assigned themes.
    """
    # Validate input
    if isinstance(keywords, list):
        df = pd.DataFrame({'Keywords': keywords})
    elif isinstance(keywords, pd.DataFrame):
        if 'Keywords' not in keywords.columns:
            raise ValueError("DataFrame must contain a 'Keywords' column.")
        df = keywords.copy()
    else:
        raise TypeError("Input must be a list of strings or a pandas DataFrame with a 'Keywords' column.")
    
    # Initialize KeyBERT model
    kw_model = KeyBERT()
    
    # Function to extract keyphrases (n-grams) from text
    def extract_keyphrases(text):
        """
        Extracts keyphrases (unigrams, bigrams, trigrams) from text using KeyBERT.

        Parameters:
        - text: string to extract keyphrases from.

        Returns:
        - dict with n-grams as keys and extracted phrases as values.
        """
        keyphrases = {}
        for n in range(1, 4):
            keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(n, n), stop_words='english')
            keyphrases[f'{n}-gram'] = keywords[0][0] if keywords else ''
        return keyphrases
    
    # Apply keyphrase extraction to each keyword
    df['Keyphrases'] = df['Keywords'].apply(extract_keyphrases)
    
    # Function to clean keyphrases by removing the seed keyword if provided
    def clean_phrase(phrase):
        """
        Removes the seed keyword from the phrase if present.

        Parameters:
        - phrase: string to clean.

        Returns:
        - cleaned phrase.
        """
        if seed_keyword:
            pattern = rf'\b{re.escape(seed_keyword)}\b'
            cleaned = re.sub(pattern, '', phrase, flags=re.IGNORECASE).strip()
            return cleaned if len(cleaned.split()) > 0 else phrase  # Retain phrase if empty after removal
        return phrase
    
    # Clean keyphrases in the DataFrame
    df['Cleaned Keyphrases'] = df['Keyphrases'].apply(
        lambda kp_dict: {k: clean_phrase(v) for k, v in kp_dict.items()}
    )
    
    # Initialize an empty dictionary to store clusters
    clusters = {}
    
    # Recursive function to cluster keywords
    def recursive_cluster(dataframe, depth=0, max_depth=5):
        """
        Recursively clusters the keywords based on common terms in keyphrases.

        Parameters:
        - dataframe: pandas DataFrame with 'Keywords' and 'Cleaned Keyphrases' columns.
        - depth: current depth of recursion.
        - max_depth: maximum depth to prevent infinite recursion.

        Returns:
        - None (clusters are stored in the 'clusters' dictionary).
        """
        # Base case: stop recursion if depth limit is reached or dataframe is small
        if depth >= max_depth or len(dataframe) <= 1:
            # Assign a unique cluster label
            cluster_label = f"Cluster_{len(clusters)+1}"
            clusters[cluster_label] = dataframe['Keywords'].tolist()
            return
        
        # Combine all cleaned keyphrases to find common terms
        all_phrases = []
        for kp_dict in dataframe['Cleaned Keyphrases']:
            all_phrases.extend(kp_dict.values())
        
        # Count term frequencies
        word_counts = Counter()
        for phrase in all_phrases:
            words = re.findall(r'\w+', phrase.lower())
            word_counts.update(words)
        
        # If no words found, assign to a cluster
        if not word_counts:
            cluster_label = f"Cluster_{len(clusters)+1}"
            clusters[cluster_label] = dataframe['Keywords'].tolist()
            return
        
        # Find the most common term
        common_term, freq = word_counts.most_common(1)[0]
        # If no term is common enough, assign to a cluster
        if freq < 2:
            cluster_label = f"Cluster_{len(clusters)+1}"
            clusters[cluster_label] = dataframe['Keywords'].tolist()
            return
        
        # Split dataframe into two groups: contains the common term, and does not
        contains_term = dataframe[dataframe['Cleaned Keyphrases'].apply(
            lambda kp_dict: any(re.search(rf'\b{common_term}\b', phrase, re.IGNORECASE) for phrase in kp_dict.values())
        )]
        does_not_contain_term = dataframe.drop(contains_term.index)
        
        # Recursively cluster each subgroup
        recursive_cluster(contains_term, depth+1, max_depth)
        recursive_cluster(does_not_contain_term, depth+1, max_depth)
    
    # Start recursive clustering from the top level
    recursive_cluster(df)
    
    # Assign cluster labels to the original DataFrame
    cluster_assignments = {}
    for cluster_label, keywords_list in clusters.items():
        for keyword in keywords_list:
            cluster_assignments[keyword] = cluster_label
    
    df['Cluster'] = df['Keywords'].map(cluster_assignments)
    
    # Return the DataFrame with clusters
    return df[['Keywords', 'Cluster']]
