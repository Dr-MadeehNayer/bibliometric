# app.py

import streamlit as st
from scholarly import scholarly
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from collections import Counter

# Function to fetch publications based on a search query
def fetch_publications(query, num_results=10):
    search_query = scholarly.search_pubs(query)
    results = []
    for _ in range(num_results):
        try:
            pub = next(search_query)
            results.append(pub)
        except StopIteration:
            break
    return results

# Function to extract relevant data from publications
def extract_publication_data(results):
    data = {
        'title': [],
        'author': [],
        'year': [],
        'citations': [],
        'abstract': [],  # Added abstract field
        'url': []        # Added URL field
    }
    
    for pub in results:
        data['title'].append(pub['bib'].get('title', 'N/A'))
        data['author'].append(pub['bib'].get('author', 'N/A'))
        data['year'].append(pub['bib'].get('pub_year', 'N/A'))
        data['citations'].append(pub.get('num_citations', 0))
        data['abstract'].append(pub.get('bib', {}).get('abstract', 'N/A'))  # Fetching the abstract
        data['url'].append(pub.get('pub_url', 'N/A'))  # Fetching the article URL
    
    return pd.DataFrame(data)

# Function for citation trend visualization
def visualize_citation_trends(df):
    # Convert the 'year' column to numeric, setting errors='coerce' will convert non-integer values to NaN
    df['year'] = pd.to_numeric(df['year'], errors='coerce')
    
    # Drop rows where 'year' is NaN (non-integer values)
    df = df.dropna(subset=['year'])
    
    # Convert 'year' column to integers
    df['year'] = df['year'].astype(int)

    # Plot number of publications per year
    st.subheader("Number of Publications per Year")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.countplot(data=df, x='year', palette='Blues', ax=ax)
    plt.xticks(rotation=45)
    ax.set_title('Number of Publications per Year')
    ax.set_xlabel('Publication Year')
    ax.set_ylabel('Count')
    st.pyplot(fig)

    # Plot citation distribution
    st.subheader("Citation Distribution")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(df['citations'], bins=20, color='skyblue', kde=True, ax=ax)
    ax.set_title('Citation Distribution')
    ax.set_xlabel('Number of Citations')
    ax.set_ylabel('Frequency')
    st.pyplot(fig)


# Function to analyze keywords in titles
def analyze_keywords_in_titles(df, num_keywords=10):
    titles = df['title'].tolist()
    words = []
    
    for title in titles:
        words.extend(re.findall(r'\w+', title.lower()))

    # Filter out common stopwords
    stopwords = set(['and', 'the', 'of', 'in', 'for', 'on', 'with', 'to', 'a', 'is', 'an', 'at', 'using'])
    filtered_words = [word for word in words if word not in stopwords]
    
    # Count frequency of words
    common_words = Counter(filtered_words)
    common_keywords = common_words.most_common(num_keywords)

    # Plot most common keywords
    st.subheader(f'Top {num_keywords} Keywords in Titles')
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=[item[1] for item in common_keywords], y=[item[0] for item in common_keywords], palette='coolwarm', ax=ax)
    ax.set_title(f'Top {num_keywords} Keywords in Titles')
    ax.set_xlabel('Frequency')
    ax.set_ylabel('Keywords')
    st.pyplot(fig)
    
    return common_keywords

# Streamlit app layout
st.title('Bibliometric Analysis with Google Scholar Data - IPA Case Study')

# User inputs
query = st.text_input('Enter your search query (e.g., "explainable artificial intelligence")', '')
num_results = st.number_input('Enter the number of articles to retrieve', min_value=1, max_value=500, value=100)

if query:
    # Fetch and display data based on user input
    results = fetch_publications(query, num_results=num_results)
    
    if results:
        df_publications = extract_publication_data(results)
        
        # Display fetched data
        st.write("Fetched Data:")
        st.dataframe(df_publications)

        # Tabs for different functionalities
        tab1, tab2, tab3 = st.tabs(["Citation Analysis", "Keyword Analysis", "Citation Statistics"])

        with tab1:
            st.write("### Citation Analysis")
            st.write("This tab shows the number of publications per year and the citation distribution.")
            visualize_citation_trends(df_publications)

        with tab2:
            st.write("### Keyword Analysis")
            st.write("This tab shows the most frequent keywords in the publication titles.")
            common_keywords = analyze_keywords_in_titles(df_publications, num_keywords=10)
            st.write("Most common keywords:", common_keywords)

        with tab3:
            st.write("### Citation Statistics")
            st.write("This tab shows basic statistics for citation counts (e.g., mean, median, standard deviation).")
            citation_stats = df_publications['citations'].describe()
            st.write(citation_stats)
    else:
        st.write("No results found for the query.")

st.markdown("<hr><center><small>This tool is developed by Dr. Madeeh Elgedawy</small></center>", unsafe_allow_html=True)
