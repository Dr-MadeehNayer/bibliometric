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
        'url': [],       # Added URL field
        'APA_citation': []  # Added APA citation field
    }
    
    for pub in results:
        title = pub['bib'].get('title', 'N/A')
        author = pub['bib'].get('author', 'N/A')
        year = pub['bib'].get('pub_year', 'N/A')
        citations = pub.get('num_citations', 0)
        abstract = pub.get('bib', {}).get('abstract', 'N/A')
        url = pub.get('pub_url', 'N/A')

        # Extract the first author and format
        if isinstance(author, list):
            author = author[0] + " et al." if len(author) > 1 else author[0]

        # Generate APA-style citation
        apa_citation = f"{author} ({year}). {title}."

        data['title'].append(title)
        data['author'].append(author)
        data['year'].append(year)
        data['citations'].append(citations)
        data['abstract'].append(abstract)
        data['url'].append(url)
        data['APA_citation'].append(apa_citation)
    
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
st.title('Bibliometric Analysis with Google Scholar Data')

# File upload section
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

# Initialize DataFrame for publications
df_publications = pd.DataFrame()

# If CSV file is uploaded, load the data into DataFrame
if uploaded_file:
    df_publications = pd.read_csv(uploaded_file)
    st.write("Uploaded Data:")
    st.dataframe(df_publications)
else:
    # User inputs if no file is uploaded
    query = st.text_input('Enter your search query (e.g., "artificial intelligence in healthcare")', '')

    # Combine slider and textbox to pick number of articles
    st.write("Select or enter the number of articles to retrieve:")
    num_results_slider = st.slider('Pick the number of articles:', min_value=1, max_value=100, value=10)
    num_results_textbox = st.number_input('Or enter the number of articles:', min_value=1, max_value=100, value=num_results_slider)

    # Use the textbox value if it's been entered, otherwise use the slider value
    num_results = num_results_textbox if num_results_textbox else num_results_slider

    if query:
        # Fetch and display data based on user input
        results = fetch_publications(query, num_results=num_results)
        
        if results:
            df_publications = extract_publication_data(results)
            st.write("Fetched Data:")
            st.dataframe(df_publications)
        else:
            st.write("No results found for the query.")

# Tabs for different functionalities
if not df_publications.empty:
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

# Footer
st.markdown("<hr><center><small>This tool is developed by Dr. Madeeh Elgedawy</small></center>", unsafe_allow_html=True)
