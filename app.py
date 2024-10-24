# app.py

import streamlit as st
from scholarly import scholarly
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from collections import Counter
import networkx as nx

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

def extract_publication_data(results):
    data = {
        'title': [],
        'author': [],  # Keep full author list
        'author_display': [],  # Add separate field for display
        'year': [],
        'citations': [],
        'abstract': [],
        'url': [],
        'APA_citation': []
    }
    
    for pub in results:
        # Extract basic publication info
        title = pub['bib'].get('title', 'N/A')
        authors = pub['bib'].get('author', ['N/A'])
        year = pub['bib'].get('pub_year', 'N/A')
        citations = pub.get('num_citations', 0)
        abstract = pub.get('bib', {}).get('abstract', 'N/A')
        url = pub.get('pub_url', 'N/A')
        
        # Get additional information for citation
        journal = pub['bib'].get('journal', '')
        volume = pub['bib'].get('volume', '')
        issue = pub['bib'].get('number', '')
        pages = pub['bib'].get('pages', '')
        doi = pub.get('doi', '')

        # Format authors for APA citation
        if len(authors) == 1:
            author_citation = authors[0]
        elif len(authors) == 2:
            author_citation = f"{authors[0]} & {authors[1]}"
        elif len(authors) > 2:
            author_citation = f"{authors[0]} et al."

        # Create display version with et al.
        author_display = authors[0] + " et al." if len(authors) > 1 else authors[0]

        # Build APA citation
        apa_citation = f"{author_citation} ({year}). {title}"
        
        # Add journal info if available
        if journal:
            apa_citation += f". {journal}"
            
            # Add volume, issue, pages
            if volume:
                apa_citation += f", {volume}"
                if issue:
                    apa_citation += f"({issue})"
            if pages:
                apa_citation += f", {pages}"
        
        # Add DOI or URL
        if doi:
            apa_citation += f". https://doi.org/{doi}"
        elif url and url != 'N/A':
            apa_citation += f". Retrieved from {url}"
        
        apa_citation += "."  # Final period

        # Store data
        data['title'].append(title)
        data['author'].append(authors)
        data['author_display'].append(author_display)
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
    plt.close()

    # Plot citation distribution
    st.subheader("Citation Distribution")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(df['citations'], bins=20, color='skyblue', kde=True, ax=ax)
    ax.set_title('Citation Distribution')
    ax.set_xlabel('Number of Citations')
    ax.set_ylabel('Frequency')
    st.pyplot(fig)
    plt.close()

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
    plt.close()
    
    return common_keywords

# Function for author insights using a network graph
def visualize_author_network(df):
    st.subheader("Author Collaboration Network")

    # Create a new graph
    G = nx.Graph()
    
    # Process each paper's author list
    for authors in df['author']:
        if isinstance(authors, list) and len(authors) > 1:
            # Add nodes for each author
            for author in authors:
                if not G.has_node(author):
                    G.add_node(author)
            
            # Add edges between all pairs of authors in this paper
            for i in range(len(authors)):
                for j in range(i + 1, len(authors)):
                    if G.has_edge(authors[i], authors[j]):
                        # Increment weight if edge exists
                        G[authors[i]][authors[j]]['weight'] += 1
                    else:
                        # Create new edge with weight 1
                        G.add_edge(authors[i], authors[j], weight=1)
    
    if len(G.nodes()) > 0:
        # Clear any existing plots
        plt.clf()
        
        # Create the visualization
        plt.figure(figsize=(12, 8))
        
        # Calculate node sizes based on degree centrality
        node_size = [3000 * (1 + G.degree(node)) for node in G.nodes()]
        
        # Calculate edge weights for width
        edge_width = [G[u][v]['weight'] * 2 for (u, v) in G.edges()]
        
        # Use spring layout with adjusted parameters
        pos = nx.spring_layout(G, k=1, iterations=50)
        
        # Draw the network
        nx.draw(G, pos,
                node_color='lightblue',
                node_size=node_size,
                edge_color='gray',
                width=edge_width,
                with_labels=True,
                font_size=8,
                font_weight='bold')
        
        # Add a title
        plt.title("Author Collaboration Network\nNode size indicates number of collaborations")
        
        # Display the plot
        st.pyplot(plt)
        plt.close()
    else:
        st.write("No collaboration network could be generated. This might be because there are no papers with multiple authors in the dataset.")

# Streamlit app layout
def main():
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

        # Slider to pick number of articles
        num_results_slider = st.slider('Pick the number of articles:', min_value=1, max_value=100, value=10)

        if query:
            with st.spinner('Fetching publications...'):
                # Fetch and display data based on user input
                results = fetch_publications(query, num_results=num_results_slider)
                
                if results:
                    df_publications = extract_publication_data(results)
                    st.write("Fetched Data:")
                    st.dataframe(df_publications)
                else:
                    st.write("No results found for the query.")

    # Tabs for different functionalities
    if not df_publications.empty:
        tab1, tab2, tab3, tab4 = st.tabs(["Citation Analysis", "Keyword Analysis", "Citation Statistics", "Author Insights"])

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

        with tab4:
            st.write("### Author Insights")
            st.write("This tab shows a network graph of author collaborations.")
            visualize_author_network(df_publications)

    # Footer
    st.markdown("<hr><center><small>This tool is developed by Dr. Madeeh Elgedawy</small></center>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
