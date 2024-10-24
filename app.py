import streamlit as st
from scholarly import scholarly
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from collections import Counter
import networkx as nx
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np
import pyLDAvis
import pyLDAvis.sklearn
import warnings
warnings.filterwarnings('ignore')

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

def preprocess_text(text):
    """Preprocess text without using NLTK"""
    if isinstance(text, str):
        # Convert to lowercase
        text = text.lower()
        
        # Custom stopwords list
        stop_words = {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the', 
            'to', 'was', 'were', 'will', 'with', 'the', 'this', 'but', 'they',
            'have', 'had', 'what', 'when', 'where', 'who', 'which', 'why',
            'et', 'al', 'paper', 'study', 'research', 'method', 'results', 
            'analysis', 'data', 'using', 'used', 'proposed', 'based'
        }
        return text
    return ''

def get_feature_names(vectorizer):
    """Helper function to get feature names from vectorizer regardless of sklearn version"""
    try:
        return vectorizer.get_feature_names_out()
    except AttributeError:
        try:
            return vectorizer.get_feature_names()
        except AttributeError:
            return np.array(vectorizer.vocabulary_)

def analyze_topics(df, num_topics=5, num_words=10):
    """Analyze topics in the abstracts using LDA"""
    # Preprocess abstracts
    processed_abstracts = [preprocess_text(abstract) for abstract in df['abstract']]
    
    # Create document-term matrix with built-in stopwords and preprocessing
    vectorizer = CountVectorizer(
        max_features=1000,
        min_df=2,
        max_df=0.95,
        stop_words='english',
        token_pattern=r'[a-zA-Z]+(?:\s[a-zA-Z]+)*',
    )
    
    doc_term_matrix = vectorizer.fit_transform(processed_abstracts)
    
    # Create and fit LDA model
    lda_model = LatentDirichletAllocation(
        n_components=num_topics,
        random_state=42,
        max_iter=20,
        learning_method='batch',
        evaluate_every=-1,
        n_jobs=-1,
    )
    
    lda_output = lda_model.fit_transform(doc_term_matrix)
    
    # Get feature names using the helper function
    feature_names = get_feature_names(vectorizer)
    
    # Prepare topic visualization data
    topics_data = []
    for topic_idx, topic in enumerate(lda_model.components_):
        top_words_idx = topic.argsort()[:-num_words-1:-1]
        top_words = [feature_names[i] for i in top_words_idx]
        word_weights = [topic[i] for i in top_words_idx]
        topics_data.append({
            'topic_id': topic_idx + 1,
            'words': top_words,
            'weights': word_weights
        })
    
    # Get dominant topic for each document
    dominant_topics = np.argmax(lda_output, axis=1)
    topic_distributions = lda_output
    
    return topics_data, dominant_topics, topic_distributions, vectorizer, lda_model, doc_term_matrix


def visualize_topics(topics_data):
    """Create visualizations for topic analysis"""
    # Topic-Word Distribution Plot
    for topic in topics_data:
        fig, ax = plt.subplots(figsize=(10, 4))
        y_pos = np.arange(len(topic['words']))
        
        # Create horizontal bar chart
        bars = ax.barh(y_pos, topic['weights'], align='center')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(topic['words'])
        ax.invert_yaxis()
        ax.set_xlabel('Weight')
        ax.set_title(f'Top Words in Topic {topic["topic_id"]}')
        
        # Add weight values on the bars - Fixed the text positioning
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2,
                   f'{topic["weights"][i]:.3f}',
                   ha='left', va='center')  # Removed the x parameter
        
        st.pyplot(fig)
        plt.close()

def create_interactive_topic_viz(vectorizer, lda_model, doc_term_matrix):
    """Create interactive topic visualization using pyLDAvis"""
    try:
        # Get feature names
        feature_names = get_feature_names(vectorizer)
        
        # Convert document term matrix to dense if it's sparse
        doc_term_matrix_dense = doc_term_matrix.toarray()
        
        # Calculate term frequencies
        term_frequencies = np.sum(doc_term_matrix_dense, axis=0)
        
        # Create the visualization manually
        data = {
            'topic_term_dists': lda_model.components_,
            'doc_topic_dists': lda_model.transform(doc_term_matrix),
            'doc_lengths': np.sum(doc_term_matrix_dense, axis=1),
            'vocab': feature_names,
            'term_frequency': term_frequencies
        }
        
        # Prepare the visualization
        prepared_data = pyLDAvis.prepare(**data)
        
        # Convert to HTML
        html_string = pyLDAvis.prepared_data_to_html(prepared_data)
        
        # Display in Streamlit
        st.components.v1.html(html_string, width=1300, height=800)
        
    except Exception as e:
        st.error(f"Error in visualization preparation: {str(e)}")
        
        # Fallback visualization
        st.write("Unable to create interactive visualization. Displaying simple topic summary instead:")
        
        # Get feature names for fallback visualization
        feature_names = get_feature_names(vectorizer)
        
        # Display simple topic summary
        for idx, topic_dist in enumerate(lda_model.components_):
            top_terms_idx = topic_dist.argsort()[:-10-1:-1]
            top_terms = [feature_names[i] for i in top_terms_idx]
            st.write(f"Topic {idx + 1}: {', '.join(top_terms)}")
            
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
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "Citation Analysis", 
            "Keyword Analysis", 
            "Citation Statistics", 
            "Author Insights",
            "Topics Analysis"
        ])

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

        with tab5:
            st.write("### Topics Analysis")
            st.write("This tab shows the main themes and topics discussed in the papers based on abstract analysis.")
            
            # Add number of topics selector
            num_topics = st.slider('Select number of topics to extract:', 
                             min_value=2, max_value=10, value=5)
            
            # Add number of words per topic selector
            num_words = st.slider('Select number of words per topic:', 
                            min_value=5, max_value=20, value=10)
            
            # Check if we have abstracts to analyze
            if df_publications['abstract'].isna().all() or all(df_publications['abstract'] == 'N/A'):
                st.warning("No abstracts available for topic analysis.")
            else:
                # Perform topic analysis
                with st.spinner('Analyzing topics...'):
                    try:
                        topics_data, dominant_topics, topic_distributions, vectorizer, lda_model, doc_term_matrix = \
                            analyze_topics(df_publications, num_topics=num_topics, num_words=num_words)
                        
                        # Display topic visualizations
                        st.subheader("Topic-Word Distributions")
                        visualize_topics(topics_data)
                        
                        # Display document-topic assignments
                        st.subheader("Document-Topic Assignments")
                        doc_topics_df = pd.DataFrame({
                            'Title': df_publications['title'],
                            'Dominant Topic': dominant_topics + 1,
                            'Top Words': [topics_data[topic]['words'][:5] for topic in dominant_topics]
                        })
                        st.dataframe(doc_topics_df)
                        
                        # Add interactive visualization
                        if doc_term_matrix.shape[1] > 0:  # Check if we have terms to visualize
                            st.subheader("Interactive Topic Visualization")
                            st.write("This visualization shows the relationships between topics and terms. " +
                                    "Each bubble represents a topic, and the distance between bubbles " +
                                    "indicates their similarity.")
                            try:
                                create_interactive_topic_viz(vectorizer, lda_model, doc_term_matrix)
                            except Exception as viz_error:
                                st.error(f"Error creating interactive visualization: {str(viz_error)}")
                        else:
                            st.warning("Not enough terms found for interactive visualization.")
                    except Exception as e:
                        st.error(f"Error during topic analysis: {str(e)}")

    # Footer
    st.markdown("<hr><center><small>This tool is developed by Dr. Madeeh Elgedawy</small></center>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
