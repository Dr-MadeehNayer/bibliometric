import streamlit as st
from scholarly import scholarly, ProxyGenerator
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
from st_aggrid import AgGrid, GridUpdateMode, DataReturnMode
import io
from time import sleep
warnings.filterwarnings('ignore')

# Function to fetch publications
def fetch_publications(query, num_results=10):
    try:
        pg = ProxyGenerator()
        pg.FreeProxies()
        scholarly.use_proxy(pg)
    except:
        pass
    
    search_query = scholarly.search_pubs(query)
    results = []
    for i in range(num_results):
        try:
            pub = next(search_query)
            results.append(pub)
            if i % 5 == 0:
                sleep(1)
        except StopIteration:
            break
        except Exception as e:
            st.error(f"Error fetching publication {i+1}: {str(e)}")
            continue
    return results

# Function to extract publication data
def extract_publication_data(results):
    data = {
        'selected': [],
        'title': [],
        'author': [],
        'author_display': [],
        'year': [],
        'citations': [],
        'abstract': [],
        'url': [],
        'journal': [],
        'APA_citation': []
    }
    
    for pub in results:
        data['selected'].append(False)
        
        title = pub['bib'].get('title', 'N/A')
        authors = pub['bib'].get('author', ['N/A'])
        year = pub['bib'].get('pub_year', 'N/A')
        citations = pub.get('num_citations', 0)
        abstract = pub.get('bib', {}).get('abstract', 'N/A')
        url = pub.get('pub_url', 'N/A')
        
        journal = 'N/A'
        if 'journal' in pub['bib']:
            journal = pub['bib']['journal']
        elif 'venue' in pub['bib']:
            journal = pub['bib']['venue']
        elif 'conference' in pub['bib']:
            journal = pub['bib']['conference']
        elif 'publisher' in pub['bib']:
            journal = pub['bib']['publisher']

        author_display = authors[0] + " et al." if len(authors) > 1 else authors[0]
        
        data['title'].append(title)
        data['author'].append(authors)
        data['author_display'].append(author_display)
        data['year'].append(year)
        data['citations'].append(citations)
        data['abstract'].append(abstract)
        data['url'].append(url)
        data['journal'].append(journal)
        data['APA_citation'].append(f"{author_display} ({year}). {title}. {journal}.")
    
    return pd.DataFrame(data)

# Function to display publications grid
def display_publications_grid(df_publications):
    grid_options = {
        'columnDefs': [
            {
                'field': 'selected',
                'headerName': '',
                'checkboxSelection': True,
                'headerCheckboxSelection': True,
                'headerCheckboxSelectionFilteredOnly': True,
                'width': 50,
                'pinned': 'left',
                'lockPosition': True,
                'suppressMenu': True,
                'suppressMovable': True,
            },
            {
                'field': 'title',
                'headerName': 'Title',
                'width': 300,
                'sortable': True,
                'filter': True
            },
            {
                'field': 'author_display',
                'headerName': 'Authors',
                'width': 200,
                'sortable': True,
                'filter': True
            },
            {
                'field': 'year',
                'headerName': 'Year',
                'width': 100,
                'sortable': True,
                'filter': True
            },
            {
                'field': 'citations',
                'headerName': 'Citations',
                'width': 100,
                'sortable': True,
                'filter': True
            },
            {
                'field': 'journal',
                'headerName': 'Journal',
                'width': 200,
                'sortable': True,
                'filter': True
            }
        ],
        'rowSelection': 'multiple',
        'suppressRowClickSelection': True,
        'suppressCellSelection': False,
        'enableRangeSelection': True,
        'pagination': True,
        'paginationAutoPageSize': True,
    }

    return AgGrid(
        df_publications,
        gridOptions=grid_options,
        update_mode=GridUpdateMode.MODEL_CHANGED | GridUpdateMode.SELECTION_CHANGED,
        data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
        fit_columns_on_grid_load=True,
        allow_unsafe_jscode=True,
        theme='streamlit'
    )

# Function to delete selected rows
def delete_selected_rows(df, selected_rows):
    if selected_rows:
        selected_titles = [row['title'] for row in selected_rows]
        df = df[~df['title'].isin(selected_titles)].copy()
        df.reset_index(drop=True, inplace=True)
    return df

# Citation analysis function
def analyze_citations(df):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(df['citations'], bins=20, kde=True)
    plt.title("Citation Distribution")
    plt.xlabel("Number of Citations")
    plt.ylabel("Frequency")
    return fig

# Author network analysis function
def visualize_author_network(df):
    G = nx.Graph()
    
    for authors in df['author']:
        if isinstance(authors, list) and len(authors) > 1:
            for i in range(len(authors)):
                for j in range(i + 1, len(authors)):
                    if G.has_edge(authors[i], authors[j]):
                        G[authors[i]][authors[j]]['weight'] += 1
                    else:
                        G.add_edge(authors[i], authors[j], weight=1)
    
    if len(G.nodes()) > 0:
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(G)
        nx.draw(G, pos,
                node_color='lightblue',
                node_size=[3000 * (1 + G.degree(node)) for node in G.nodes()],
                with_labels=True,
                font_size=8,
                font_weight='bold')
        plt.title("Author Collaboration Network")
        return plt.gcf()
    return None

# Journal analysis function
def analyze_journals(df):
    journal_data = df[df['journal'] != 'N/A']
    if len(journal_data) == 0:
        return None
        
    journal_stats = journal_data.groupby('journal').agg({
        'citations': ['count', 'sum', 'mean']
    }).sort_values(('citations', 'sum'), ascending=False)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    journal_stats[('citations', 'sum')][:10].plot(kind='bar')
    plt.title("Top 10 Most Cited Journals")
    plt.xlabel("Journal")
    plt.ylabel("Total Citations")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    return fig, journal_stats

# Topic analysis functions
def analyze_topics(df, num_topics=5):
    def preprocess_text(text):
        if isinstance(text, str):
            text = text.lower()
            text = re.sub(r'http\S+|www\S+|https\S+', '', text)
            text = re.sub(r'\d+', '', text)
            text = re.sub(r'[^\w\s]', '', text)
            return text
        return ''

    processed_abstracts = [preprocess_text(abstract) for abstract in df['abstract']]
    
    vectorizer = CountVectorizer(
        max_features=1000,
        min_df=2,
        max_df=0.95,
        stop_words='english',
        token_pattern=r'[a-zA-Z]{3,}'
    )
    
    doc_term_matrix = vectorizer.fit_transform(processed_abstracts)
    feature_names = vectorizer.get_feature_names_out()
    
    lda_model = LatentDirichletAllocation(
        n_components=num_topics,
        random_state=42,
        max_iter=25,
        learning_method='online',
        n_jobs=-1
    )
    
    lda_output = lda_model.fit_transform(doc_term_matrix)
    
    topics_data = []
    for topic_idx, topic in enumerate(lda_model.components_):
        top_words_idx = topic.argsort()[:-10-1:-1]
        topics_data.append({
            'topic': f"Topic {topic_idx + 1}",
            'words': [feature_names[i] for i in top_words_idx],
            'weights': [topic[i] for i in top_words_idx]
        })
    
    return topics_data

def visualize_topics(topics_data):
    for topic in topics_data:
        fig, ax = plt.subplots(figsize=(10, 4))
        y_pos = np.arange(len(topic['words']))
        
        ax.barh(y_pos, topic['weights'])
        ax.set_yticks(y_pos)
        ax.set_yticklabels(topic['words'])
        ax.invert_yaxis()
        ax.set_title(topic['topic'])
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

# Main application
def main():
    st.title('Bibliometric Analysis with Google Scholar Data')

    # Initialize session state
    if 'df_publications' not in st.session_state:
        st.session_state.df_publications = pd.DataFrame()

    # Main interface with two columns for input options
    col1, col2 = st.columns(2)
    
    with col1:
        uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    
    with col2:
        query = st.text_input('Or enter a search query (e.g., "artificial intelligence in healthcare")')
        if query:
            num_results_slider = st.slider('Number of articles to fetch:', min_value=1, max_value=100, value=10)

    # Handle file upload
    if uploaded_file:
        try:
            st.session_state.df_publications = pd.read_csv(uploaded_file)
            st.success("File successfully uploaded!")
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
    
    # Handle search query
    elif query:
        with st.spinner('Fetching publications...'):
            try:
                results = fetch_publications(query, num_results=num_results_slider)
                if results:
                    st.session_state.df_publications = extract_publication_data(results)
                    st.success(f"Successfully fetched {len(results)} publications!")
                else:
                    st.warning("No results found for the query.")
            except Exception as e:
                st.error(f"Error fetching publications: {str(e)}")

    # Display and interact with publications
    if not st.session_state.df_publications.empty:
        st.divider()
        
        # Delete button
        col1, col2 = st.columns([1, 6])
        with col1:
            delete_button = st.button('🗑️ Delete Selected', type='primary')
        
        # Display grid
        grid_response = display_publications_grid(st.session_state.df_publications)
        
        # Handle deletion
        if delete_button:
            if grid_response['selected_rows']:
                num_selected = len(grid_response['selected_rows'])
                st.session_state.df_publications = delete_selected_rows(
                    st.session_state.df_publications,
                    grid_response['selected_rows']
                )
                st.success(f"Deleted {num_selected} selected row{'s' if num_selected > 1 else ''}!")
                st.experimental_rerun()
            else:
                st.warning("Please select rows to delete.")

        # Analysis tabs
        if len(st.session_state.df_publications) > 0:
            st.divider()
            
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "Citation Analysis",
                "Topic Analysis",
                "Author Network",
                "Journal Analysis",
                "Export Options"
            ])

            with tab1:
                st.write("### Citation Analysis")
                try:
                    citations = st.session_state.df_publications['citations']
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Citations", f"{citations.sum():,}")
                    with col2:
                        st.metric("Average Citations", f"{citations.mean():.1f}")
                    with col3:
                        st.metric("Median Citations", f"{citations.median():.1f}")
                    
                    fig = analyze_citations(st.session_state.df_publications)
                    st.pyplot(fig)
                    plt.close()
                except Exception as e:
                    st.error(f"Error in citation analysis: {str(e)}")

            with tab2:
                st.write("### Topic Analysis")
                if st.session_state.df_publications['abstract'].isna().all():
                    st.warning("No abstracts available for topic analysis.")
                else:
                    num_topics = st.slider("Number of topics", 2, 10, 5)
                    topics_data = analyze_topics(
                        st.session_state.df_publications,
                        num_topics=num_topics
                    )
                    visualize_topics(topics_data)

            with tab3:
                st.write("### Author Network")
                fig = visualize_author_network(st.session_state.df_publications)
                if fig:
                    st.pyplot(fig)
                    plt.close()
                else:
                    st.warning("Not enough collaboration data for network analysis.")

            with tab4:
                st.write("### Journal Analysis")
                try:
                    fig, stats = analyze_journals(st.session_state.df_publications)
                    if fig:
                        st.pyplot(fig)
                        plt.close()
                        
                        st.write("### Journal Statistics")
                        st.dataframe(stats)
                    else:
                        st.warning("No journal data available for analysis.")
                except Exception as e:
                    st.error(f"Error in journal analysis: {str(e)}")

            with tab5:
                st.write("### Export Options")
                if st.button("Export to CSV"):
                    try:
                        csv = st.session_state.df_publications.to_csv(index=False)
                        st.download_button(
                            label="Download CSV",
                            data=csv,
                            file_name="bibliometric_analysis.csv",
                            mime="text/csv"
                        )
                    except Exception as e:
                        st.error(f"Error exporting data: {str(e)}")

    # Footer
    st.markdown("""---\n<center>Bibliometric Analysis Tool</center>""", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
