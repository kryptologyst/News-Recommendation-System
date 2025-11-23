"""Streamlit demo for news recommendation system."""

import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import streamlit as st
import yaml

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from news_recommender.data.loader import NewsDataLoader
from news_recommender.models.recommenders import (
    TFIDFRecommender,
    SBERTRecommender,
    HybridRecommender
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="News Recommendation System",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        color: #1f77b4;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .recommendation-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data_and_models():
    """Load data and train models (cached for performance)."""
    try:
        # Load configuration
        config_path = Path(__file__).parent.parent / "configs" / "config.yaml"
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Initialize data loader
        data_loader = NewsDataLoader(
            data_dir="data",
            random_seed=config["data"]["random_seed"]
        )
        
        # Load data
        interactions = data_loader.load_interactions()
        items = data_loader.load_items()
        
        # Split data (use only train for demo)
        train_interactions, _, _ = data_loader.split_data(
            interactions,
            test_size=config["data"]["test_size"],
            val_size=config["data"]["val_size"]
        )
        
        # Train models
        models = {}
        
        # TF-IDF Model
        tfidf_model = TFIDFRecommender(
            max_features=config["models"]["tfidf"]["max_features"],
            ngram_range=tuple(config["models"]["tfidf"]["ngram_range"]),
            min_df=config["models"]["tfidf"]["min_df"],
            max_df=config["models"]["tfidf"]["max_df"]
        )
        tfidf_model.fit(items, train_interactions)
        models["TF-IDF"] = tfidf_model
        
        # SBERT Model
        sbert_model = SBERTRecommender(
            model_name=config["models"]["sbert"]["model_name"],
            batch_size=config["models"]["sbert"]["batch_size"]
        )
        sbert_model.fit(items, train_interactions)
        models["SBERT"] = sbert_model
        
        # Hybrid Model
        hybrid_model = HybridRecommender(
            tfidf_weight=config["models"]["hybrid"]["tfidf_weight"],
            sbert_weight=config["models"]["hybrid"]["sbert_weight"]
        )
        hybrid_model.fit(items, train_interactions)
        models["Hybrid"] = hybrid_model
        
        return items, train_interactions, models
        
    except Exception as e:
        st.error(f"Error loading data and models: {str(e)}")
        return None, None, None


def display_article_info(item_id: int, items: pd.DataFrame) -> None:
    """Display article information."""
    article = items[items["item_id"] == item_id].iloc[0]
    
    st.markdown(f"""
    <div class="recommendation-card">
        <h4>{article['title']}</h4>
        <p><strong>Category:</strong> {article.get('category', 'N/A')}</p>
        <p><strong>Content:</strong> {article['content'][:200]}...</p>
        <p><strong>Tags:</strong> {article.get('tags', 'N/A')}</p>
    </div>
    """, unsafe_allow_html=True)


def main():
    """Main Streamlit application."""
    st.markdown('<h1 class="main-header">📰 News Recommendation System</h1>', unsafe_allow_html=True)
    
    # Load data and models
    with st.spinner("Loading data and training models..."):
        items, interactions, models = load_data_and_models()
    
    if items is None or models is None:
        st.error("Failed to load data or models. Please check the configuration.")
        return
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["User Recommendations", "Item Similarity", "Model Comparison", "About"]
    )
    
    if page == "User Recommendations":
        st.header("👤 User Recommendations")
        
        # User selection
        available_users = sorted(interactions["user_id"].unique())
        selected_user = st.selectbox(
            "Select a user:",
            available_users,
            help="Choose a user to see personalized recommendations"
        )
        
        # Model selection
        selected_model = st.selectbox(
            "Select recommendation model:",
            list(models.keys()),
            help="Choose the model to use for recommendations"
        )
        
        # Number of recommendations
        n_recommendations = st.slider(
            "Number of recommendations:",
            min_value=5,
            max_value=20,
            value=10,
            help="Number of articles to recommend"
        )
        
        if st.button("Get Recommendations", type="primary"):
            with st.spinner("Generating recommendations..."):
                try:
                    # Get recommendations
                    recommendations = models[selected_model].recommend(
                        selected_user,
                        n_recommendations=n_recommendations,
                        exclude_seen=True,
                        interactions=interactions
                    )
                    
                    # Display user's reading history
                    user_history = interactions[interactions["user_id"] == selected_user]
                    st.subheader("📚 User's Reading History")
                    
                    if len(user_history) > 0:
                        for _, interaction in user_history.head(5).iterrows():
                            item_id = interaction["item_id"]
                            article = items[items["item_id"] == item_id].iloc[0]
                            st.write(f"• {article['title']} (Weight: {interaction['weight']})")
                    else:
                        st.write("No reading history available.")
                    
                    # Display recommendations
                    st.subheader(f"🎯 Recommendations from {selected_model} Model")
                    
                    for i, item_id in enumerate(recommendations, 1):
                        st.write(f"**{i}.**")
                        display_article_info(item_id, items)
                        
                except Exception as e:
                    st.error(f"Error generating recommendations: {str(e)}")
    
    elif page == "Item Similarity":
        st.header("🔍 Item Similarity Search")
        
        # Item selection
        available_items = sorted(items["item_id"].tolist())
        selected_item = st.selectbox(
            "Select an article:",
            available_items,
            format_func=lambda x: f"{x}: {items[items['item_id']==x].iloc[0]['title']}"
        )
        
        # Model selection
        selected_model = st.selectbox(
            "Select similarity model:",
            list(models.keys()),
            key="similarity_model"
        )
        
        # Number of similar items
        n_similar = st.slider(
            "Number of similar articles:",
            min_value=5,
            max_value=15,
            value=10,
            key="n_similar"
        )
        
        if st.button("Find Similar Articles", type="primary"):
            with st.spinner("Finding similar articles..."):
                try:
                    # Get similar items
                    similar_items = models[selected_model].get_similar_items(
                        selected_item, n_similar
                    )
                    
                    # Display selected article
                    st.subheader("📄 Selected Article")
                    display_article_info(selected_item, items)
                    
                    # Display similar articles
                    st.subheader(f"🔗 Similar Articles from {selected_model} Model")
                    
                    for i, (item_id, similarity_score) in enumerate(similar_items, 1):
                        st.write(f"**{i}.** (Similarity: {similarity_score:.3f})")
                        display_article_info(item_id, items)
                        
                except Exception as e:
                    st.error(f"Error finding similar articles: {str(e)}")
    
    elif page == "Model Comparison":
        st.header("📊 Model Comparison")
        
        # Display dataset statistics
        st.subheader("📈 Dataset Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Articles", len(items))
        with col2:
            st.metric("Total Users", len(interactions["user_id"].unique()))
        with col3:
            st.metric("Total Interactions", len(interactions))
        with col4:
            st.metric("Avg Interactions/User", 
                     f"{len(interactions) / len(interactions['user_id'].unique()):.1f}")
        
        # Display model information
        st.subheader("🤖 Available Models")
        
        model_info = {
            "TF-IDF": "Content-based filtering using TF-IDF vectorization and cosine similarity",
            "SBERT": "Content-based filtering using Sentence-BERT embeddings for semantic similarity",
            "Hybrid": "Combines TF-IDF and SBERT models using weighted voting"
        }
        
        for model_name, description in model_info.items():
            st.markdown(f"""
            <div class="metric-card">
                <h4>{model_name}</h4>
                <p>{description}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Category distribution
        if "category" in items.columns:
            st.subheader("📋 Article Categories")
            category_counts = items["category"].value_counts()
            st.bar_chart(category_counts)
    
    elif page == "About":
        st.header("ℹ️ About This System")
        
        st.markdown("""
        ## News Recommendation System
        
        This is a modern content-based news recommendation system that uses advanced NLP techniques 
        to recommend relevant news articles to users based on their reading preferences.
        
        ### Features
        
        - **Multiple Models**: TF-IDF, Sentence-BERT, and Hybrid approaches
        - **Content-Based Filtering**: Recommendations based on article content similarity
        - **Interactive Demo**: User-friendly interface for exploring recommendations
        - **Real-time Recommendations**: Generate personalized recommendations instantly
        
        ### Models
        
        1. **TF-IDF Model**: Uses Term Frequency-Inverse Document Frequency to convert 
           article text into numerical features and computes cosine similarity.
        
        2. **SBERT Model**: Uses Sentence-BERT embeddings to capture semantic similarity 
           between articles for more nuanced recommendations.
        
        3. **Hybrid Model**: Combines both TF-IDF and SBERT models using weighted voting 
           to leverage the strengths of both approaches.
        
        ### How It Works
        
        1. **Content Analysis**: Each article is processed to extract meaningful features
        2. **User Profiling**: User preferences are inferred from their reading history
        3. **Similarity Computation**: Articles are compared using various similarity metrics
        4. **Recommendation Generation**: Top similar articles are recommended to users
        
        ### Technical Stack
        
        - **Python 3.10+**: Core programming language
        - **scikit-learn**: TF-IDF vectorization and similarity computation
        - **Sentence-Transformers**: Pre-trained BERT models for semantic embeddings
        - **Streamlit**: Interactive web interface
        - **Pandas & NumPy**: Data manipulation and numerical computations
        
        ### Usage
        
        - **User Recommendations**: Select a user and get personalized article recommendations
        - **Item Similarity**: Find articles similar to a selected article
        - **Model Comparison**: Compare different recommendation approaches
        
        This system demonstrates modern recommendation techniques and provides a foundation 
        for building production-ready news recommendation systems.
        """)


if __name__ == "__main__":
    main()
