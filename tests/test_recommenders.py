"""Tests for news recommendation system."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from news_recommender.data.loader import NewsDataLoader
from news_recommender.models.recommenders import TFIDFRecommender, SBERTRecommender, HybridRecommender


class TestDataLoader:
    """Test cases for NewsDataLoader."""
    
    def test_init(self):
        """Test DataLoader initialization."""
        loader = NewsDataLoader("data", random_seed=42)
        assert loader.data_dir == Path("data")
        assert loader.random_seed == 42
        
    def test_create_synthetic_items(self):
        """Test synthetic item creation."""
        loader = NewsDataLoader("test_data", random_seed=42)
        items = loader._create_synthetic_items()
        
        assert isinstance(items, pd.DataFrame)
        assert len(items) == 500
        assert "item_id" in items.columns
        assert "title" in items.columns
        assert "content" in items.columns
        
    def test_create_synthetic_interactions(self):
        """Test synthetic interaction creation."""
        loader = NewsDataLoader("test_data", random_seed=42)
        interactions = loader._create_synthetic_interactions()
        
        assert isinstance(interactions, pd.DataFrame)
        assert len(interactions) > 0
        assert "user_id" in interactions.columns
        assert "item_id" in interactions.columns
        assert "timestamp" in interactions.columns
        assert "weight" in interactions.columns
        
    def test_split_data(self):
        """Test data splitting."""
        loader = NewsDataLoader("test_data", random_seed=42)
        interactions = loader._create_synthetic_interactions()
        
        train, val, test = loader.split_data(interactions, test_size=0.2, val_size=0.1)
        
        assert len(train) + len(val) + len(test) == len(interactions)
        assert len(train) > len(val)
        assert len(train) > len(test)


class TestTFIDFRecommender:
    """Test cases for TFIDFRecommender."""
    
    def test_init(self):
        """Test TFIDFRecommender initialization."""
        model = TFIDFRecommender()
        assert model.name == "TF-IDF"
        assert not model.is_fitted
        
    def test_fit(self):
        """Test model fitting."""
        model = TFIDFRecommender()
        items = pd.DataFrame({
            "item_id": [1, 2, 3],
            "title": ["Title 1", "Title 2", "Title 3"],
            "content": ["Content 1", "Content 2", "Content 3"]
        })
        
        model.fit(items)
        assert model.is_fitted
        assert model.tfidf_matrix is not None
        
    def test_recommend(self):
        """Test recommendation generation."""
        model = TFIDFRecommender()
        items = pd.DataFrame({
            "item_id": [1, 2, 3, 4, 5],
            "title": ["Title 1", "Title 2", "Title 3", "Title 4", "Title 5"],
            "content": ["Content 1", "Content 2", "Content 3", "Content 4", "Content 5"]
        })
        
        model.fit(items)
        recommendations = model.recommend(1, n_recommendations=3)
        
        assert isinstance(recommendations, list)
        assert len(recommendations) <= 3
        assert all(isinstance(item_id, int) for item_id in recommendations)
        
    def test_get_similar_items(self):
        """Test similar item retrieval."""
        model = TFIDFRecommender()
        items = pd.DataFrame({
            "item_id": [1, 2, 3, 4, 5],
            "title": ["Title 1", "Title 2", "Title 3", "Title 4", "Title 5"],
            "content": ["Content 1", "Content 2", "Content 3", "Content 4", "Content 5"]
        })
        
        model.fit(items)
        similar_items = model.get_similar_items(1, n_similar=3)
        
        assert isinstance(similar_items, list)
        assert len(similar_items) <= 3
        assert all(isinstance(item, tuple) and len(item) == 2 for item in similar_items)


class TestSBERTRecommender:
    """Test cases for SBERTRecommender."""
    
    def test_init(self):
        """Test SBERTRecommender initialization."""
        model = SBERTRecommender()
        assert model.name == "SBERT"
        assert not model.is_fitted
        
    def test_fit(self):
        """Test model fitting."""
        model = SBERTRecommender()
        items = pd.DataFrame({
            "item_id": [1, 2, 3],
            "title": ["Title 1", "Title 2", "Title 3"],
            "content": ["Content 1", "Content 2", "Content 3"]
        })
        
        model.fit(items)
        assert model.is_fitted
        assert model.embeddings is not None


class TestHybridRecommender:
    """Test cases for HybridRecommender."""
    
    def test_init(self):
        """Test HybridRecommender initialization."""
        model = HybridRecommender()
        assert model.name == "Hybrid"
        assert not model.is_fitted
        
    def test_fit(self):
        """Test model fitting."""
        model = HybridRecommender()
        items = pd.DataFrame({
            "item_id": [1, 2, 3],
            "title": ["Title 1", "Title 2", "Title 3"],
            "content": ["Content 1", "Content 2", "Content 3"]
        })
        
        model.fit(items)
        assert model.is_fitted
        assert model.tfidf_model.is_fitted
        assert model.sbert_model.is_fitted


if __name__ == "__main__":
    pytest.main([__file__])
