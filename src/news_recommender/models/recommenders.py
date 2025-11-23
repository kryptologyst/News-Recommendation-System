"""News recommendation models including TF-IDF, SBERT, and hybrid approaches."""

import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


class BaseRecommender(ABC):
    """Abstract base class for recommendation models."""
    
    def __init__(self, name: str):
        """Initialize the recommender.
        
        Args:
            name: Name of the recommender model
        """
        self.name = name
        self.is_fitted = False
        
    @abstractmethod
    def fit(self, items: pd.DataFrame, interactions: Optional[pd.DataFrame] = None) -> None:
        """Fit the recommendation model.
        
        Args:
            items: DataFrame with item information
            interactions: Optional DataFrame with user-item interactions
        """
        pass
        
    @abstractmethod
    def recommend(
        self, 
        user_id: int, 
        n_recommendations: int = 10,
        exclude_seen: bool = True,
        interactions: Optional[pd.DataFrame] = None
    ) -> List[int]:
        """Generate recommendations for a user.
        
        Args:
            user_id: ID of the user
            n_recommendations: Number of recommendations to generate
            exclude_seen: Whether to exclude already seen items
            interactions: Optional DataFrame with user-item interactions
            
        Returns:
            List of recommended item IDs
        """
        pass
        
    def get_similar_items(
        self, 
        item_id: int, 
        n_similar: int = 10
    ) -> List[Tuple[int, float]]:
        """Get items similar to the given item.
        
        Args:
            item_id: ID of the item
            n_similar: Number of similar items to return
            
        Returns:
            List of tuples (item_id, similarity_score)
        """
        raise NotImplementedError("Similar items not implemented for this model")


class TFIDFRecommender(BaseRecommender):
    """TF-IDF based content recommendation model."""
    
    def __init__(
        self, 
        max_features: int = 10000,
        ngram_range: Tuple[int, int] = (1, 2),
        min_df: int = 2,
        max_df: float = 0.95
    ):
        """Initialize TF-IDF recommender.
        
        Args:
            max_features: Maximum number of features
            ngram_range: Range of n-grams to consider
            min_df: Minimum document frequency
            max_df: Maximum document frequency
        """
        super().__init__("TF-IDF")
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            stop_words='english'
        )
        self.tfidf_matrix = None
        self.items = None
        
    def fit(self, items: pd.DataFrame, interactions: Optional[pd.DataFrame] = None) -> None:
        """Fit the TF-IDF model.
        
        Args:
            items: DataFrame with item information
            interactions: Optional DataFrame with user-item interactions (not used)
        """
        logger.info(f"Fitting {self.name} model...")
        
        # Combine title and content for better representation
        texts = items["title"] + " " + items["content"]
        
        # Fit TF-IDF vectorizer
        self.tfidf_matrix = self.vectorizer.fit_transform(texts)
        self.items = items.copy()
        self.is_fitted = True
        
        logger.info(f"TF-IDF model fitted with {self.tfidf_matrix.shape[1]} features")
        
    def recommend(
        self, 
        user_id: int, 
        n_recommendations: int = 10,
        exclude_seen: bool = True,
        interactions: Optional[pd.DataFrame] = None
    ) -> List[int]:
        """Generate recommendations using TF-IDF similarity.
        
        Args:
            user_id: ID of the user
            n_recommendations: Number of recommendations to generate
            exclude_seen: Whether to exclude already seen items
            interactions: Optional DataFrame with user-item interactions
            
        Returns:
            List of recommended item IDs
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making recommendations")
            
        # Get user's interaction history
        seen_items = set()
        if interactions is not None and exclude_seen:
            user_interactions = interactions[interactions["user_id"] == user_id]
            seen_items = set(user_interactions["item_id"].tolist())
            
        # If no interactions, recommend popular items
        if not seen_items:
            return self._recommend_popular_items(n_recommendations)
            
        # Get user's preferred content by averaging TF-IDF vectors of seen items
        user_vector = self._get_user_profile(user_id, interactions)
        
        # Compute similarities
        similarities = cosine_similarity(user_vector, self.tfidf_matrix).flatten()
        
        # Get recommendations
        recommendations = self._get_top_items(
            similarities, 
            n_recommendations, 
            exclude_items=seen_items
        )
        
        return recommendations
        
    def get_similar_items(
        self, 
        item_id: int, 
        n_similar: int = 10
    ) -> List[Tuple[int, float]]:
        """Get items similar to the given item using TF-IDF."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting similar items")
            
        # Find item index
        item_idx = self.items[self.items["item_id"] == item_id].index[0]
        
        # Compute similarities
        similarities = cosine_similarity(
            self.tfidf_matrix[item_idx:item_idx+1], 
            self.tfidf_matrix
        ).flatten()
        
        # Get top similar items (excluding the item itself)
        similar_indices = similarities.argsort()[-n_similar-1:-1][::-1]
        
        results = []
        for idx in similar_indices:
            item_id_similar = self.items.iloc[idx]["item_id"]
            similarity_score = similarities[idx]
            results.append((item_id_similar, similarity_score))
            
        return results
        
    def _get_user_profile(
        self, 
        user_id: int, 
        interactions: Optional[pd.DataFrame]
    ) -> np.ndarray:
        """Get user profile by averaging TF-IDF vectors of seen items."""
        if interactions is None:
            return np.zeros((1, self.tfidf_matrix.shape[1]))
            
        user_interactions = interactions[interactions["user_id"] == user_id]
        if len(user_interactions) == 0:
            return np.zeros((1, self.tfidf_matrix.shape[1]))
            
        # Get TF-IDF vectors for user's items
        user_item_ids = user_interactions["item_id"].tolist()
        user_item_indices = []
        
        for item_id in user_item_ids:
            item_idx = self.items[self.items["item_id"] == item_id].index[0]
            user_item_indices.append(item_idx)
            
        # Average the vectors
        user_vectors = self.tfidf_matrix[user_item_indices]
        user_profile = np.mean(user_vectors, axis=0)
        
        return user_profile.reshape(1, -1)
        
    def _recommend_popular_items(self, n_recommendations: int) -> List[int]:
        """Recommend popular items when no user history is available."""
        # Simple popularity based on item ID (in real scenario, use actual popularity)
        popular_items = self.items["item_id"].head(n_recommendations).tolist()
        return popular_items
        
    def _get_top_items(
        self, 
        similarities: np.ndarray, 
        n_recommendations: int,
        exclude_items: Optional[set] = None
    ) -> List[int]:
        """Get top items based on similarity scores."""
        if exclude_items:
            # Set similarity to -1 for excluded items
            for item_id in exclude_items:
                item_idx = self.items[self.items["item_id"] == item_id].index[0]
                similarities[item_idx] = -1
                
        # Get top recommendations
        top_indices = similarities.argsort()[-n_recommendations:][::-1]
        recommendations = [self.items.iloc[idx]["item_id"] for idx in top_indices]
        
        return recommendations


class SBERTRecommender(BaseRecommender):
    """Sentence-BERT based content recommendation model."""
    
    def __init__(
        self, 
        model_name: str = "all-MiniLM-L6-v2",
        batch_size: int = 32
    ):
        """Initialize SBERT recommender.
        
        Args:
            model_name: Name of the SBERT model
            batch_size: Batch size for encoding
        """
        super().__init__("SBERT")
        self.model = SentenceTransformer(model_name)
        self.batch_size = batch_size
        self.embeddings = None
        self.items = None
        
    def fit(self, items: pd.DataFrame, interactions: Optional[pd.DataFrame] = None) -> None:
        """Fit the SBERT model.
        
        Args:
            items: DataFrame with item information
            interactions: Optional DataFrame with user-item interactions (not used)
        """
        logger.info(f"Fitting {self.name} model...")
        
        # Combine title and content
        texts = items["title"] + " " + items["content"]
        
        # Generate embeddings
        self.embeddings = self.model.encode(
            texts.tolist(), 
            batch_size=self.batch_size,
            show_progress_bar=True
        )
        self.items = items.copy()
        self.is_fitted = True
        
        logger.info(f"SBERT model fitted with embeddings shape: {self.embeddings.shape}")
        
    def recommend(
        self, 
        user_id: int, 
        n_recommendations: int = 10,
        exclude_seen: bool = True,
        interactions: Optional[pd.DataFrame] = None
    ) -> List[int]:
        """Generate recommendations using SBERT embeddings."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making recommendations")
            
        # Get user's interaction history
        seen_items = set()
        if interactions is not None and exclude_seen:
            user_interactions = interactions[interactions["user_id"] == user_id]
            seen_items = set(user_interactions["item_id"].tolist())
            
        # If no interactions, recommend popular items
        if not seen_items:
            return self._recommend_popular_items(n_recommendations)
            
        # Get user's preferred content by averaging embeddings of seen items
        user_embedding = self._get_user_embedding(user_id, interactions)
        
        # Compute similarities
        similarities = cosine_similarity(user_embedding, self.embeddings).flatten()
        
        # Get recommendations
        recommendations = self._get_top_items(
            similarities, 
            n_recommendations, 
            exclude_items=seen_items
        )
        
        return recommendations
        
    def get_similar_items(
        self, 
        item_id: int, 
        n_similar: int = 10
    ) -> List[Tuple[int, float]]:
        """Get items similar to the given item using SBERT embeddings."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting similar items")
            
        # Find item index
        item_idx = self.items[self.items["item_id"] == item_id].index[0]
        
        # Compute similarities
        similarities = cosine_similarity(
            self.embeddings[item_idx:item_idx+1], 
            self.embeddings
        ).flatten()
        
        # Get top similar items (excluding the item itself)
        similar_indices = similarities.argsort()[-n_similar-1:-1][::-1]
        
        results = []
        for idx in similar_indices:
            item_id_similar = self.items.iloc[idx]["item_id"]
            similarity_score = similarities[idx]
            results.append((item_id_similar, similarity_score))
            
        return results
        
    def _get_user_embedding(
        self, 
        user_id: int, 
        interactions: Optional[pd.DataFrame]
    ) -> np.ndarray:
        """Get user embedding by averaging embeddings of seen items."""
        if interactions is None:
            return np.zeros((1, self.embeddings.shape[1]))
            
        user_interactions = interactions[interactions["user_id"] == user_id]
        if len(user_interactions) == 0:
            return np.zeros((1, self.embeddings.shape[1]))
            
        # Get embeddings for user's items
        user_item_ids = user_interactions["item_id"].tolist()
        user_item_indices = []
        
        for item_id in user_item_ids:
            item_idx = self.items[self.items["item_id"] == item_id].index[0]
            user_item_indices.append(item_idx)
            
        # Average the embeddings
        user_embeddings = self.embeddings[user_item_indices]
        user_embedding = np.mean(user_embeddings, axis=0)
        
        return user_embedding.reshape(1, -1)
        
    def _recommend_popular_items(self, n_recommendations: int) -> List[int]:
        """Recommend popular items when no user history is available."""
        popular_items = self.items["item_id"].head(n_recommendations).tolist()
        return popular_items
        
    def _get_top_items(
        self, 
        similarities: np.ndarray, 
        n_recommendations: int,
        exclude_items: Optional[set] = None
    ) -> List[int]:
        """Get top items based on similarity scores."""
        if exclude_items:
            # Set similarity to -1 for excluded items
            for item_id in exclude_items:
                item_idx = self.items[self.items["item_id"] == item_id].index[0]
                similarities[item_idx] = -1
                
        # Get top recommendations
        top_indices = similarities.argsort()[-n_recommendations:][::-1]
        recommendations = [self.items.iloc[idx]["item_id"] for idx in top_indices]
        
        return recommendations


class HybridRecommender(BaseRecommender):
    """Hybrid recommender combining TF-IDF and SBERT models."""
    
    def __init__(
        self, 
        tfidf_weight: float = 0.6,
        sbert_weight: float = 0.4,
        tfidf_params: Optional[Dict] = None,
        sbert_params: Optional[Dict] = None
    ):
        """Initialize hybrid recommender.
        
        Args:
            tfidf_weight: Weight for TF-IDF model
            sbert_weight: Weight for SBERT model
            tfidf_params: Parameters for TF-IDF model
            sbert_params: Parameters for SBERT model
        """
        super().__init__("Hybrid")
        self.tfidf_weight = tfidf_weight
        self.sbert_weight = sbert_weight
        
        # Initialize component models
        tfidf_params = tfidf_params or {}
        sbert_params = sbert_params or {}
        
        self.tfidf_model = TFIDFRecommender(**tfidf_params)
        self.sbert_model = SBERTRecommender(**sbert_params)
        
    def fit(self, items: pd.DataFrame, interactions: Optional[pd.DataFrame] = None) -> None:
        """Fit both component models.
        
        Args:
            items: DataFrame with item information
            interactions: Optional DataFrame with user-item interactions
        """
        logger.info(f"Fitting {self.name} model...")
        
        # Fit both models
        self.tfidf_model.fit(items, interactions)
        self.sbert_model.fit(items, interactions)
        
        self.is_fitted = True
        logger.info("Hybrid model fitted successfully")
        
    def recommend(
        self, 
        user_id: int, 
        n_recommendations: int = 10,
        exclude_seen: bool = True,
        interactions: Optional[pd.DataFrame] = None
    ) -> List[int]:
        """Generate recommendations using hybrid approach."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making recommendations")
            
        # Get recommendations from both models
        tfidf_recs = self.tfidf_model.recommend(
            user_id, n_recommendations * 2, exclude_seen, interactions
        )
        sbert_recs = self.sbert_model.recommend(
            user_id, n_recommendations * 2, exclude_seen, interactions
        )
        
        # Combine recommendations using weighted voting
        item_scores = {}
        
        # Add TF-IDF scores
        for i, item_id in enumerate(tfidf_recs):
            score = self.tfidf_weight * (len(tfidf_recs) - i) / len(tfidf_recs)
            item_scores[item_id] = item_scores.get(item_id, 0) + score
            
        # Add SBERT scores
        for i, item_id in enumerate(sbert_recs):
            score = self.sbert_weight * (len(sbert_recs) - i) / len(sbert_recs)
            item_scores[item_id] = item_scores.get(item_id, 0) + score
            
        # Sort by combined score and return top recommendations
        sorted_items = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)
        recommendations = [item_id for item_id, _ in sorted_items[:n_recommendations]]
        
        return recommendations
        
    def get_similar_items(
        self, 
        item_id: int, 
        n_similar: int = 10
    ) -> List[Tuple[int, float]]:
        """Get similar items using hybrid approach."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting similar items")
            
        # Get similar items from both models
        tfidf_similar = self.tfidf_model.get_similar_items(item_id, n_similar)
        sbert_similar = self.sbert_model.get_similar_items(item_id, n_similar)
        
        # Combine similarities
        item_scores = {}
        
        # Add TF-IDF similarities
        for item_id_sim, score in tfidf_similar:
            item_scores[item_id_sim] = item_scores.get(item_id_sim, 0) + self.tfidf_weight * score
            
        # Add SBERT similarities
        for item_id_sim, score in sbert_similar:
            item_scores[item_id_sim] = item_scores.get(item_id_sim, 0) + self.sbert_weight * score
            
        # Sort by combined score
        sorted_items = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)
        
        return sorted_items[:n_similar]
