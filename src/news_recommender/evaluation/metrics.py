"""Evaluation metrics and utilities for news recommendation system."""

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score

logger = logging.getLogger(__name__)


class RecommendationEvaluator:
    """Evaluator for recommendation models."""
    
    def __init__(self, k_values: List[int] = [5, 10, 20]):
        """Initialize the evaluator.
        
        Args:
            k_values: List of k values for evaluation metrics
        """
        self.k_values = k_values
        
    def evaluate_model(
        self,
        model,
        test_interactions: pd.DataFrame,
        items: pd.DataFrame,
        n_recommendations: int = 10
    ) -> Dict[str, float]:
        """Evaluate a recommendation model.
        
        Args:
            model: Trained recommendation model
            test_interactions: Test set interactions
            items: Item information
            n_recommendations: Number of recommendations per user
            
        Returns:
            Dictionary of evaluation metrics
        """
        logger.info(f"Evaluating {model.name} model...")
        
        metrics = {}
        
        # Get all unique users in test set
        test_users = test_interactions["user_id"].unique()
        
        # Calculate metrics for each k value
        for k in self.k_values:
            precision_scores = []
            recall_scores = []
            ndcg_scores = []
            hit_rates = []
            
            for user_id in test_users:
                # Get user's test items
                user_test_items = set(
                    test_interactions[test_interactions["user_id"] == user_id]["item_id"]
                )
                
                if len(user_test_items) == 0:
                    continue
                    
                # Get recommendations
                recommendations = model.recommend(
                    user_id, 
                    n_recommendations=n_recommendations,
                    exclude_seen=True,
                    interactions=None  # Don't use test interactions for training
                )
                
                # Calculate metrics
                precision = self._precision_at_k(recommendations, user_test_items, k)
                recall = self._recall_at_k(recommendations, user_test_items, k)
                ndcg = self._ndcg_at_k(recommendations, user_test_items, k)
                hit_rate = self._hit_rate_at_k(recommendations, user_test_items, k)
                
                precision_scores.append(precision)
                recall_scores.append(recall)
                ndcg_scores.append(ndcg)
                hit_rates.append(hit_rate)
                
            # Average metrics across users
            metrics[f"precision@{k}"] = np.mean(precision_scores)
            metrics[f"recall@{k}"] = np.mean(recall_scores)
            metrics[f"ndcg@{k}"] = np.mean(ndcg_scores)
            metrics[f"hit_rate@{k}"] = np.mean(hit_rates)
            
        # Calculate additional metrics
        metrics["coverage"] = self._calculate_coverage(model, test_interactions, items)
        metrics["novelty"] = self._calculate_novelty(model, test_interactions, items)
        
        logger.info(f"Evaluation completed for {model.name}")
        return metrics
        
    def _precision_at_k(
        self, 
        recommendations: List[int], 
        relevant_items: set, 
        k: int
    ) -> float:
        """Calculate Precision@K."""
        if k == 0:
            return 0.0
            
        top_k = recommendations[:k]
        relevant_in_top_k = len(set(top_k) & relevant_items)
        
        return relevant_in_top_k / k
        
    def _recall_at_k(
        self, 
        recommendations: List[int], 
        relevant_items: set, 
        k: int
    ) -> float:
        """Calculate Recall@K."""
        if len(relevant_items) == 0:
            return 0.0
            
        top_k = recommendations[:k]
        relevant_in_top_k = len(set(top_k) & relevant_items)
        
        return relevant_in_top_k / len(relevant_items)
        
    def _ndcg_at_k(
        self, 
        recommendations: List[int], 
        relevant_items: set, 
        k: int
    ) -> float:
        """Calculate NDCG@K."""
        if k == 0:
            return 0.0
            
        top_k = recommendations[:k]
        
        # Calculate DCG
        dcg = 0.0
        for i, item in enumerate(top_k):
            if item in relevant_items:
                dcg += 1.0 / np.log2(i + 2)  # i+2 because log2(1) = 0
                
        # Calculate IDCG (ideal DCG)
        idcg = 0.0
        for i in range(min(k, len(relevant_items))):
            idcg += 1.0 / np.log2(i + 2)
            
        if idcg == 0:
            return 0.0
            
        return dcg / idcg
        
    def _hit_rate_at_k(
        self, 
        recommendations: List[int], 
        relevant_items: set, 
        k: int
    ) -> float:
        """Calculate Hit Rate@K."""
        if k == 0:
            return 0.0
            
        top_k = recommendations[:k]
        return 1.0 if len(set(top_k) & relevant_items) > 0 else 0.0
        
    def _calculate_coverage(
        self, 
        model, 
        test_interactions: pd.DataFrame, 
        items: pd.DataFrame
    ) -> float:
        """Calculate catalog coverage."""
        test_users = test_interactions["user_id"].unique()
        all_recommended_items = set()
        
        for user_id in test_users:
            recommendations = model.recommend(user_id, n_recommendations=10)
            all_recommended_items.update(recommendations)
            
        total_items = len(items)
        covered_items = len(all_recommended_items)
        
        return covered_items / total_items if total_items > 0 else 0.0
        
    def _calculate_novelty(
        self, 
        model, 
        test_interactions: pd.DataFrame, 
        items: pd.DataFrame
    ) -> float:
        """Calculate novelty (average popularity of recommended items)."""
        # Calculate item popularity based on interactions
        item_popularity = test_interactions["item_id"].value_counts()
        total_interactions = len(test_interactions)
        
        test_users = test_interactions["user_id"].unique()
        novelty_scores = []
        
        for user_id in test_users:
            recommendations = model.recommend(user_id, n_recommendations=10)
            
            user_novelty = 0.0
            for item_id in recommendations:
                popularity = item_popularity.get(item_id, 0) / total_interactions
                user_novelty += -np.log2(popularity + 1e-8)  # Add small epsilon to avoid log(0)
                
            if len(recommendations) > 0:
                user_novelty /= len(recommendations)
                novelty_scores.append(user_novelty)
                
        return np.mean(novelty_scores) if novelty_scores else 0.0
        
    def compare_models(
        self, 
        models: List, 
        test_interactions: pd.DataFrame, 
        items: pd.DataFrame
    ) -> pd.DataFrame:
        """Compare multiple models and return results as DataFrame.
        
        Args:
            models: List of trained models
            test_interactions: Test set interactions
            items: Item information
            
        Returns:
            DataFrame with comparison results
        """
        results = []
        
        for model in models:
            logger.info(f"Evaluating {model.name} model...")
            metrics = self.evaluate_model(model, test_interactions, items)
            metrics["model"] = model.name
            results.append(metrics)
            
        # Convert to DataFrame
        df = pd.DataFrame(results)
        
        # Reorder columns to put model name first
        cols = ["model"] + [col for col in df.columns if col != "model"]
        df = df[cols]
        
        return df
        
    def print_results(self, results_df: pd.DataFrame) -> None:
        """Print evaluation results in a formatted table."""
        print("\n" + "="*80)
        print("RECOMMENDATION MODEL EVALUATION RESULTS")
        print("="*80)
        
        # Round numeric columns
        numeric_cols = results_df.select_dtypes(include=[np.number]).columns
        results_df[numeric_cols] = results_df[numeric_cols].round(4)
        
        print(results_df.to_string(index=False))
        print("="*80)
        
        # Print best model for each metric
        print("\nBEST MODELS BY METRIC:")
        print("-" * 40)
        
        for col in numeric_cols:
            if col != "model":
                best_idx = results_df[col].idxmax()
                best_model = results_df.loc[best_idx, "model"]
                best_score = results_df.loc[best_idx, col]
                print(f"{col:20}: {best_model:15} ({best_score:.4f})")
                
        print("-" * 40)
