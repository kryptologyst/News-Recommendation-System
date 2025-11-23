#!/usr/bin/env python3
"""Main training script for news recommendation system."""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd
import yaml

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from news_recommender.data.loader import NewsDataLoader
from news_recommender.evaluation.metrics import RecommendationEvaluator
from news_recommender.models.recommenders import (
    TFIDFRecommender,
    SBERTRecommender,
    HybridRecommender
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def train_models(
    items: pd.DataFrame,
    train_interactions: pd.DataFrame,
    config: Dict
) -> List:
    """Train all recommendation models.
    
    Args:
        items: Item information
        train_interactions: Training interactions
        config: Configuration dictionary
        
    Returns:
        List of trained models
    """
    models = []
    
    # TF-IDF Model
    logger.info("Training TF-IDF model...")
    tfidf_model = TFIDFRecommender(
        max_features=config["models"]["tfidf"]["max_features"],
        ngram_range=tuple(config["models"]["tfidf"]["ngram_range"]),
        min_df=config["models"]["tfidf"]["min_df"],
        max_df=config["models"]["tfidf"]["max_df"]
    )
    tfidf_model.fit(items, train_interactions)
    models.append(tfidf_model)
    
    # SBERT Model
    logger.info("Training SBERT model...")
    sbert_model = SBERTRecommender(
        model_name=config["models"]["sbert"]["model_name"],
        batch_size=config["models"]["sbert"]["batch_size"]
    )
    sbert_model.fit(items, train_interactions)
    models.append(sbert_model)
    
    # Hybrid Model
    logger.info("Training Hybrid model...")
    hybrid_model = HybridRecommender(
        tfidf_weight=config["models"]["hybrid"]["tfidf_weight"],
        sbert_weight=config["models"]["hybrid"]["sbert_weight"]
    )
    hybrid_model.fit(items, train_interactions)
    models.append(hybrid_model)
    
    logger.info("All models trained successfully!")
    return models


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train news recommendation models")
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--data-dir", 
        type=str, 
        default="data",
        help="Directory containing data files"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="models",
        help="Directory to save trained models"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Initialize data loader
    data_loader = NewsDataLoader(
        data_dir=args.data_dir,
        random_seed=config["data"]["random_seed"]
    )
    
    # Load data
    logger.info("Loading data...")
    interactions = data_loader.load_interactions()
    items = data_loader.load_items()
    users = data_loader.load_users()
    
    # Split data
    logger.info("Splitting data...")
    train_interactions, val_interactions, test_interactions = data_loader.split_data(
        interactions,
        test_size=config["data"]["test_size"],
        val_size=config["data"]["val_size"]
    )
    
    # Train models
    logger.info("Training models...")
    models = train_models(items, train_interactions, config)
    
    # Evaluate models
    logger.info("Evaluating models...")
    evaluator = RecommendationEvaluator(
        k_values=config["evaluation"]["k_values"]
    )
    
    results_df = evaluator.compare_models(models, test_interactions, items)
    
    # Print results
    evaluator.print_results(results_df)
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    results_df.to_csv(output_dir / "evaluation_results.csv", index=False)
    logger.info(f"Results saved to {output_dir / 'evaluation_results.csv'}")
    
    # Save models (simplified - in production, use proper serialization)
    for model in models:
        model_file = output_dir / f"{model.name.lower().replace('-', '_')}_model.pkl"
        # Note: In production, implement proper model serialization
        logger.info(f"Model {model.name} would be saved to {model_file}")
    
    logger.info("Training completed successfully!")


if __name__ == "__main__":
    main()
