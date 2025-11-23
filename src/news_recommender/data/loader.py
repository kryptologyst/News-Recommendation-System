"""Data loading and preprocessing utilities for news recommendation system."""

import logging
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NewsDataLoader:
    """Data loader for news recommendation system."""
    
    def __init__(self, data_dir: Union[str, Path], random_seed: int = 42):
        """Initialize the data loader.
        
        Args:
            data_dir: Directory containing the data files
            random_seed: Random seed for reproducibility
        """
        self.data_dir = Path(data_dir)
        self.random_seed = random_seed
        self._set_seeds()
        
    def _set_seeds(self) -> None:
        """Set random seeds for reproducibility."""
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        
    def load_interactions(self, filename: str = "interactions.csv") -> pd.DataFrame:
        """Load user-item interactions.
        
        Args:
            filename: Name of the interactions file
            
        Returns:
            DataFrame with columns: user_id, item_id, timestamp, weight
        """
        filepath = self.data_dir / filename
        if not filepath.exists():
            logger.warning(f"Interactions file {filepath} not found. Creating synthetic data.")
            return self._create_synthetic_interactions()
            
        df = pd.read_csv(filepath)
        required_cols = ["user_id", "item_id", "timestamp", "weight"]
        
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"Interactions file must contain columns: {required_cols}")
            
        logger.info(f"Loaded {len(df)} interactions from {filepath}")
        return df
        
    def load_items(self, filename: str = "items.csv") -> pd.DataFrame:
        """Load news articles/items.
        
        Args:
            filename: Name of the items file
            
        Returns:
            DataFrame with columns: item_id, title, content, category, tags
        """
        filepath = self.data_dir / filename
        if not filepath.exists():
            logger.warning(f"Items file {filepath} not found. Creating synthetic data.")
            return self._create_synthetic_items()
            
        df = pd.read_csv(filepath)
        required_cols = ["item_id", "title", "content"]
        
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"Items file must contain columns: {required_cols}")
            
        logger.info(f"Loaded {len(df)} items from {filepath}")
        return df
        
    def load_users(self, filename: str = "users.csv") -> Optional[pd.DataFrame]:
        """Load user information.
        
        Args:
            filename: Name of the users file
            
        Returns:
            DataFrame with user information or None if file doesn't exist
        """
        filepath = self.data_dir / filename
        if not filepath.exists():
            logger.info(f"Users file {filepath} not found. Skipping user data.")
            return None
            
        df = pd.read_csv(filepath)
        logger.info(f"Loaded {len(df)} users from {filepath}")
        return df
        
    def _create_synthetic_interactions(self) -> pd.DataFrame:
        """Create synthetic interaction data for demonstration."""
        logger.info("Creating synthetic interaction data...")
        
        # Create synthetic data
        n_users = 1000
        n_items = 500
        n_interactions = 10000
        
        # Generate interactions with some patterns
        interactions = []
        
        # Popular items get more interactions
        item_popularity = np.random.power(2, n_items)
        item_popularity = item_popularity / item_popularity.sum()
        
        for _ in range(n_interactions):
            user_id = random.randint(1, n_users)
            item_id = np.random.choice(n_items, p=item_popularity) + 1
            
            # Add some temporal patterns
            timestamp = pd.Timestamp.now() - pd.Timedelta(
                days=random.randint(0, 30),
                hours=random.randint(0, 23),
                minutes=random.randint(0, 59)
            )
            
            # Weight based on interaction type (1.0 for read, 0.5 for click)
            weight = random.choice([1.0, 0.5])
            
            interactions.append({
                "user_id": user_id,
                "item_id": item_id,
                "timestamp": timestamp,
                "weight": weight
            })
            
        df = pd.DataFrame(interactions)
        df = df.drop_duplicates(subset=["user_id", "item_id"])
        
        # Save for future use
        self.data_dir.mkdir(exist_ok=True)
        df.to_csv(self.data_dir / "interactions.csv", index=False)
        
        logger.info(f"Created {len(df)} synthetic interactions")
        return df
        
    def _create_synthetic_items(self) -> pd.DataFrame:
        """Create synthetic news articles for demonstration."""
        logger.info("Creating synthetic news articles...")
        
        # News categories and sample content
        categories = ["Politics", "Sports", "Technology", "Entertainment", "Health", "Business", "Science"]
        
        sample_content = [
            "Breaking news: Major developments in international relations",
            "Sports update: Championship results and player transfers",
            "Technology breakthrough: New innovations in artificial intelligence",
            "Entertainment news: Latest movie releases and celebrity updates",
            "Health update: New medical research and treatment options",
            "Business news: Market trends and economic analysis",
            "Science discovery: Research findings and scientific breakthroughs",
            "Political analysis: Government policies and legislative changes",
            "Sports commentary: Game analysis and team performance",
            "Tech review: Product launches and industry trends"
        ]
        
        items = []
        for i in range(1, 501):  # 500 articles
            category = random.choice(categories)
            base_content = random.choice(sample_content)
            
            # Add some variation to content
            variations = [
                f"{base_content} with significant implications for the future.",
                f"Recent developments show that {base_content.lower()}",
                f"Experts suggest that {base_content.lower()}",
                f"New research indicates {base_content.lower()}",
                f"Analysis reveals {base_content.lower()}"
            ]
            
            content = random.choice(variations)
            title = f"{category} News Article {i}"
            
            # Generate tags
            tags = f"{category.lower()},news,article{i}"
            
            items.append({
                "item_id": i,
                "title": title,
                "content": content,
                "category": category,
                "tags": tags,
                "publish_date": pd.Timestamp.now() - pd.Timedelta(days=random.randint(0, 30))
            })
            
        df = pd.DataFrame(items)
        
        # Save for future use
        self.data_dir.mkdir(exist_ok=True)
        df.to_csv(self.data_dir / "items.csv", index=False)
        
        logger.info(f"Created {len(df)} synthetic news articles")
        return df
        
    def split_data(
        self, 
        interactions: pd.DataFrame, 
        test_size: float = 0.2, 
        val_size: float = 0.1
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split interactions into train, validation, and test sets.
        
        Args:
            interactions: DataFrame with user-item interactions
            test_size: Proportion of data for testing
            val_size: Proportion of data for validation
            
        Returns:
            Tuple of (train, validation, test) DataFrames
        """
        # Sort by timestamp for temporal splitting
        interactions = interactions.sort_values("timestamp")
        
        # Calculate split indices
        n_total = len(interactions)
        n_test = int(n_total * test_size)
        n_val = int(n_total * val_size)
        n_train = n_total - n_test - n_val
        
        # Split the data
        train = interactions.iloc[:n_train].copy()
        val = interactions.iloc[n_train:n_train + n_val].copy()
        test = interactions.iloc[n_train + n_val:].copy()
        
        logger.info(f"Split data: {len(train)} train, {len(val)} val, {len(test)} test")
        
        return train, val, test
        
    def get_user_item_matrix(
        self, 
        interactions: pd.DataFrame, 
        users: Optional[pd.DataFrame] = None
    ) -> Tuple[np.ndarray, List[int], List[int]]:
        """Create user-item interaction matrix.
        
        Args:
            interactions: DataFrame with user-item interactions
            users: Optional DataFrame with user information
            
        Returns:
            Tuple of (matrix, user_ids, item_ids)
        """
        # Get unique users and items
        user_ids = sorted(interactions["user_id"].unique())
        item_ids = sorted(interactions["item_id"].unique())
        
        # Create mapping
        user_to_idx = {user_id: idx for idx, user_id in enumerate(user_ids)}
        item_to_idx = {item_id: idx for idx, item_id in enumerate(item_ids)}
        
        # Create matrix
        matrix = np.zeros((len(user_ids), len(item_ids)))
        
        for _, row in interactions.iterrows():
            user_idx = user_to_idx[row["user_id"]]
            item_idx = item_to_idx[row["item_id"]]
            matrix[user_idx, item_idx] = row["weight"]
            
        logger.info(f"Created user-item matrix: {matrix.shape}")
        
        return matrix, user_ids, item_ids
