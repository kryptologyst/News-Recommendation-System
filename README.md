# News Recommendation System

A content-based news recommendation system that uses advanced NLP techniques to recommend relevant news articles to users based on their reading preferences.

## Features

- **Multiple Recommendation Models**: TF-IDF, Sentence-BERT, and Hybrid approaches
- **Content-Based Filtering**: Recommendations based on article content similarity
- **Interactive Demo**: User-friendly Streamlit interface for exploring recommendations
- **Comprehensive Evaluation**: Multiple metrics including Precision@K, Recall@K, NDCG@K, and Hit Rate
- **Production-Ready Structure**: Clean code with type hints, proper documentation, and testing

## Models

### 1. TF-IDF Model
Uses Term Frequency-Inverse Document Frequency to convert article text into numerical features and computes cosine similarity for recommendations.

### 2. SBERT Model
Uses Sentence-BERT embeddings to capture semantic similarity between articles for more nuanced recommendations.

### 3. Hybrid Model
Combines both TF-IDF and SBERT models using weighted voting to leverage the strengths of both approaches.

## Installation

### Prerequisites
- Python 3.10 or higher
- pip or conda package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/kryptologyst/News-Recommendation-System.git
cd News-Recommendation-System
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

Or using conda:
```bash
conda env create -f environment.yml
conda activate news-recommender
```

3. Install the package in development mode:
```bash
pip install -e .
```

## Quick Start

### 1. Generate Synthetic Data

The system will automatically create synthetic news data if no real data is provided:

```bash
python scripts/train.py
```

### 2. Run the Interactive Demo

Launch the Streamlit demo:

```bash
streamlit run scripts/demo.py
```

Open your browser to `http://localhost:8501` to access the demo.

### 3. Train and Evaluate Models

Train all models and evaluate their performance:

```bash
python scripts/train.py --config configs/config.yaml
```

## Data Format

The system expects the following data files in the `data/` directory:

### interactions.csv
```csv
user_id,item_id,timestamp,weight
1,101,2024-01-01 10:00:00,1.0
1,102,2024-01-01 11:00:00,0.5
2,101,2024-01-01 12:00:00,1.0
```

### items.csv
```csv
item_id,title,content,category,tags
101,"Breaking News","Article content here...",Politics,"politics,news"
102,"Sports Update","Article content here...",Sports,"sports,news"
```

### users.csv (optional)
```csv
user_id,age,gender,location
1,25,Male,New York
2,30,Female,California
```

## Configuration

Modify `configs/config.yaml` to adjust model parameters:

```yaml
# Data settings
data:
  interactions_file: "data/interactions.csv"
  items_file: "data/items.csv"
  test_size: 0.2
  val_size: 0.1
  random_seed: 42

# Model settings
models:
  tfidf:
    max_features: 10000
    ngram_range: [1, 2]
    min_df: 2
    max_df: 0.95
    
  sbert:
    model_name: "all-MiniLM-L6-v2"
    batch_size: 32
    
  hybrid:
    tfidf_weight: 0.6
    sbert_weight: 0.4

# Evaluation settings
evaluation:
  metrics: ["precision", "recall", "map", "ndcg", "hit_rate"]
  k_values: [5, 10, 20]
```

## Usage

### Programmatic Usage

```python
from news_recommender.data.loader import NewsDataLoader
from news_recommender.models.recommenders import TFIDFRecommender

# Load data
loader = NewsDataLoader("data")
items = loader.load_items()
interactions = loader.load_interactions()

# Train model
model = TFIDFRecommender()
model.fit(items, interactions)

# Get recommendations
recommendations = model.recommend(user_id=1, n_recommendations=10)
print(f"Recommendations: {recommendations}")

# Get similar items
similar_items = model.get_similar_items(item_id=101, n_similar=5)
print(f"Similar items: {similar_items}")
```

### Command Line Interface

Train models:
```bash
python scripts/train.py --config configs/config.yaml --data-dir data --output-dir models
```

Run demo:
```bash
streamlit run scripts/demo.py
```

## Evaluation Metrics

The system evaluates models using multiple metrics:

- **Precision@K**: Fraction of recommended items that are relevant
- **Recall@K**: Fraction of relevant items that are recommended
- **NDCG@K**: Normalized Discounted Cumulative Gain
- **Hit Rate@K**: Fraction of users with at least one relevant recommendation
- **Coverage**: Fraction of catalog items that can be recommended
- **Novelty**: Average popularity of recommended items

## Project Structure

```
news-recommendation-system/
├── src/
│   └── news_recommender/
│       ├── data/
│       │   └── loader.py
│       ├── models/
│       │   └── recommenders.py
│       ├── evaluation/
│       │   └── metrics.py
│       └── utils/
├── configs/
│   └── config.yaml
├── scripts/
│   ├── train.py
│   └── demo.py
├── tests/
│   └── test_recommenders.py
├── data/
├── models/
├── notebooks/
├── requirements.txt
├── pyproject.toml
└── README.md
```

## Development

### Running Tests

```bash
pytest tests/
```

### Code Formatting

```bash
black src/ scripts/ tests/
ruff check src/ scripts/ tests/
```

### Type Checking

```bash
mypy src/
```

## API Reference

### NewsDataLoader

Main class for loading and preprocessing news data.

#### Methods

- `load_interactions(filename)`: Load user-item interactions
- `load_items(filename)`: Load news articles
- `load_users(filename)`: Load user information
- `split_data(interactions, test_size, val_size)`: Split data into train/val/test
- `get_user_item_matrix(interactions)`: Create user-item interaction matrix

### BaseRecommender

Abstract base class for all recommendation models.

#### Methods

- `fit(items, interactions)`: Train the model
- `recommend(user_id, n_recommendations)`: Generate recommendations
- `get_similar_items(item_id, n_similar)`: Find similar items

### RecommendationEvaluator

Class for evaluating recommendation models.

#### Methods

- `evaluate_model(model, test_interactions, items)`: Evaluate a single model
- `compare_models(models, test_interactions, items)`: Compare multiple models
- `print_results(results_df)`: Print formatted results

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Sentence-Transformers library for pre-trained BERT models
- scikit-learn for TF-IDF implementation
- Streamlit for the interactive demo interface
- The recommendation systems research community

## Future Enhancements

- [ ] Collaborative filtering models
- [ ] Real-time recommendation API
- [ ] Advanced evaluation metrics
- [ ] Model persistence and loading
- [ ] Distributed training support
- [ ] A/B testing framework
- [ ] Fairness and bias evaluation
- [ ] Multi-objective optimization
# News-Recommendation-System
