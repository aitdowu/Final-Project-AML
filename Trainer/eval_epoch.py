import os
import logging
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import List, Dict, Any
from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def calculate_precision_at_k(relevant_chunks: List[Dict[str, Any]], k: int = 5) -> float:
    """Calculate precision@k. Uses similarity > 0.5 as relevant (simplified)."""
    if not relevant_chunks:
        return 0.0
    
    top_k_chunks = relevant_chunks[:k]
    relevant_count = sum(1 for chunk in top_k_chunks if chunk['similarity_score'] > 0.5)
    
    return relevant_count / len(top_k_chunks)


def calculate_recall_at_k(relevant_chunks: List[Dict[str, Any]], total_relevant: int, k: int = 5) -> float:
    """Calculate recall@k."""
    if total_relevant == 0:
        return 0.0
    
    # Take top k chunks
    top_k_chunks = relevant_chunks[:k]
    
    # Count relevant chunks in top k
    relevant_count = sum(1 for chunk in top_k_chunks if chunk['similarity_score'] > 0.5)
    
    return relevant_count / total_relevant


def evaluate_retrieval_performance(queries: List[str], ground_truth: List[List[str]], 
                                 vector_store: Any, embedding_generator: Any) -> Dict[str, float]:
    """
    Evaluate retrieval performance on a set of queries.
    
    Args:
        queries: List of test queries
        ground_truth: List of ground truth relevant document IDs for each query
        vector_store: Vector store instance
        embedding_generator: Embedding generator instance
        
    Returns:
        Dictionary of evaluation metrics
    """
    metrics = {
        'precision_at_5': [],
        'recall_at_5': [],
        'precision_at_10': [],
        'recall_at_10': [],
        'mean_similarity': []
    }
    
    for query, gt_docs in zip(queries, ground_truth):
        # Generate query embedding (with is_query=True for Snowflake model)
        query_embedding = embedding_generator.generate_embeddings([query], is_query=True)[0]
        
        # Search vector store
        results = vector_store.search(query_embedding, k=10)
        
        # Calculate metrics
        precision_5 = calculate_precision_at_k(results, k=5)
        precision_10 = calculate_precision_at_k(results, k=10)
        
        recall_5 = calculate_recall_at_k(results, len(gt_docs), k=5)
        recall_10 = calculate_recall_at_k(results, len(gt_docs), k=10)
        
        mean_sim = np.mean([chunk['similarity_score'] for chunk in results])
        
        metrics['precision_at_5'].append(precision_5)
        metrics['recall_at_5'].append(recall_5)
        metrics['precision_at_10'].append(precision_10)
        metrics['recall_at_10'].append(recall_10)
        metrics['mean_similarity'].append(mean_sim)
    
    # Calculate averages
    avg_metrics = {key: np.mean(values) for key, values in metrics.items()}
    
    return avg_metrics


def plot_retrieval_metrics(metrics: Dict[str, float], save_path: str = "figures/evaluation_performance_metrics.png"):
    """Plot evaluation metrics."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Retrieval Performance Metrics', fontsize=16)
    
    # Precision metrics
    precision_data = [metrics['precision_at_5'], metrics['precision_at_10']]
    axes[0, 0].bar(['P@5', 'P@10'], precision_data, color=['skyblue', 'lightcoral'])
    axes[0, 0].set_title('Precision@K')
    axes[0, 0].set_ylabel('Precision')
    axes[0, 0].set_ylim(0, 1)
    
    # Recall metrics
    recall_data = [metrics['recall_at_5'], metrics['recall_at_10']]
    axes[0, 1].bar(['R@5', 'R@10'], recall_data, color=['lightgreen', 'gold'])
    axes[0, 1].set_title('Recall@K')
    axes[0, 1].set_ylabel('Recall')
    axes[0, 1].set_ylim(0, 1)
    
    # Similarity distribution
    axes[1, 0].hist([metrics['mean_similarity']], bins=10, color='purple', alpha=0.7)
    axes[1, 0].set_title('Mean Similarity Score Distribution')
    axes[1, 0].set_xlabel('Similarity Score')
    axes[1, 0].set_ylabel('Frequency')
    
    # Combined metrics
    metric_names = ['P@5', 'R@5', 'P@10', 'R@10']
    metric_values = [metrics['precision_at_5'], metrics['recall_at_5'], 
                    metrics['precision_at_10'], metrics['recall_at_10']]
    
    axes[1, 1].bar(metric_names, metric_values, color=['skyblue', 'lightgreen', 'lightcoral', 'gold'])
    axes[1, 1].set_title('All Metrics Comparison')
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Retrieval metrics plot saved to {save_path}")


def create_evaluation_report(metrics: Dict[str, float], save_path: str = "results/evaluation_report.txt"):
    """Create evaluation report."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    report = f"""
RAG Course Notes Chatbot - Evaluation Report
============================================

Retrieval Performance Metrics:
-----------------------------
Precision@5:  {metrics.get('precision_at_5', 0):.3f}
Recall@5:     {metrics.get('recall_at_5', 0):.3f}
Precision@10: {metrics.get('precision_at_10', 0):.3f}
Recall@10:    {metrics.get('recall_at_10', 0):.3f}
Mean Similarity: {metrics.get('mean_similarity', 0):.3f}

Model Information:
-----------------
Embedding Model: Snowflake/snowflake-arctic-embed-m-v2.0
Chunk Size: 800 characters
Chunk Overlap: 100 characters

Evaluation Notes:
----------------
- Precision@K measures the fraction of retrieved documents that are relevant
- Recall@K measures the fraction of relevant documents that are retrieved
- Similarity scores are based on cosine similarity in embedding space
- Ground truth relevance is approximated using similarity thresholds

Recommendations:
---------------
- Consider fine-tuning embedding model on domain-specific data
- Experiment with different chunk sizes and overlap strategies
- Implement more sophisticated relevance scoring
- Add user feedback mechanism for continuous improvement
"""
    
    with open(save_path, 'w') as f:
        f.write(report)
    
    logger.info(f"Evaluation report saved to {save_path}")


def save_results_to_csv(results: List[Dict[str, Any]], save_path: str = "results/evaluation_performance_table.csv"):
    """Save results to CSV."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    df = pd.DataFrame(results)
    df.to_csv(save_path, index=False)
    
    logger.info(f"Evaluation results saved to {save_path}")

