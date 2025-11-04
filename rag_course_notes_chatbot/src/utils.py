"""
Utility Functions for RAG Course Notes Chatbot

This module contains helper functions for OCR extraction, caption merging,
embedding & retrieval utilities, metric calculations, and evaluation plotting.

Author: RAG Course Notes Chatbot Project
"""

import os
import logging
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import precision_score, recall_score, f1_score
import json
import pickle

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_vector_store(db_path: str) -> Optional[Any]:
    """
    Load vector store from disk.
    
    Args:
        db_path: Path to the vector database
        
    Returns:
        Loaded vector store or None if error
    """
    try:
        from ingest import VectorStore
        vector_store = VectorStore()
        vector_store.load(db_path)
        return vector_store
    except Exception as e:
        logger.error(f"Error loading vector store: {e}")
        return None


def generate_answer_with_llm(query: str, relevant_chunks: List[Dict[str, Any]], llm_model: str) -> str:
    """
    Generate answer using LLM with retrieved context.
    
    Args:
        query: User query
        relevant_chunks: Retrieved document chunks
        llm_model: LLM model identifier
        
    Returns:
        Generated answer
    """
    # TODO: Implement actual open-source LLM integration
    # This is a placeholder implementation
    
    # Combine context from relevant chunks
    context = "\n\n".join([chunk['text'] for chunk in relevant_chunks])
    
    # Simple template-based response (placeholder)
    answer = f"""Based on the course materials, here's what I found:

{context[:500]}...

[Note: This is a placeholder response. In the full implementation, this would use an open-source LLM like Llama 2, Mistral, or similar to generate a proper answer based on the retrieved context.]

The information above comes from {len(relevant_chunks)} relevant sections of your course materials."""
    
    return answer


def format_sources(relevant_chunks: List[Dict[str, Any]]) -> str:
    """
    Format source information for display.
    
    Args:
        relevant_chunks: Retrieved document chunks
        
    Returns:
        Formatted source string
    """
    if not relevant_chunks:
        return "No sources found."
    
    sources = []
    for i, chunk in enumerate(relevant_chunks, 1):
        source_info = f"{i}. {chunk['source']} (Chunk {chunk['chunk_id']}) - Similarity: {chunk['similarity_score']:.3f}"
        sources.append(source_info)
    
    return "\n".join(sources)


def calculate_cosine_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """
    Calculate cosine similarity between two embeddings.
    
    Args:
        embedding1: First embedding vector
        embedding2: Second embedding vector
        
    Returns:
        Cosine similarity score
    """
    return cosine_similarity([embedding1], [embedding2])[0][0]


def calculate_precision_at_k(relevant_chunks: List[Dict[str, Any]], k: int = 5) -> float:
    """
    Calculate precision@k for retrieval evaluation.
    
    Args:
        relevant_chunks: Retrieved chunks with relevance scores
        k: Number of top results to consider
        
    Returns:
        Precision@k score
    """
    if not relevant_chunks:
        return 0.0
    
    # Take top k chunks
    top_k_chunks = relevant_chunks[:k]
    
    # For this implementation, we'll use similarity score > 0.5 as relevant
    # In a real evaluation, you'd have ground truth relevance labels
    relevant_count = sum(1 for chunk in top_k_chunks if chunk['similarity_score'] > 0.5)
    
    return relevant_count / len(top_k_chunks)


def calculate_recall_at_k(relevant_chunks: List[Dict[str, Any]], total_relevant: int, k: int = 5) -> float:
    """
    Calculate recall@k for retrieval evaluation.
    
    Args:
        relevant_chunks: Retrieved chunks with relevance scores
        total_relevant: Total number of relevant documents
        k: Number of top results to consider
        
    Returns:
        Recall@k score
    """
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
        # Generate query embedding
        query_embedding = embedding_generator.generate_embeddings([query])[0]
        
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


def plot_retrieval_metrics(metrics: Dict[str, float], save_path: str = "results/retrieval_metrics.png"):
    """
    Plot retrieval evaluation metrics.
    
    Args:
        metrics: Dictionary of evaluation metrics
        save_path: Path to save the plot
    """
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


def plot_embedding_distribution(embeddings: np.ndarray, save_path: str = "results/embedding_distribution.png"):
    """
    Plot embedding distribution analysis.
    
    Args:
        embeddings: Numpy array of embeddings
        save_path: Path to save the plot
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Embedding Distribution Analysis', fontsize=16)
    
    # Embedding dimensions histogram
    axes[0, 0].hist(embeddings.flatten(), bins=50, alpha=0.7, color='blue')
    axes[0, 0].set_title('Embedding Values Distribution')
    axes[0, 0].set_xlabel('Embedding Value')
    axes[0, 0].set_ylabel('Frequency')
    
    # Mean embedding per dimension
    mean_embeddings = np.mean(embeddings, axis=0)
    axes[0, 1].plot(mean_embeddings, color='red')
    axes[0, 1].set_title('Mean Embedding per Dimension')
    axes[0, 1].set_xlabel('Dimension')
    axes[0, 1].set_ylabel('Mean Value')
    
    # Embedding magnitude distribution
    magnitudes = np.linalg.norm(embeddings, axis=1)
    axes[1, 0].hist(magnitudes, bins=30, alpha=0.7, color='green')
    axes[1, 0].set_title('Embedding Magnitude Distribution')
    axes[1, 0].set_xlabel('Magnitude')
    axes[1, 0].set_ylabel('Frequency')
    
    # Similarity matrix heatmap (sample)
    if len(embeddings) > 100:
        # Sample for visualization
        sample_indices = np.random.choice(len(embeddings), 50, replace=False)
        sample_embeddings = embeddings[sample_indices]
        similarity_matrix = cosine_similarity(sample_embeddings)
    else:
        similarity_matrix = cosine_similarity(embeddings)
    
    im = axes[1, 1].imshow(similarity_matrix, cmap='viridis')
    axes[1, 1].set_title('Cosine Similarity Matrix (Sample)')
    axes[1, 1].set_xlabel('Document Index')
    axes[1, 1].set_ylabel('Document Index')
    plt.colorbar(im, ax=axes[1, 1])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Embedding distribution plot saved to {save_path}")


def plot_text_length_distribution(text_lengths: List[int], save_path: str = "results/text_length_distribution.png"):
    """
    Plot text length distribution analysis.
    
    Args:
        text_lengths: List of text lengths
        save_path: Path to save the plot
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Text Length Distribution Analysis', fontsize=16)
    
    # Histogram of text lengths
    axes[0, 0].hist(text_lengths, bins=30, alpha=0.7, color='blue')
    axes[0, 0].set_title('Text Length Distribution')
    axes[0, 0].set_xlabel('Text Length (characters)')
    axes[0, 0].set_ylabel('Frequency')
    
    # Box plot
    axes[0, 1].boxplot(text_lengths)
    axes[0, 1].set_title('Text Length Box Plot')
    axes[0, 1].set_ylabel('Text Length (characters)')
    
    # Cumulative distribution
    sorted_lengths = np.sort(text_lengths)
    cumulative = np.arange(1, len(sorted_lengths) + 1) / len(sorted_lengths)
    axes[1, 0].plot(sorted_lengths, cumulative, color='red')
    axes[1, 0].set_title('Cumulative Distribution')
    axes[1, 0].set_xlabel('Text Length (characters)')
    axes[1, 0].set_ylabel('Cumulative Probability')
    
    # Statistics summary
    stats_text = f"""
    Statistics Summary:
    Mean: {np.mean(text_lengths):.1f}
    Median: {np.median(text_lengths):.1f}
    Std: {np.std(text_lengths):.1f}
    Min: {np.min(text_lengths)}
    Max: {np.max(text_lengths)}
    """
    axes[1, 1].text(0.1, 0.5, stats_text, transform=axes[1, 1].transAxes, 
                    fontsize=12, verticalalignment='center')
    axes[1, 1].set_title('Statistics Summary')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Text length distribution plot saved to {save_path}")


def create_evaluation_report(metrics: Dict[str, float], save_path: str = "results/evaluation_report.txt"):
    """
    Create a comprehensive evaluation report.
    
    Args:
        metrics: Dictionary of evaluation metrics
        save_path: Path to save the report
    """
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
Embedding Model: all-MiniLM-L6-v2
Embedding Dimension: 384
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


def save_results_to_csv(results: List[Dict[str, Any]], save_path: str = "results/evaluation_results.csv"):
    """
    Save evaluation results to CSV file.
    
    Args:
        results: List of result dictionaries
        save_path: Path to save the CSV file
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    df = pd.DataFrame(results)
    df.to_csv(save_path, index=False)
    
    logger.info(f"Evaluation results saved to {save_path}")


if __name__ == "__main__":
    # Example usage and testing
    logger.info("Utility functions loaded successfully")
    
    # Test embedding distribution plotting
    test_embeddings = np.random.randn(100, 384)
    plot_embedding_distribution(test_embeddings)
    
    # Test text length distribution plotting
    test_lengths = np.random.randint(100, 1000, 200)
    plot_text_length_distribution(test_lengths)
    
    logger.info("Test plots generated successfully")

