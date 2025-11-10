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
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

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


class QwenLLM:
    """Qwen2-1.5B-Instruct LLM wrapper for answer generation."""
    
    def __init__(self, model_name: str = "Qwen/Qwen2-1.5B-Instruct", use_quantization: bool = True):
        """
        Initialize Qwen2 LLM.
        
        Args:
            model_name: HuggingFace model identifier
            use_quantization: Whether to use 8-bit quantization for CPU (reduces memory usage)
        """
        self.model_name = model_name
        self.device = self._get_device()
        self.use_quantization = use_quantization and self.device == "cpu"
        
        # Get HuggingFace token from environment (usually not needed for Qwen2, but just in case)
        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
        
        logger.info(f"Loading Qwen2 model: {model_name} on device: {self.device}")
        if self.use_quantization:
            logger.info("Using 8-bit quantization for reduced memory usage")
        
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True,
                token=hf_token if hf_token else None
            )
            
            # Set pad token if not set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model with appropriate settings
            model_kwargs = {
                "trust_remote_code": True,
                "token": hf_token if hf_token else None
            }
            
            if self.device == "cuda":
                # CUDA: Use float16 for faster inference
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    **model_kwargs
                )
            elif self.device == "mps":
                # MPS (Apple Silicon): Use float32
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float32,
                    **model_kwargs
                )
                self.model = self.model.to(self.device)
            else:
                # CPU: Use quantization if available, otherwise float32
                # Note: bitsandbytes doesn't work on macOS/ARM, so it will fall back gracefully
                if self.use_quantization:
                    try:
                        # Try to use 8-bit quantization to reduce memory usage
                        # This requires bitsandbytes which only works on Linux/Windows with CUDA
                        try:
                            from transformers import BitsAndBytesConfig
                            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
                            self.model = AutoModelForCausalLM.from_pretrained(
                                model_name,
                                quantization_config=quantization_config,
                                low_cpu_mem_usage=True,
                                **model_kwargs
                            )
                            logger.info("Loaded model with 8-bit quantization")
                        except (ImportError, ValueError, RuntimeError) as e:
                            # bitsandbytes not available or not supported (e.g., on macOS)
                            logger.info(f"Quantization not available ({e}), using float32 instead")
                            self.use_quantization = False
                            self.model = AutoModelForCausalLM.from_pretrained(
                                model_name,
                                torch_dtype=torch.float32,
                                low_cpu_mem_usage=True,
                                **model_kwargs
                            )
                            self.model = self.model.to(self.device)
                    except Exception as e:
                        logger.warning(f"Quantization failed: {e}, falling back to float32")
                        self.use_quantization = False
                        self.model = AutoModelForCausalLM.from_pretrained(
                            model_name,
                            torch_dtype=torch.float32,
                            low_cpu_mem_usage=True,
                            **model_kwargs
                        )
                        self.model = self.model.to(self.device)
                else:
                    # No quantization, use float32
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        torch_dtype=torch.float32,
                        low_cpu_mem_usage=True,
                        **model_kwargs
                    )
                    self.model = self.model.to(self.device)
            
            # Set model to evaluation mode
            self.model.eval()
            logger.info("Qwen2 model loaded successfully")
            
        except Exception as e:
            error_msg = str(e).lower()
            if "token" in error_msg or "authentication" in error_msg:
                logger.error(f"Authentication error loading Qwen2 model: {e}")
                logger.error("Qwen2 model may require a HuggingFace access token.")
                logger.error("Set it as an environment variable:")
                logger.error("  export HF_TOKEN=your_token_here")
                logger.error("Get your token from: https://huggingface.co/settings/tokens")
            else:
                logger.error(f"Error loading Qwen2 model: {e}")
            raise
    
    def _get_device(self) -> str:
        """Determine the best device to use."""
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    
    def generate(self, prompt: str, max_new_tokens: int = 512, temperature: float = 0.7, top_p: float = 0.9) -> str:
        """
        Generate text from a prompt.
        
        Args:
            prompt: Input prompt (can be a string or formatted chat messages)
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            
        Returns:
            Generated text
        """
        try:
            # Tokenize input
            inputs = self.tokenizer(prompt, return_tensors="pt")
            
            # Move to device only if not using quantization (quantized models handle this automatically)
            if not self.use_quantization:
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate
            with torch.no_grad():
                generate_kwargs = {
                    **inputs,
                    "max_new_tokens": max_new_tokens,
                    "temperature": temperature,
                    "top_p": top_p,
                    "do_sample": temperature > 0,
                    "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                }
                
                outputs = self.model.generate(**generate_kwargs)
            
            # Decode output (only the newly generated tokens)
            # Get the length of input tokens to extract only new tokens
            input_length = inputs['input_ids'].shape[1]
            generated_tokens = outputs[0][input_length:]
            generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            return generated_text.strip()
            
        except Exception as e:
            logger.error(f"Error generating text: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return f"Error generating response: {str(e)}"


def generate_answer_with_llm(query: str, relevant_chunks: List[Dict[str, Any]], llm_model: Any) -> str:
    """
    Generate answer using LLM with retrieved context.
    
    Args:
        query: User query
        relevant_chunks: Retrieved document chunks
        llm_model: QwenLLM instance or placeholder string
        
    Returns:
        Generated answer
    """
    # Check if we have an actual LLM model
    if llm_model == "placeholder_llm" or llm_model is None:
        # Fallback to placeholder if model not loaded
        context = "\n\n".join([chunk['text'] for chunk in relevant_chunks])
        return f"""Based on the course materials, here's what I found:

{context[:500]}...

[Note: LLM model not loaded. This is a placeholder response.]"""
    
    # Combine context from relevant chunks
    context_parts = []
    for i, chunk in enumerate(relevant_chunks, 1):
        context_parts.append(f"[Context {i} from {chunk['source']}]:\n{chunk['text']}")
    
    context = "\n\n".join(context_parts)
    
    # Create RAG prompt for Qwen2
    system_message = "You are a helpful assistant that answers questions about course materials using the provided context. Provide clear and accurate answers based only on the context. If the context doesn't contain enough information, say so."
    user_message = f"""Context from course materials:
{context}

Question: {query}"""
    
    # Use Qwen2's chat template
    try:
        if hasattr(llm_model.tokenizer, 'apply_chat_template') and llm_model.tokenizer.chat_template is not None:
            # Qwen2 uses a specific chat format
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ]
            prompt = llm_model.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
        else:
            # Fallback format for Qwen2 (though it should have a chat template)
            prompt = f"""<|im_start|>system
{system_message}<|im_end|>
<|im_start|>user
{user_message}<|im_end|>
<|im_start|>assistant
"""
    except Exception as e:
        logger.warning(f"Error applying chat template: {e}, using fallback format")
        # Qwen2 fallback format
        prompt = f"""<|im_start|>system
{system_message}<|im_end|>
<|im_start|>user
{user_message}<|im_end|>
<|im_start|>assistant
"""
    
    try:
        # Generate answer with slightly lower temperature for more focused responses
        answer = llm_model.generate(prompt, max_new_tokens=512, temperature=0.6, top_p=0.9)
        return answer.strip()
    except Exception as e:
        logger.error(f"Error generating answer with LLM: {e}")
        import traceback
        logger.error(traceback.format_exc())
        # Fallback to simple context return
        context = "\n\n".join([chunk['text'] for chunk in relevant_chunks])
        return f"Error generating answer: {str(e)}\n\nRelevant context found:\n{context[:1000]}"


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

