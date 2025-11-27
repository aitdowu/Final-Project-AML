import os
import logging
from typing import List, Dict, Any, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_vector_store(db_path: str) -> Optional[Any]:
    """Load vector store from disk."""
    try:
        from Models.vector_store import VectorStore
        vector_store = VectorStore()
        vector_store.load(db_path)
        return vector_store
    except Exception as e:
        logger.error(f"Error loading vector store: {e}")
        return None


def generate_answer_with_llm(query: str, relevant_chunks: List[Dict[str, Any]], llm_model: Any) -> str:
    """Generate answer using LLM with the retrieved chunks."""
    # Check if model is actually loaded
    if llm_model == "placeholder_llm" or llm_model is None:
        # Fallback if model didn't load
        context = "\n\n".join([chunk['text'] for chunk in relevant_chunks])
        return f"""Based on the course materials, here's what I found:

{context[:500]}...

[Note: LLM model not loaded. This is a placeholder response.]"""
    
    # Combine chunks into context
    context_parts = []
    for i, chunk in enumerate(relevant_chunks, 1):
        context_parts.append(f"[Context {i} from {chunk['source']}]:\n{chunk['text']}")
    
    context = "\n\n".join(context_parts)
    
    # Build prompt for Qwen2
    system_message = "You are a helpful assistant that answers questions about course materials using the provided context. Provide clear and accurate answers based only on the context. If the context doesn't contain enough information, say so."
    user_message = f"""Context from course materials:
{context}

Question: {query}"""
    
    # Try to use Qwen2's chat template
    try:
        if hasattr(llm_model.tokenizer, 'apply_chat_template') and llm_model.tokenizer.chat_template is not None:
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
            # Fallback format
            prompt = f"""<|im_start|>system
{system_message}<|im_end|>
<|im_start|>user
{user_message}<|im_end|>
<|im_start|>assistant
"""
    except Exception as e:
        logger.warning(f"Chat template error: {e}, using fallback")
        prompt = f"""<|im_start|>system
{system_message}<|im_end|>
<|im_start|>user
{user_message}<|im_end|>
<|im_start|>assistant
"""
    
    try:
        # Lower temperature = more focused answers
        answer = llm_model.generate(prompt, max_new_tokens=512, temperature=0.6, top_p=0.9)
        return answer.strip()
    except Exception as e:
        logger.error(f"Error generating answer: {e}")
        import traceback
        logger.error(traceback.format_exc())
        # Just return the context if generation fails
        context = "\n\n".join([chunk['text'] for chunk in relevant_chunks])
        return f"Error generating answer: {str(e)}\n\nRelevant context found:\n{context[:1000]}"


def format_sources(relevant_chunks: List[Dict[str, Any]]) -> str:
    """Format sources for display."""
    if not relevant_chunks:
        return "No sources found."
    
    sources = []
    for i, chunk in enumerate(relevant_chunks, 1):
        source_info = f"{i}. {chunk['source']} (Chunk {chunk['chunk_id']}) - Similarity: {chunk['similarity_score']:.3f}"
        sources.append(source_info)
    
    return "\n".join(sources)

