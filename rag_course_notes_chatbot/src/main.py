"""
Main Chat Interface for RAG Course Notes Chatbot

This script provides a Gradio-based chat interface for the RAG chatbot.
Users can ask questions about course materials, and the system will retrieve
relevant chunks and generate answers using open-source LLMs.

Author: RAG Course Notes Chatbot Project
"""

import os

# Set HuggingFace cache directory BEFORE importing transformers
# This must be done before any transformers/sentence_transformers imports
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
local_cache = os.path.join(project_root, ".cache", "huggingface")
os.makedirs(local_cache, exist_ok=True)

# Set environment variables to use local cache
os.environ["HF_HOME"] = local_cache
os.environ["SENTENCE_TRANSFORMERS_HOME"] = local_cache
# Disable tokenizers parallelism warning when forking processes
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import logging
import gradio as gr
from typing import List, Dict, Any, Tuple
import json

# Import our custom modules
from ingest import VectorStore, EmbeddingGenerator
from utils import load_vector_store, generate_answer_with_llm, format_sources, QwenLLM

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set HuggingFace token if available (for Qwen2 model access)
# Note: Qwen2 models are usually publicly available, but token may be needed for some models
# Token should be set via environment variable: export HF_TOKEN=your_token_here
hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
if hf_token:
    logger.info("Using HuggingFace token from environment")
else:
    logger.info("No HuggingFace token found - using public model access (should work for Qwen2)")


class RAGChatbot:
    """Main RAG chatbot class that handles queries and responses."""
    
    def __init__(self, db_path: str = None):
        """
        Initialize the RAG chatbot.
        
        Args:
            db_path: Path to the vector database (if None, uses project root)
        """
        # Get project root directory
        if db_path is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(script_dir)
            db_path = os.path.join(project_root, "db", "vector_store")
        
        self.db_path = db_path
        self.vector_store = None
        self.embedding_generator = None
        self.llm_model = None
        
        # Load components
        self._load_components()
    
    def _load_components(self):
        """Load the vector store, embedding model, and LLM."""
        try:
            # Load embedding generator first to get dimension
            self.embedding_generator = EmbeddingGenerator()
            expected_dim = self.embedding_generator.get_dimension()
            logger.info("Embedding generator loaded successfully")
            
            # Load vector store
            self.vector_store = VectorStore()
            self.vector_store.load(self.db_path)
            logger.info("Vector store loaded successfully")
            
            # Check if dimensions match
            if self.vector_store.dimension != expected_dim:
                logger.warning(
                    f"Dimension mismatch! Database has {self.vector_store.dimension} dimensions, "
                    f"but model produces {expected_dim} dimensions. "
                    f"Please regenerate the database with: rm -rf db/ && python3 src/ingest.py"
                )
                raise ValueError(
                    f"Embedding dimension mismatch. Database: {self.vector_store.dimension}, "
                    f"Model: {expected_dim}. Regenerate database with new model."
                )
            
            # Initialize Qwen2 LLM
            try:
                self.llm_model = QwenLLM()
                logger.info("Qwen2 LLM loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load Qwen2 LLM: {e}")
                logger.warning("Falling back to placeholder mode")
                self.llm_model = "placeholder_llm"
            
        except Exception as e:
            logger.error(f"Error loading components: {e}")
            raise
    
    def search_documents(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for relevant document chunks.
        
        Args:
            query: User query string
            k: Number of chunks to retrieve
            
        Returns:
            List of relevant chunks with metadata
        """
        try:
            # Generate query embedding (with is_query=True to add "query: " prefix for Snowflake model)
            query_embedding = self.embedding_generator.generate_embeddings([query], is_query=True)[0]
            
            # Search vector store
            results = self.vector_store.search(query_embedding, k=k)
            
            return results
        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            return []
    
    def generate_response(self, query: str, chat_history: List[List[str]]) -> Tuple[str, str]:
        """
        Generate a response to a user query using RAG.
        
        Args:
            query: User query string
            chat_history: Previous chat messages
            
        Returns:
            Tuple of (response, sources)
        """
        try:
            # Search for relevant documents
            relevant_chunks = self.search_documents(query, k=5)
            
            if not relevant_chunks:
                return "I couldn't find any relevant information in the course materials for your question.", ""
            
            # Generate answer using LLM
            answer = generate_answer_with_llm(query, relevant_chunks, self.llm_model)
            
            # Format sources
            sources = format_sources(relevant_chunks)
            
            return answer, sources
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"I encountered an error while processing your question: {str(e)}", ""


def create_chat_interface():
    """Create and configure the Gradio chat interface."""
    
    # Initialize chatbot
    try:
        chatbot = RAGChatbot()
    except Exception as e:
        logger.error(f"Failed to initialize chatbot: {e}")
        return None
    
    # Custom CSS for better styling
    css = """
    .gradio-container {
        max-width: 1200px !important;
        margin: auto !important;
    }
    .chat-message {
        padding: 10px !important;
        margin: 5px 0 !important;
    }
    .sources-box {
        background-color: #f0f0f0 !important;
        border: 1px solid #ddd !important;
        border-radius: 5px !important;
        padding: 10px !important;
        margin-top: 10px !important;
    }
    """
    
    def chat_function(message: str, history: List[List[str]]) -> Tuple[str, str]:
        """
        Handle chat interactions.
        
        Args:
            message: User message
            history: Chat history
            
        Returns:
            Tuple of (response, sources)
        """
        if not message.strip():
            return "", ""
        
        response, sources = chatbot.generate_response(message, history)
        return response, sources
    
    # Create Gradio interface
    with gr.Blocks(css=css, title="AML Notes RAG Chatbot") as interface:
        gr.Markdown(
            """
            # AML Notes RAG Chatbot
            
            Ask questions about Applied Machine Learning course materials. This chatbot uses Retrieval-Augmented Generation (RAG) 
            to find relevant information from course PDFs and generate answers.
            
            **How it works:**
            1. PDFs are processed and embedded into a vector database
            2. Questions are matched against course content using semantic search
            3. Relevant chunks are retrieved and used to generate answers
            4. Sources are cited for verification
            """
        )
        
        with gr.Row():
            with gr.Column(scale=3):
                # Chat interface
                chatbot_interface = gr.Chatbot(
                    label="Chat",
                    height=500,
                    show_label=True,
                    container=True,
                    bubble_full_width=False
                )
                
                with gr.Row():
                    msg_input = gr.Textbox(
                        placeholder="Ask a question about Applied Machine Learning...",
                        label="Question",
                        lines=2,
                        scale=4
                    )
                    send_btn = gr.Button("Send", variant="primary", scale=1)
                
                # Sources display
                sources_output = gr.Textbox(
                    label="Sources",
                    lines=4,
                    interactive=False,
                    visible=True
                )
            
            with gr.Column(scale=1):
                gr.Markdown(
                    """
                    ### Tips for Better Results
                    
                    - Ask specific questions about concepts, formulas, or topics
                    - Use keywords from the course materials
                    - Try rephrasing if you don't get good results
                    - Check the sources to verify information
                    
                    ### Technical Details
                    
                    - **Embedding Model**: Snowflake Arctic Embed M v2.0
                    - **LLM Model**: Qwen2-1.5B-Instruct
                    - **Vector Store**: FAISS
                    - **Chunk Size**: 800 characters
                    - **Retrieval**: Top-5 most similar chunks
                    """
                )
        
        # Event handlers
        def user(user_message: str, history: List[List[str]]) -> Tuple[List[List[str]], str, str]:
            """Handle user input."""
            return history + [[user_message, None]], "", ""
        
        def bot(history: List[List[str]]) -> Tuple[List[List[str]], str, str]:
            """Handle bot response."""
            if not history or not history[-1][0]:
                return history, "", ""
            
            user_message = history[-1][0]
            response, sources = chat_function(user_message, history[:-1])
            
            history[-1][1] = response
            return history, "", sources
        
        # Connect events
        msg_input.submit(
            user, 
            [msg_input, chatbot_interface], 
            [chatbot_interface, msg_input, sources_output], 
            queue=False
        ).then(
            bot, 
            chatbot_interface, 
            [chatbot_interface, msg_input, sources_output]
        )
        
        send_btn.click(
            user, 
            [msg_input, chatbot_interface], 
            [chatbot_interface, msg_input, sources_output], 
            queue=False
        ).then(
            bot, 
            chatbot_interface, 
            [chatbot_interface, msg_input, sources_output]
        )
    
    return interface


def main():
    """Main function to launch the chat interface."""
    
    # Get project root directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    db_path = os.path.join(project_root, "db", "vector_store")
    
    # Check if vector database exists
    if not os.path.exists(f"{db_path}.index"):
        logger.error("Vector database not found!")
        logger.info(f"Expected database at: {db_path}.index")
        logger.info("Please run 'python src/ingest.py' first to process your PDF files.")
        return
    
    # Create and launch interface
    interface = create_chat_interface()
    
    if interface is None:
        logger.error("Failed to create chat interface")
        return
    
    logger.info("Starting RAG Course Notes Chatbot...")
    logger.info("Open your browser and navigate to the URL shown below")
    
    # Try to find an available port
    import socket
    def find_free_port(start_port=7860):
        for port in range(start_port, start_port + 10):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                try:
                    s.bind(('localhost', port))
                    return port
                except OSError:
                    continue
        return start_port  # Fallback
    
    port = find_free_port(7860)
    if port != 7860:
        logger.info(f"Port 7860 in use, using port {port} instead")
    
    # Launch the interface
    interface.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=False,
        show_error=True,
        quiet=False
    )


if __name__ == "__main__":
    main()

