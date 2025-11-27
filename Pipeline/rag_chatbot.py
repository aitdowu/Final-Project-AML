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
from Models.embedding_model import EmbeddingGenerator
from Models.vector_store import VectorStore
from Models.llm_model import QwenLLM
from Utils.helper_functions import generate_answer_with_llm, format_sources

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
    """Main chatbot class."""
    
    def __init__(self, db_path: str = None):
        """Initialize chatbot. db_path defaults to project root if not provided."""
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
        """Load vector store, embedding model, and LLM."""
        try:
            # Get model name from summary if it exists
            summary_path = f"{self.db_path}.summary.json"
            if os.path.exists(summary_path):
                with open(summary_path, 'r') as f:
                    summary = json.load(f)
                model_name = summary.get('embedding_model', 'Snowflake/snowflake-arctic-embed-m-v2.0')
            else:
                model_name = 'Snowflake/snowflake-arctic-embed-m-v2.0'
            
            # Load embedding generator
            self.embedding_generator = EmbeddingGenerator(model_name=model_name)
            expected_dim = self.embedding_generator.get_dimension()
            logger.info("Embedding generator loaded")
            
            # Load vector store
            self.vector_store = VectorStore()
            self.vector_store.load(self.db_path)
            logger.info("Vector store loaded")
            
            # Check dimensions match
            if self.vector_store.dimension != expected_dim:
                logger.warning(
                    f"Dimension mismatch! DB has {self.vector_store.dimension}, "
                    f"model has {expected_dim}. Regenerate with: rm -rf db/ && python Pipeline/main_pipeline.py"
                )
                raise ValueError(
                    f"Dimension mismatch. DB: {self.vector_store.dimension}, Model: {expected_dim}"
                )
            
            # Load LLM
            try:
                self.llm_model = QwenLLM()
                logger.info("Qwen2 LLM loaded")
            except Exception as e:
                logger.warning(f"Failed to load LLM: {e}, using placeholder")
                self.llm_model = "placeholder_llm"
            
        except Exception as e:
            logger.error(f"Error loading components: {e}")
            raise
    
    def search_documents(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search for relevant chunks."""
        try:
            # Generate query embedding (is_query=True adds "query: " prefix for Snowflake)
            query_embedding = self.embedding_generator.generate_embeddings([query], is_query=True)[0]
            
            # Search
            results = self.vector_store.search(query_embedding, k=k)
            return results
        except Exception as e:
            logger.error(f"Error searching: {e}")
            return []
    
    def generate_response(self, query: str, chat_history: List[List[str]]) -> Tuple[str, str]:
        """Generate response using RAG."""
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
    """Create Gradio interface."""
    
    # Initialize chatbot
    try:
        chatbot = RAGChatbot()
    except Exception as e:
        logger.error(f"Failed to initialize: {e}")
        return None
    
    # Some basic styling
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
        """Handle chat messages."""
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
    """Launch the chat interface."""
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    db_path = os.path.join(project_root, "db", "vector_store")
    
    # Check if database exists
    if not os.path.exists(f"{db_path}.index"):
        logger.error("Vector database not found!")
        logger.info(f"Expected at: {db_path}.index")
        logger.info("Run 'python Pipeline/main_pipeline.py' first")
        return
    
    # Create interface
    interface = create_chat_interface()
    
    if interface is None:
        logger.error("Failed to create interface")
        return
    
    logger.info("Starting chatbot...")
    logger.info("Open the URL in your browser")
    
    # Find free port
    import socket
    def find_free_port(start_port=7860):
        for port in range(start_port, start_port + 10):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                try:
                    s.bind(('localhost', port))
                    return port
                except OSError:
                    continue
        return start_port
    
    port = find_free_port(7860)
    if port != 7860:
        logger.info(f"Port 7860 in use, using {port}")
    
    # Launch
    interface.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=False,
        show_error=True,
        quiet=False
    )


if __name__ == "__main__":
    main()

