"""
Main Chat Interface for RAG Course Notes Chatbot

This script provides a Gradio-based chat interface for the RAG chatbot.
Users can ask questions about course materials, and the system will retrieve
relevant chunks and generate answers using open-source LLMs.

Author: RAG Course Notes Chatbot Project
"""

import os
import logging
import gradio as gr
from typing import List, Dict, Any, Tuple
import json

# Import our custom modules
from ingest import VectorStore, EmbeddingGenerator
from utils import load_vector_store, generate_answer_with_llm, format_sources

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGChatbot:
    """Main RAG chatbot class that handles queries and responses."""
    
    def __init__(self, db_path: str = "db/vector_store"):
        """
        Initialize the RAG chatbot.
        
        Args:
            db_path: Path to the vector database
        """
        self.db_path = db_path
        self.vector_store = None
        self.embedding_generator = None
        self.llm_model = None
        
        # Load components
        self._load_components()
    
    def _load_components(self):
        """Load the vector store and embedding model."""
        try:
            # Load vector store
            self.vector_store = VectorStore()
            self.vector_store.load(self.db_path)
            logger.info("Vector store loaded successfully")
            
            # Load embedding generator
            self.embedding_generator = EmbeddingGenerator()
            logger.info("Embedding generator loaded successfully")
            
            # TODO: Initialize open-source LLM for generation
            # Current placeholder - will be replaced with actual model
            self.llm_model = "placeholder_llm"
            logger.info("LLM model placeholder initialized")
            
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
            # Generate query embedding
            query_embedding = self.embedding_generator.generate_embeddings([query])[0]
            
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
            # TODO: Replace with actual open-source LLM implementation
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
    with gr.Blocks(css=css, title="RAG Course Notes Chatbot") as interface:
        gr.Markdown(
            """
            # 📚 RAG Course Notes Chatbot
            
            Ask questions about your course materials! This chatbot uses Retrieval-Augmented Generation (RAG) 
            to find relevant information from your uploaded PDFs and generate helpful answers.
            
            **How it works:**
            1. Upload your course PDFs to the `data/` directory
            2. Run `python src/ingest.py` to process the documents
            3. Ask questions about the course content
            4. Get answers with source citations
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
                        placeholder="Ask a question about your course materials...",
                        label="Your Question",
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
                    ### 💡 Tips for Better Results
                    
                    - Ask specific questions about concepts, formulas, or topics
                    - Use keywords from your course materials
                    - Try rephrasing if you don't get good results
                    - Check the sources to verify information
                    
                    ### 🔧 Technical Details
                    
                    - **Embedding Model**: all-MiniLM-L6-v2
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
    
    # Check if vector database exists
    db_path = "db/vector_store"
    if not os.path.exists(f"{db_path}.index"):
        logger.error("Vector database not found!")
        logger.info("Please run 'python src/ingest.py' first to process your PDF files.")
        return
    
    # Create and launch interface
    interface = create_chat_interface()
    
    if interface is None:
        logger.error("Failed to create chat interface")
        return
    
    logger.info("Starting RAG Course Notes Chatbot...")
    logger.info("Open your browser and navigate to the URL shown below")
    
    # Launch the interface
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        quiet=False
    )


if __name__ == "__main__":
    main()

