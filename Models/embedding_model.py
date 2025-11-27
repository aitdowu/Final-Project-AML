import os

# Set up cache directory locally - had issues with permissions using default location
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
local_cache = os.path.join(project_root, ".cache", "huggingface")
os.makedirs(local_cache, exist_ok=True)

# Point everything to local cache
os.environ["HF_HOME"] = local_cache
os.environ["SENTENCE_TRANSFORMERS_HOME"] = local_cache
# This warning was annoying
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# xformers doesn't work on mac, so just disabled it
os.environ["DISABLE_XFORMERS"] = "1"

import logging
import warnings
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """Generates embeddings from text. Works with sentence-transformers or transformers."""
    
    def __init__(self, model_name: str = "Snowflake/snowflake-arctic-embed-m-v2.0"):
        """Initialize with a model name. Default is Snowflake which works pretty well."""
        logger.info(f"Loading embedding model: {model_name}")
        
        # Get the local cache path (set at module level)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        local_cache = os.path.join(project_root, ".cache", "huggingface")
        
        self.model_name = model_name
        # Check if it's a Snowflake model (needs special handling)
        self.is_snowflake_model = ("snowflake" in model_name.lower() or 
                                   "arctic-embed" in model_name.lower() or
                                   "arctic_embed" in model_name.lower())
        
        logger.info(f"Model detection: is_snowflake_model = {self.is_snowflake_model}")
        
        try:
            if self.is_snowflake_model:
                # Snowflake model needs transformers, not sentence-transformers
                logger.info("Loading Snowflake Arctic Embed model with transformers")
                
                # Suppress annoying warnings
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore")
                    
                    # Load tokenizer first
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        model_name,
                        cache_dir=local_cache,
                        trust_remote_code=True
                    )
                    
                    # Need to disable memory_efficient_attention or it wants xformers
                    # which doesn't work on mac
                    from transformers import AutoConfig
                    config = AutoConfig.from_pretrained(
                        model_name,
                        cache_dir=local_cache,
                        trust_remote_code=True
                    )
                    
                    logger.info("Disabling memory_efficient_attention so we don't need xformers")
                    
                    # Try to set it to False in a bunch of ways (config is weird)
                    original_value = getattr(config, 'use_memory_efficient_attention', None)
                    
                    config.use_memory_efficient_attention = False
                    setattr(config, 'use_memory_efficient_attention', False)
                    
                    # Also try the internal dict
                    if hasattr(config, '__dict__'):
                        config.__dict__['use_memory_efficient_attention'] = False
                    
                    # Try updating via to_dict too
                    try:
                        config_dict = config.to_dict()
                        config_dict['use_memory_efficient_attention'] = False
                        # Create new config from modified dict
                        config = type(config).from_dict(config_dict)
                        config.use_memory_efficient_attention = False
                    except Exception as e:
                        logger.debug(f"Could not recreate config from dict: {e}")
                        # Continue with modified config object
                    
                    logger.info(f"  Changed use_memory_efficient_attention: {original_value} -> False")
                    
                    # Verify the change took effect
                    final_value = getattr(config, 'use_memory_efficient_attention', None)
                    if final_value is not False:
                        logger.warning(f"  WARNING: use_memory_efficient_attention is still {final_value}, may need xformers")
                    else:
                        logger.info(f"  Verified: use_memory_efficient_attention = {final_value}")
                    
                    # Load model with the modified config
                    logger.info("Loading model with modified config")
                    try:
                        self.model = AutoModel.from_pretrained(
                            model_name,
                            config=config,
                            cache_dir=local_cache,
                            add_pooling_layer=False,
                            trust_remote_code=True
                        )
                    except AssertionError as e:
                        if "xformers" in str(e).lower() or "please install xformers" in str(e):
                            # Still wants xformers - config mod didn't work
                            logger.error("=" * 60)
                            logger.error("XFORMERS ERROR: Config modification didn't prevent xformers requirement")
                            logger.error("=" * 60)
                            logger.error("The Snowflake model's custom code is still requiring xformers.")
                            logger.error("")
                            logger.error("Possible solutions:")
                            logger.error("1. Install xformers (may not work on macOS):")
                            logger.error("   pip install xformers")
                            logger.error("")
                            logger.error("2. Use a different embedding model that doesn't require xformers")
                            logger.error("   Example: 'sentence-transformers/all-MiniLM-L6-v2'")
                            logger.error("")
                            logger.error("3. Check Hugging Face discussions for workarounds:")
                            logger.error("   https://huggingface.co/Snowflake/snowflake-arctic-embed-m-v2.0/discussions")
                            logger.error("=" * 60)
                            raise RuntimeError(
                                "Snowflake model requires xformers, but xformers is not available. "
                                "Please install xformers or use a different embedding model."
                            ) from e
                        else:
                            raise
                
                self.model.eval()
                
                # Determine device
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                self.model = self.model.to(self.device)
                logger.info(f"Model loaded on device: {self.device}")
                
                # Figure out the embedding dimension by testing
                test_text = "test"
                test_tokens = self.tokenizer([test_text], padding=True, truncation=True, 
                                            return_tensors='pt', max_length=8192)
                test_tokens = {k: v.to(self.device) for k, v in test_tokens.items()}
                with torch.no_grad():
                    test_outputs = self.model(**test_tokens)
                    test_last_hidden = test_outputs[0]
                    # Mean pooling (same as in generate_embeddings)
                    test_attention_mask = test_tokens.get('attention_mask', None)
                    if test_attention_mask is not None:
                        test_mask_expanded = test_attention_mask.unsqueeze(-1).expand(test_last_hidden.size()).float()
                        test_sum = torch.sum(test_last_hidden * test_mask_expanded, dim=1)
                        test_sum_mask = torch.clamp(test_mask_expanded.sum(dim=1), min=1e-9)
                        test_output = test_sum / test_sum_mask
                    else:
                        test_output = test_last_hidden.mean(dim=1)
                self.embedding_dimension = test_output.shape[1]
                
            else:
                # Try sentence-transformers first
                # But check if it's actually a Snowflake model that we missed
                if "snowflake" in model_name.lower() or "arctic" in model_name.lower():
                    logger.warning(f"Looks like Snowflake model but detection missed it: {model_name}")
                    logger.warning("Trying transformers instead...")
                    self.is_snowflake_model = True
                    self._load_snowflake_model(model_name, local_cache)
                else:
                    # Regular sentence-transformers model
                    logger.info("Loading model with sentence-transformers")
                    try:
                        self.model = SentenceTransformer(
                            model_name, 
                            cache_folder=local_cache,
                            trust_remote_code=True  # Some models may need this
                        )
                        self.tokenizer = None
                        self.device = None
                        
                        # Get embedding dimension from the model
                        test_embedding = self.model.encode(["test"])
                        self.embedding_dimension = test_embedding.shape[1]
                    except Exception as st_error:
                        error_str = str(st_error).lower()
                        # Check if this is a Snowflake model that sentence-transformers can't handle
                        if "trust_remote_code" in error_str or "custom code" in error_str:
                            logger.warning(f"sentence-transformers cannot load {model_name} (requires custom code)")
                            logger.warning("This model likely requires transformers library. Attempting to load with transformers...")
                            # Check if it looks like a Snowflake model
                            if "snowflake" in model_name.lower() or "arctic" in model_name.lower():
                                self.is_snowflake_model = True
                                self._load_snowflake_model(model_name, local_cache)
                            else:
                                raise RuntimeError(
                                    f"Model {model_name} requires trust_remote_code=True but sentence-transformers cannot load it. "
                                    "This model may need to be loaded with transformers library directly."
                                ) from st_error
                        else:
                            raise
            
            logger.info(f"Model loaded successfully. Embedding dimension: {self.embedding_dimension}")
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            error_msg = str(e).lower()
            
            # Check if this looks like a Snowflake model that needs transformers
            if ("snowflake" in model_name.lower() or "arctic" in model_name.lower()) and \
               ("trust_remote_code" in error_msg or "custom code" in error_msg):
                logger.error("")
                logger.error("=" * 60)
                logger.error("MODEL LOADING ERROR - Snowflake Model Detected")
                logger.error("=" * 60)
                logger.error(f"The model {model_name} requires transformers library with trust_remote_code=True")
                logger.error("The code should automatically use transformers for Snowflake models.")
                logger.error("")
                logger.error("If you're seeing this, there may be an issue with model detection.")
                logger.error("Try restarting the notebook kernel to reload the updated code.")
                logger.error("=" * 60)
                raise RuntimeError(
                    f"Snowflake model {model_name} requires transformers library. "
                    "Please restart the notebook kernel and try again, or check that the model detection is working."
                ) from e
            elif self.is_snowflake_model:
                if "xformers" in error_msg or "flash" in error_msg:
                    logger.error("")
                    logger.error("=" * 60)
                    logger.error("XFORMERS ERROR DETECTED")
                    logger.error("=" * 60)
                    logger.error("xformers is NOT required for the Snowflake model to work!")
                    logger.error("=" * 60)
                logger.error("Make sure transformers is installed: pip install transformers")
                logger.error("Make sure torch is installed: pip install torch")
            else:
                logger.error("Make sure sentence-transformers is installed: pip install sentence-transformers")
            raise
    
    def _load_snowflake_model(self, model_name: str, local_cache: str):
        """Helper method to load Snowflake model with transformers."""
        logger.info("Loading Snowflake model with transformers library...")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=local_cache,
            trust_remote_code=True
        )
        
        # Load and modify config
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(
            model_name,
            cache_dir=local_cache,
            trust_remote_code=True
        )
        
        # Disable memory_efficient_attention
        config.use_memory_efficient_attention = False
        setattr(config, 'use_memory_efficient_attention', False)
        if hasattr(config, '__dict__'):
            config.__dict__['use_memory_efficient_attention'] = False
        
        try:
            config_dict = config.to_dict()
            config_dict['use_memory_efficient_attention'] = False
            config = type(config).from_dict(config_dict)
            config.use_memory_efficient_attention = False
        except Exception:
            pass  # Continue with modified config
        
        # Load model
        self.model = AutoModel.from_pretrained(
            model_name,
            config=config,
            cache_dir=local_cache,
            add_pooling_layer=False,
            trust_remote_code=True
        )
        self.model.eval()
        
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        
        # Get dimension
        test_text = "test"
        test_tokens = self.tokenizer([test_text], padding=True, truncation=True, 
                                    return_tensors='pt', max_length=8192)
        test_tokens = {k: v.to(self.device) for k, v in test_tokens.items()}
        with torch.no_grad():
            test_outputs = self.model(**test_tokens)
            test_last_hidden = test_outputs[0]
            test_attention_mask = test_tokens.get('attention_mask', None)
            if test_attention_mask is not None:
                test_mask_expanded = test_attention_mask.unsqueeze(-1).expand(test_last_hidden.size()).float()
                test_sum = torch.sum(test_last_hidden * test_mask_expanded, dim=1)
                test_sum_mask = torch.clamp(test_mask_expanded.sum(dim=1), min=1e-9)
                test_output = test_sum / test_sum_mask
            else:
                test_output = test_last_hidden.mean(dim=1)
        self.embedding_dimension = test_output.shape[1]
        logger.info(f"Snowflake model loaded successfully. Embedding dimension: {self.embedding_dimension}")
    
    def generate_embeddings(self, texts: list, is_query: bool = False, batch_size: int = 32, show_progress: bool = True) -> np.ndarray:
        """Generate embeddings for a list of texts."""
        if self.is_snowflake_model:
            # Snowflake needs special handling
            if is_query:
                # Add "query: " prefix (Snowflake requirement)
                query_prefix = "query: "
                texts = [f"{query_prefix}{text}" for text in texts]
            
            # Process in batches (memory issues otherwise)
            all_embeddings = []
            
            if show_progress:
                from tqdm import tqdm
                batch_iter = tqdm(range(0, len(texts), batch_size), desc="Generating embeddings")
            else:
                batch_iter = range(0, len(texts), batch_size)
            
            for i in batch_iter:
                batch_texts = texts[i:i + batch_size]
                
                # Tokenize
                tokens = self.tokenizer(batch_texts, padding=True, truncation=True, 
                                      return_tensors='pt', max_length=8192)
                tokens = {k: v.to(self.device) for k, v in tokens.items()}
                
                with torch.no_grad():
                    outputs = self.model(**tokens)
                    last_hidden_state = outputs[0]  # (batch_size, seq_len, hidden_size)
                    
                    # Mean pooling (averages all tokens, not just first one)
                    attention_mask = tokens.get('attention_mask', None)
                    if attention_mask is not None:
                        # Exclude padding tokens
                        attention_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
                        sum_embeddings = torch.sum(last_hidden_state * attention_mask_expanded, dim=1)
                        sum_mask = torch.clamp(attention_mask_expanded.sum(dim=1), min=1e-9)
                        batch_embeddings = (sum_embeddings / sum_mask).cpu().numpy()
                    else:
                        # No mask, just average
                        batch_embeddings = last_hidden_state.mean(dim=1).cpu().numpy()
                    
                    all_embeddings.append(batch_embeddings)
            
            embeddings = np.vstack(all_embeddings)
            return embeddings
        else:
            # Regular sentence-transformers
            embeddings = self.model.encode(texts, show_progress_bar=show_progress)
            return embeddings
    
    def get_dimension(self) -> int:
        """Returns embedding dimension."""
        return self.embedding_dimension

