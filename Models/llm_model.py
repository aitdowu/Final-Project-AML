import os
import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QwenLLM:
    """Wrapper for Qwen2-1.5B-Instruct LLM."""
    
    def __init__(self, model_name: str = "Qwen/Qwen2-1.5B-Instruct", use_quantization: bool = True):
        """Initialize Qwen2. use_quantization helps on CPU."""
        self.model_name = model_name
        self.device = self._get_device()
        self.use_quantization = use_quantization and self.device == "cpu"
        
        # Get HF token if set (usually not needed)
        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
        
        # Local cache
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        local_cache = os.path.join(project_root, ".cache", "huggingface")
        os.makedirs(local_cache, exist_ok=True)
        
        logger.info(f"Loading Qwen2: {model_name} on {self.device}")
        if self.use_quantization:
            logger.info("Using 8-bit quantization")
        
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True,
                token=hf_token if hf_token else None,
                cache_dir=local_cache
            )
            
            # Set pad token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            model_kwargs = {
                "trust_remote_code": True,
                "token": hf_token if hf_token else None,
                "cache_dir": local_cache
            }
            
            if self.device == "cuda":
                # GPU - use float16
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    **model_kwargs
                )
            elif self.device == "mps":
                # Apple Silicon - float32
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float32,
                    **model_kwargs
                )
                self.model = self.model.to(self.device)
            else:
                # CPU - try quantization (doesn't work on mac though)
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
        """Generate text from prompt."""
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

