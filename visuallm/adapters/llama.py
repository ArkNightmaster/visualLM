"""Adapter for Llama models, including Llama-3."""
from typing import Dict, List, Optional, Tuple, Union, Any
import os

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from visuallm.core.models.base import ModelAnalyzer
from visuallm.core.utils.hooks import ActivationStore

class LlamaAnalyzer(ModelAnalyzer):
    """Analyzer for Llama models, optimized for Llama-3.
    
    This class provides specialized methods for analyzing Llama architecture models,
    with specific support for Llama-3's architecture.
    """
    
    def __init__(
        self,
        model_path: str,
        device: Optional[Union[str, torch.device]] = None,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        use_flash_attention: bool = False,  # Default to False since flash_attn may not be installed
    ):
        """Initialize a Llama analyzer.
        
        Args:
            model_path: Path to the Llama model or model name
            device: Device to place the model on
            load_in_8bit: Whether to load the model in 8-bit precision
            load_in_4bit: Whether to load the model in 4-bit precision
            use_flash_attention: Whether to use flash attention for faster inference (requires flash_attn package)
        """
        # Validate the model path
        if not os.path.exists(model_path):
            raise ValueError(f"Model path does not exist: {model_path}")
            
        # Check if the model path contains required files
        config_path = os.path.join(model_path, "config.json")
        if not os.path.exists(config_path):
            raise ValueError(f"Model directory does not contain config.json: {model_path}")
            
        print(f"Loading model from: {model_path}")
        print(f"Using device: {device or 'auto'}")
        
        # Configure quantization parameters
        quantization_config = None
        if load_in_8bit or load_in_4bit:
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=load_in_8bit,
                load_in_4bit=load_in_4bit,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
        
        # Configure attention implementation
        attn_implementation = "flash_attention_2" if use_flash_attention else "eager"
        if use_flash_attention:
            try:
                import flash_attn
                print(f"Using Flash Attention 2 for faster inference")
            except ImportError:
                print(f"Warning: flash_attn package not found. Falling back to eager implementation.")
                attn_implementation = "eager"
        
        try:
            # Load the model with accelerate and local path
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map="auto" if device is None else device,
                torch_dtype=torch.float16,
                quantization_config=quantization_config,
                attn_implementation=attn_implementation,
                local_files_only=True,  # Ensure we don't try to download from Hub
                trust_remote_code=True,  # Trust custom code if present
            )
            
            # Initialize the tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                local_files_only=True,
                trust_remote_code=True,
            )
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
        
        # Initialize the base class
        super().__init__(model, device)
        
        # Create an activation store
        self.activation_store = ActivationStore()
        
        # Model-specific attributes
        self.num_layers = self._get_num_layers()
        self.hidden_size = self._get_hidden_size()
        self.model_type = self._identify_model_subtype()
        
        print(f"Model loaded successfully: {self.model_type}, {self.num_layers} layers")
    
    def _get_num_layers(self) -> int:
        """Get the number of layers in the model.
        
        Returns:
            Number of transformer layers
        """
        if hasattr(self.model, "config"):
            if hasattr(self.model.config, "num_hidden_layers"):
                return self.model.config.num_hidden_layers
            elif hasattr(self.model.config, "n_layer"):
                return self.model.config.n_layer
        
        # Fallback: count the layers
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            return len(self.model.model.layers)
        
        raise ValueError("Could not determine the number of layers")
    
    def _get_hidden_size(self) -> int:
        """Get the hidden size of the model.
        
        Returns:
            Hidden size dimension
        """
        if hasattr(self.model, "config") and hasattr(self.model.config, "hidden_size"):
            return self.model.config.hidden_size
        
        raise ValueError("Could not determine the hidden size")
    
    def _identify_model_subtype(self) -> str:
        """Identify the specific subtype of the Llama model.
        
        Returns:
            Model subtype ("llama", "llama-2", "llama-3", etc.)
        """
        if hasattr(self.model, "config") and hasattr(self.model.config, "model_type"):
            model_type = self.model.config.model_type
            
            # Check for version in config
            if hasattr(self.model.config, "_name_or_path"):
                path = self.model.config._name_or_path.lower()
                if "llama-3" in path or "llama3" in path:
                    return "llama-3"
                elif "llama-2" in path or "llama2" in path:
                    return "llama-2"
            
            return model_type
        
        return "unknown"
    
    def register_hooks(self, layers: Optional[List[int]] = None) -> None:
        """Register hooks for the specified layers.
        
        Args:
            layers: List of layer indices to register hooks for, or None for all layers
        """
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            model_layers = self.model.model.layers
            if layers is None:
                layers = list(range(len(model_layers)))
            
            # Register hooks for each specified layer
            for layer_idx in layers:
                if 0 <= layer_idx < len(model_layers):
                    layer = model_layers[layer_idx]
                    
                    # Hook into attention components
                    self.activation_store.register_hook(
                        layer.self_attn.q_proj, 
                        f"layer_{layer_idx}_attn_q_proj"
                    )
                    self.activation_store.register_hook(
                        layer.self_attn.k_proj, 
                        f"layer_{layer_idx}_attn_k_proj"
                    )
                    self.activation_store.register_hook(
                        layer.self_attn.v_proj, 
                        f"layer_{layer_idx}_attn_v_proj"
                    )
                    self.activation_store.register_hook(
                        layer.self_attn.o_proj, 
                        f"layer_{layer_idx}_attn_o_proj"
                    )
                    
                    # Hook into MLP components
                    self.activation_store.register_hook(
                        layer.mlp.gate_proj, 
                        f"layer_{layer_idx}_mlp_gate_proj"
                    )
                    self.activation_store.register_hook(
                        layer.mlp.up_proj, 
                        f"layer_{layer_idx}_mlp_up_proj"
                    )
                    self.activation_store.register_hook(
                        layer.mlp.down_proj, 
                        f"layer_{layer_idx}_mlp_down_proj"
                    )
                    
                    # Output of attention block
                    self.activation_store.register_hook(
                        layer.self_attn, 
                        f"layer_{layer_idx}_attn_output",
                        retain_input=True
                    )
                    
                    # Output of the entire layer
                    self.activation_store.register_hook(
                        layer, 
                        f"layer_{layer_idx}_output",
                        retain_input=True
                    )
    
    def remove_hooks(self) -> None:
        """Remove all registered hooks."""
        self.activation_store.remove_hooks()
    
    def get_weight_statistics(self, layers: Optional[List[int]] = None) -> Dict[str, Dict[str, float]]:
        """Get statistics for model weights, with optional layer filtering.
        
        Args:
            layers: List of layer indices to get statistics for, or None for all layers
            
        Returns:
            Dictionary of weight statistics
        """
        stats = {}
        
        # Filter parameter names by layer if specified
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if layers is not None:
                    # Check if the parameter belongs to one of the specified layers
                    layer_match = False
                    for layer_idx in layers:
                        if f"layers.{layer_idx}." in name:
                            layer_match = True
                            break
                    
                    if not layer_match:
                        continue
                
                stats[name] = {
                    'mean': param.data.mean().item(),
                    'std': param.data.std().item(),
                    'min': param.data.min().item(),
                    'max': param.data.max().item(),
                    'sparsity': (param.data == 0).float().mean().item(),
                    'skewness': self._compute_skewness(param.data).item(),
                    'kurtosis': self._compute_kurtosis(param.data).item(),
                }
        
        return stats
    
    def get_activation_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for stored activations.
        
        Returns:
            Dictionary of activation statistics
        """
        stats = {}
        for name, activation in self.activation_store.get_all_activations().items():
            # Skip non-tensor activations or tuples
            if not isinstance(activation, torch.Tensor):
                continue
                
            stats[name] = {
                'mean': activation.mean().item(),
                'std': activation.std().item(),
                'min': activation.min().item(),
                'max': activation.max().item(),
                'sparsity': (activation == 0).float().mean().item(),
                'skewness': self._compute_skewness(activation).item(),
                'kurtosis': self._compute_kurtosis(activation).item(),
            }
        
        return stats
    
    def _compute_skewness(self, tensor: torch.Tensor) -> torch.Tensor:
        """Compute the skewness of a tensor.
        
        Args:
            tensor: Input tensor
            
        Returns:
            Skewness value
        """
        # Center the data
        centered = tensor - tensor.mean()
        # Compute the third moment
        third_moment = torch.mean(centered ** 3)
        # Normalize by the standard deviation cubed
        std = tensor.std()
        if std == 0:
            return torch.tensor(0.0, device=tensor.device)
        return third_moment / (std ** 3)
    
    def _compute_kurtosis(self, tensor: torch.Tensor) -> torch.Tensor:
        """Compute the kurtosis of a tensor.
        
        Args:
            tensor: Input tensor
            
        Returns:
            Kurtosis value
        """
        # Center the data
        centered = tensor - tensor.mean()
        # Compute the fourth moment
        fourth_moment = torch.mean(centered ** 4)
        # Normalize by the variance squared
        var = tensor.var()
        if var == 0:
            return torch.tensor(0.0, device=tensor.device)
        # Subtract 3 to get excess kurtosis (normal distribution has kurtosis of 3)
        return fourth_moment / (var ** 2) - 3
    
    def run_inference(
        self, 
        input_text: str, 
        max_length: int = 20,
        capture_activations: bool = True,
        layers: Optional[List[int]] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """Run inference on the model and optionally capture activations.
        
        Args:
            input_text: The input text to process
            max_length: Maximum length of generated sequence
            capture_activations: Whether to capture activations during inference
            layers: Which layers to capture activations for (None for all)
            
        Returns:
            Tuple of (generated_text, activations_dict)
        """
        # Clear existing activations
        self.activation_store.clear_activations()
        
        # Register hooks if capturing activations
        if capture_activations:
            self.register_hooks(layers)
        
        # Tokenize input
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)
        
        # Run inference
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=inputs["input_ids"].shape[1] + max_length,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        # Remove hooks if they were registered
        if capture_activations:
            self.remove_hooks()
        
        # Decode output
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return generated_text, self.activation_store.get_all_activations()
    
    def analyze_model_structure(self) -> Dict[str, Any]:
        """Analyze the model structure and return key information.
        
        Returns:
            Dictionary with model structure information
        """
        structure = {
            'model_type': self.model_type,
            'num_parameters': sum(p.numel() for p in self.model.parameters()),
            'num_trainable_parameters': sum(p.numel() for p in self.model.parameters() if p.requires_grad),
            'num_layers': self.num_layers,
            'hidden_size': self.hidden_size,
        }
        
        # Add model-specific information
        if hasattr(self.model, 'config'):
            if hasattr(self.model.config, 'vocab_size'):
                structure['vocab_size'] = self.model.config.vocab_size
            if hasattr(self.model.config, 'num_attention_heads'):
                structure['num_attention_heads'] = self.model.config.num_attention_heads
            if hasattr(self.model.config, 'intermediate_size'):
                structure['intermediate_size'] = self.model.config.intermediate_size
            
        return structure