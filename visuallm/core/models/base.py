"""Base classes for model analysis."""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union

import torch
from torch import nn
from transformers import PreTrainedModel

class ModelAnalyzer(ABC):
    """Base class for analyzing model weights and activations."""

    def __init__(
        self,
        model: PreTrainedModel,
        device: Optional[Union[str, torch.device]] = None,
    ):
        """Initialize the model analyzer.

        Args:
            model: The pre-trained model to analyze
            device: Device to place the model on
        """
        self.model = model
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.activation_hooks = {}
        self.stored_activations = {}

    @abstractmethod
    def register_hooks(self) -> None:
        """Register hooks for capturing activations."""
        pass

    @abstractmethod
    def remove_hooks(self) -> None:
        """Remove all registered hooks."""
        pass

    def _store_activation(self, name: str) -> callable:
        """Create a hook for storing activations.

        Args:
            name: Name of the layer
        
        Returns:
            Hook function
        """
        def hook(module: nn.Module, input: torch.Tensor, output: torch.Tensor) -> None:
            self.stored_activations[name] = output.detach()
        return hook

    def get_weight_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get basic statistics for all weights in the model.

        Returns:
            Dictionary containing statistics for each weight tensor
        """
        stats = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                stats[name] = {
                    'mean': param.data.mean().item(),
                    'std': param.data.std().item(),
                    'min': param.data.min().item(),
                    'max': param.data.max().item(),
                    'sparsity': (param.data == 0).float().mean().item(),
                }
        return stats

    def get_activation_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get basic statistics for stored activations.

        Returns:
            Dictionary containing statistics for each activation tensor
        """
        stats = {}
        for name, activation in self.stored_activations.items():
            stats[name] = {
                'mean': activation.mean().item(),
                'std': activation.std().item(),
                'min': activation.min().item(),
                'max': activation.max().item(),
                'sparsity': (activation == 0).float().mean().item(),
            }
        return stats

    def clear_stored_activations(self) -> None:
        """Clear all stored activations."""
        self.stored_activations.clear() 