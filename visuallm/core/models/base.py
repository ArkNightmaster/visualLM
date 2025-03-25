"""Base classes for model analyzers."""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Any

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
        if hasattr(self.model, "to") and not isinstance(self.device, str):
            self.model.to(self.device)

    @abstractmethod
    def register_hooks(self, layers: Optional[List[int]] = None) -> None:
        """Register hooks for capturing activations.
        
        Args:
            layers: List of layer indices to register hooks for, or None for all layers
        """
        pass

    @abstractmethod
    def remove_hooks(self) -> None:
        """Remove all registered hooks."""
        pass

    @abstractmethod
    def get_weight_statistics(self, layers: Optional[List[int]] = None) -> Dict[str, Dict[str, float]]:
        """Get statistics for model weights, with optional layer filtering.
        
        Args:
            layers: List of layer indices to get statistics for, or None for all layers
            
        Returns:
            Dictionary of weight statistics
        """
        pass

    @abstractmethod
    def get_activation_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for stored activations.
        
        Returns:
            Dictionary of activation statistics
        """
        pass 