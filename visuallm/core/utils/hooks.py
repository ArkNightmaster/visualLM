"""Utilities for managing PyTorch hooks."""
from typing import Any, Callable, Dict, Optional, Set, Union, List
import weakref

import torch
from torch import nn

class ActivationStore:
    """Manages the storage and retrieval of model activations."""
    
    def __init__(self):
        """Initialize an empty activation store."""
        self._activations: Dict[str, Any] = {}
        self._hooks: Dict[str, Any] = {}
        self._modules: Set[weakref.ReferenceType] = set()
    
    def register_hook(
        self, 
        module: nn.Module, 
        name: str, 
        hook_type: str = "forward",
        retain_input: bool = False,
        retain_output: bool = True
    ) -> None:
        """Register a hook on a module to capture activations.
        
        Args:
            module: The module to attach the hook to
            name: A unique name for this hook/activation
            hook_type: Type of hook ("forward", "backward", or "full")
            retain_input: Whether to store the input activations
            retain_output: Whether to store the output activations
        """
        if name in self._hooks:
            raise ValueError(f"Hook with name {name} already exists")
            
        # Keep a weak reference to avoid memory leaks
        self._modules.add(weakref.ref(module))
        
        def _make_hook(store, name, retain_input, retain_output):
            if hook_type == "forward":
                def hook(module, inputs, outputs):
                    if retain_input:
                        if isinstance(inputs, tuple) and len(inputs) == 1:
                            store._activations[f"{name}_input"] = inputs[0].detach()
                        else:
                            store._activations[f"{name}_input"] = tuple(x.detach() if isinstance(x, torch.Tensor) else x for x in inputs)
                    
                    if retain_output:
                        store._activations[name] = outputs.detach() if isinstance(outputs, torch.Tensor) else outputs
                return hook
            
            elif hook_type == "backward":
                def hook(module, grad_input, grad_output):
                    if retain_input and grad_input is not None:
                        if isinstance(grad_input, tuple) and len(grad_input) == 1:
                            store._activations[f"{name}_grad_input"] = grad_input[0].detach()
                        else:
                            store._activations[f"{name}_grad_input"] = tuple(x.detach() if isinstance(x, torch.Tensor) and x is not None else x for x in grad_input)
                    
                    if retain_output and grad_output is not None:
                        if isinstance(grad_output, tuple) and len(grad_output) == 1:
                            store._activations[name] = grad_output[0].detach()
                        else:
                            store._activations[name] = tuple(x.detach() if isinstance(x, torch.Tensor) and x is not None else x for x in grad_output)
                return hook
            
            else:  # "full" - register both hooks
                raise ValueError("Full hooks not yet implemented")
        
        if hook_type == "forward":
            hook = module.register_forward_hook(
                _make_hook(self, name, retain_input, retain_output)
            )
        elif hook_type == "backward":
            hook = module.register_full_backward_hook(
                _make_hook(self, name, retain_input, retain_output)
            )
        else:
            raise ValueError(f"Unknown hook type: {hook_type}")
            
        self._hooks[name] = hook
    
    def get_activation(self, name: str) -> Any:
        """Get a stored activation by name.
        
        Args:
            name: The name of the activation to retrieve
            
        Returns:
            The stored activation
            
        Raises:
            KeyError: If no activation with the given name exists
        """
        if name not in self._activations:
            raise KeyError(f"No activation found with name {name}")
        return self._activations[name]
    
    def get_all_activations(self) -> Dict[str, Any]:
        """Get all stored activations.
        
        Returns:
            Dictionary mapping names to activations
        """
        return self._activations.copy()
    
    def clear_activations(self) -> None:
        """Clear all stored activations."""
        self._activations.clear()
    
    def remove_hooks(self, names: Optional[List[str]] = None) -> None:
        """Remove hooks by name.
        
        Args:
            names: List of hook names to remove, or None to remove all
        """
        if names is None:
            names = list(self._hooks.keys())
            
        for name in names:
            if name in self._hooks:
                self._hooks[name].remove()
                del self._hooks[name]
    
    def __del__(self):
        """Clean up by removing all hooks."""
        for hook in self._hooks.values():
            hook.remove()
        self._hooks.clear()
        self._activations.clear() 