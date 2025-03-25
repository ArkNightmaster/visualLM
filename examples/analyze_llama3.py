"""Example script for analyzing Llama-3 model weights and activations."""
import argparse
import json
import os
import sys
from typing import Dict, List, Optional

# Add the parent directory to sys.path to resolve imports
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

import matplotlib.pyplot as plt
import numpy as np
import torch
import seaborn as sns

from visuallm.adapters.llama import LlamaAnalyzer

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Analyze Llama-3 model weights and activations.")
    parser.add_argument("--model_path", type=str, default="/data2/Llama3", 
                      help="Path to the Llama-3 model")
    parser.add_argument("--output_dir", type=str, default="./outputs",
                      help="Directory to save analysis results")
    parser.add_argument("--layers", type=str, default=None,
                      help="Comma-separated list of layer indices to analyze (e.g., '0,1,2'), or 'all' for all layers")
    parser.add_argument("--use_4bit", action="store_true",
                      help="Whether to load model in 4-bit precision")
    parser.add_argument("--use_flash_attention", action="store_true",
                      help="Whether to use flash attention for faster inference (requires flash_attn package)")
    parser.add_argument("--prompt", type=str, default="Once upon a time in a land far away",
                      help="Prompt for activation analysis")
    return parser.parse_args()

def plot_weight_distribution(weight_stats: Dict[str, Dict[str, float]], output_dir: str, layer_idx: Optional[int] = None):
    """Plot weight distribution statistics.
    
    Args:
        weight_stats: Dictionary of weight statistics
        output_dir: Directory to save plots
        layer_idx: Optional specific layer to plot
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract parameter names and statistics
    param_names = []
    means = []
    stds = []
    mins = []
    maxs = []
    sparsities = []
    skewness = []
    kurtosis = []
    
    for name, stats in weight_stats.items():
        if layer_idx is not None and f"layers.{layer_idx}." not in name:
            continue
            
        param_names.append(name)
        means.append(stats['mean'])
        stds.append(stats['std'])
        mins.append(stats['min'])
        maxs.append(stats['max'])
        sparsities.append(stats['sparsity'])
        skewness.append(stats['skewness'])
        kurtosis.append(stats['kurtosis'])
    
    # Plot means and standard deviations
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.bar(range(len(param_names)), means, alpha=0.7)
    plt.ylabel('Mean')
    plt.title('Mean of Parameter Weights')
    plt.xticks([])
    
    plt.subplot(2, 1, 2)
    plt.bar(range(len(param_names)), stds, alpha=0.7, color='orange')
    plt.ylabel('Standard Deviation')
    plt.title('Standard Deviation of Parameter Weights')
    plt.xticks([])
    
    layer_str = f"_layer{layer_idx}" if layer_idx is not None else ""
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"weight_mean_std{layer_str}.png"))
    plt.close()
    
    # Plot min, max values
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.bar(range(len(param_names)), mins, alpha=0.7, color='red')
    plt.ylabel('Min')
    plt.title('Minimum Values of Parameter Weights')
    plt.xticks([])
    
    plt.subplot(2, 1, 2)
    plt.bar(range(len(param_names)), maxs, alpha=0.7, color='green')
    plt.ylabel('Max')
    plt.title('Maximum Values of Parameter Weights')
    plt.xticks([])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"weight_min_max{layer_str}.png"))
    plt.close()
    
    # Plot skewness and kurtosis
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.bar(range(len(param_names)), skewness, alpha=0.7, color='purple')
    plt.ylabel('Skewness')
    plt.title('Skewness of Parameter Weights')
    plt.xticks([])
    
    plt.subplot(2, 1, 2)
    plt.bar(range(len(param_names)), kurtosis, alpha=0.7, color='brown')
    plt.ylabel('Kurtosis')
    plt.title('Excess Kurtosis of Parameter Weights')
    plt.xticks([])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"weight_skew_kurt{layer_str}.png"))
    plt.close()
    
    # Plot sparsity
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(param_names)), sparsities, alpha=0.7, color='teal')
    plt.ylabel('Sparsity')
    plt.title('Sparsity of Parameter Weights')
    plt.xticks([])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"weight_sparsity{layer_str}.png"))
    plt.close()

def plot_activation_distribution(activation_stats: Dict[str, Dict[str, float]], output_dir: str):
    """Plot activation distribution statistics.
    
    Args:
        activation_stats: Dictionary of activation statistics
        output_dir: Directory to save plots
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Group activations by layer and component
    layer_stats = {}
    for name, stats in activation_stats.items():
        if "layer_" in name:
            # Extract layer number
            parts = name.split("_")
            layer_idx = int(parts[1])
            if layer_idx not in layer_stats:
                layer_stats[layer_idx] = {}
            
            # Extract component type
            if "attn" in name:
                component = "attention"
            elif "mlp" in name:
                component = "mlp"
            else:
                component = "other"
            
            if component not in layer_stats[layer_idx]:
                layer_stats[layer_idx][component] = []
            
            layer_stats[layer_idx][component].append(stats)
    
    # Plot statistics by layer
    for layer_idx, components in layer_stats.items():
        # Plot means across components
        plt.figure(figsize=(15, 10))
        
        # Plot means
        plt.subplot(2, 2, 1)
        for component, stats_list in components.items():
            means = [s['mean'] for s in stats_list]
            plt.plot(range(len(means)), means, 'o-', label=component)
        plt.title(f'Layer {layer_idx} - Mean Activations')
        plt.legend()
        
        # Plot standard deviations
        plt.subplot(2, 2, 2)
        for component, stats_list in components.items():
            stds = [s['std'] for s in stats_list]
            plt.plot(range(len(stds)), stds, 'o-', label=component)
        plt.title(f'Layer {layer_idx} - Standard Deviation of Activations')
        plt.legend()
        
        # Plot sparsity
        plt.subplot(2, 2, 3)
        for component, stats_list in components.items():
            sparsities = [s['sparsity'] for s in stats_list]
            plt.plot(range(len(sparsities)), sparsities, 'o-', label=component)
        plt.title(f'Layer {layer_idx} - Activation Sparsity')
        plt.legend()
        
        # Plot skewness
        plt.subplot(2, 2, 4)
        for component, stats_list in components.items():
            skews = [s['skewness'] for s in stats_list]
            plt.plot(range(len(skews)), skews, 'o-', label=component)
        plt.title(f'Layer {layer_idx} - Activation Skewness')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"activations_layer_{layer_idx}.png"))
        plt.close()

def main():
    """Main function."""
    args = parse_args()
    
    # Parse layers
    layers = None
    if args.layers:
        if args.layers.lower() == 'all':
            layers = None
        else:
            layers = [int(l) for l in args.layers.split(',')]
    
    print(f"Analyzing Llama-3 model at: {args.model_path}")
    print(f"Loading model...")
    
    # Initialize the Llama analyzer
    analyzer = LlamaAnalyzer(
        model_path=args.model_path,
        load_in_4bit=args.use_4bit,
        use_flash_attention=args.use_flash_attention,
    )
    
    # Get model structure
    model_info = analyzer.analyze_model_structure()
    print("\nModel information:")
    for key, value in model_info.items():
        print(f"  {key}: {value}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save model info
    with open(os.path.join(args.output_dir, "model_info.json"), 'w') as f:
        json.dump(model_info, f, indent=2)
    
    # Analyze weights
    print("\nAnalyzing model weights...")
    weight_stats = analyzer.get_weight_statistics(layers=layers)
    
    # Save weight stats
    with open(os.path.join(args.output_dir, "weight_stats.json"), 'w') as f:
        # Convert values that might not be JSON serializable (like numpy arrays)
        serializable_stats = {}
        for name, stats in weight_stats.items():
            serializable_stats[name] = {k: float(v) for k, v in stats.items()}
        json.dump(serializable_stats, f, indent=2)
    
    # Plot weight distributions
    print("Plotting weight distributions...")
    plot_weight_distribution(weight_stats, os.path.join(args.output_dir, "weights"))
    
    # Plot per-layer weight distributions if analyzing all layers
    if layers is None or len(layers) > 1:
        for layer_idx in range(analyzer.num_layers):
            plot_weight_distribution(weight_stats, os.path.join(args.output_dir, "weights/by_layer"), layer_idx)
    
    # Run inference and analyze activations
    print(f"\nRunning inference with prompt: {args.prompt}")
    output_text, _ = analyzer.run_inference(
        input_text=args.prompt,
        max_length=20,
        capture_activations=True,
        layers=layers
    )
    print(f"Model output: {output_text}")
    
    # Get activation statistics
    print("\nAnalyzing activations...")
    activation_stats = analyzer.get_activation_statistics()
    
    # Save activation stats
    with open(os.path.join(args.output_dir, "activation_stats.json"), 'w') as f:
        # Convert values that might not be JSON serializable
        serializable_stats = {}
        for name, stats in activation_stats.items():
            serializable_stats[name] = {k: float(v) for k, v in stats.items()}
        json.dump(serializable_stats, f, indent=2)
    
    # Plot activation distributions
    print("Plotting activation distributions...")
    plot_activation_distribution(activation_stats, os.path.join(args.output_dir, "activations"))
    
    print(f"\nAnalysis complete. Results saved to: {args.output_dir}")

if __name__ == "__main__":
    main() 