# VisualLM: Large Language Model Weight and Activation Analysis Tool

VisualLM is a Python library designed to help researchers analyze and visualize weight distributions and activation patterns in Large Language Models (LLMs). The library provides comprehensive tools for statistical analysis, anomaly detection, and advanced visualization for various model architectures.

## 🌟 Features

- **Multi-Architecture Support**
  - Decoder-only models (GPT family, Llama)
  - Encoder-decoder models (T5, BART)
  - Encoder-only models (BERT, RoBERTa)

- **Weight Analysis**
  - Statistical measures (mean, std, min, max, skewness, kurtosis)
  - Distribution analysis
  - Anomaly detection
  - Cross-layer comparison

- **Activation Analysis**
  - Layer-wise activation patterns
  - Neuron firing statistics
  - Feature importance analysis
  - Activation sparsity metrics

- **Advanced Visualization**
  - Distribution plots
  - Layer-wise heatmaps
  - SVD visualization
  - Attention pattern maps
  - Publication-ready outputs

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/username/visuallm.git
cd visuallm

# Install the package
pip install -e .
```

### Example: Analyzing Llama-3

```python
from visuallm.adapters.llama import LlamaAnalyzer

# Initialize analyzer
analyzer = LlamaAnalyzer(
    model_path="/path/to/llama3-model",
    use_flash_attention=True
)

# Get model info
model_info = analyzer.analyze_model_structure()
print(model_info)

# Analyze weights
weight_stats = analyzer.get_weight_statistics()

# Run inference and capture activations
text, activations = analyzer.run_inference(
    input_text="Once upon a time",
    max_length=20,
    capture_activations=True
)

# Get activation statistics
activation_stats = analyzer.get_activation_statistics()
```

### Running the Example Script

```bash
# Analyze Llama-3 model
python examples/analyze_llama3.py --model_path /data2/Llama3/llama3-8b --output_dir ./outputs/llama3-analysis

# Analyze specific layers
python examples/analyze_llama3.py --model_path /data2/Llama3/llama3-8b --layers 0,1,2

# Use 4-bit quantization for memory efficiency
python examples/analyze_llama3.py --model_path /data2/Llama3/llama3-8b --use_4bit
```

## 📊 Visualization Examples

### Weight Distributions
![Weight Distributions](./outputs/weights/weight_mean_std.png)

### Activation Patterns
![Activation Patterns](./outputs/activations/activations_layer_0.png)

## 📖 Documentation

For detailed documentation, see the [docs](docs/) directory.

## 🧩 Architecture

VisualLM follows a modular design:

```
visuallm/
├── core/            # Core functionality
│   ├── models/      # Base model abstractions
│   ├── analysis/    # Statistical analysis tools
│   └── utils/       # Utility functions
├── viz/             # Visualization components
│   ├── plots/       # Basic plotting utilities
│   └── interactive/ # Interactive visualization tools
└── adapters/        # Model-specific adapters
    ├── gpt.py
    ├── llama.py
    └── bert.py
```

## 🛠️ Requirements

- Python 3.8+
- PyTorch 2.0.1+
- Transformers 4.30.0+
- NumPy, Pandas, Matplotlib, Seaborn, Plotly
- SciPy, StatsModels
- Jupyter (for notebooks)
- Accelerate (for efficient model loading)

## ✅ TODO

- [ ] Add support for BERT
- [ ] Add support for BART
- [ ] Add support for T5
- [ ] Add support for ViT

## 💡 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the [LICENSE](LICENSE) file in the repository.

## 📧 Contact

For questions or feedback, please open an issue on GitHub. 