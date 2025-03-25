from setuptools import setup, find_packages

setup(
    name="visuallm",
    version="0.1.0",
    description="A library for analyzing and visualizing LLM weights and activations",
    author="Research Team",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.1",
        "numpy>=1.20.0",
        "pandas>=1.5.0",
        "scipy>=1.8.0",
        "matplotlib>=3.5.0",
        "plotly>=5.3.0",
        "transformers>=4.30.0",
        "statsmodels>=0.13.0",
        "seaborn>=0.12.0",
        "jupyter>=1.0.0",
        "accelerate>=0.20.0",
    ],
    python_requires=">=3.8",
) 