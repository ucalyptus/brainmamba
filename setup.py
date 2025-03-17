from setuptools import setup, find_packages

setup(
    name="brainmamba",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "tqdm>=4.60.0",
        "einops>=0.4.0",
        "requests>=2.25.0",
        "nibabel>=3.2.0",
        "nilearn>=0.8.0",
    ],
    author="BrainMamba Team",
    author_email="example@example.com",
    description="BrainMamba: A Selective State Space Model for Brain Dynamics",
    keywords="brain, mamba, fmri, deep learning",
    python_requires=">=3.8",
) 