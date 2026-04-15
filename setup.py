from setuptools import setup, find_packages

setup(
    name="neuro-ai-bridge",
    version="0.1.0",
    author="Dimitri Romanov",
    description="Mapping neuroscientific principles onto deep learning architectures",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24.0",
        "matplotlib>=3.7.0",
        "scipy>=1.10.0",
    ],
)
