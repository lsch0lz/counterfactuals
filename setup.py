from pathlib import Path

from setuptools import setup, find_packages

VERSION = "0.0.1"
DESCRIPTION = "Remove uncertainty from your machine learning models"

directory = Path(__file__).resolve().parent
with open(directory / 'README.md', encoding='utf-8') as f:
    long_description = f.read()

# Setting up
setup(
    # the name must match the folder name 'verysimplemodule'
    name="counterfactual_xai",
    version=VERSION,
    author="Lukas Scholz",
    description=DESCRIPTION,
    long_description=long_description,
    packages=find_packages(),
    install_requires=["numpy", "tqdm", "torch"],

    keywords=['python'],
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
    python_requires='>=3.8',
    extras_require={'linting': [
        "pylint",
        "mypy",
        "typing-extensions",
        "pre-commit",
        "ruff",
        "types-tqdm",
    ]
    }
)
