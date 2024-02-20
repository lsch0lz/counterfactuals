<div align="center">

![logo](https://raw.githubusercontent.com/lsch0lz/counterfactuals/feature/initial-setup/docs/counterfactuals.jpg)

Counterfactuals: Take the uncertainty out of your machine learning models

<h3>

[Documentation](/docs) | [Examples](/examples) | [Showcase](/docs/showcase.md)

</h3>

</div>

---

Counterfactuals is a Python library for machine learning that enables you to better understand your models.
We combine several techniques to provide a comprehensive understanding of your model's predictions.
With those insights you are able to eliminate uncertainties and make better decisions.

## Features

### CLUE: A Method for Explaining Uncertainty Estimates

CLUE is a method for explaining uncertainty estimates of machine learning models. It is based on the idea of counterfactuals and provides a comprehensive understanding of the model's predictions.

Model Paper: [CLUE: A Method for Explaining Uncertainty Estimates](https://arxiv.org/abs/2006.06848)



## Installation

The current recommended way to install counterfactual is from source.

### From source

```sh
git clone https://github.com/lsch0lz/counterfactuals.git
cd counterfactuals
python3 -m pip install -e .
```

### Direct (master)

```sh
python3 -m pip install git+https://github.com/lsch0lz/counterfactuals.git
```

## Documentation

Documentation along with a quick start guide can be found in the [docs/](/docs) directory.