# Llama Recipes: Examples to get started using the Llama models from Meta

llama documentation: https://www.llama.com/docs/overview/

This repository contains example scripts and notebooks to get started with the models in a variety of use-cases, including fine-tuning for domain adaptation and building LLM-based applications with Llama and other tools in the LLM ecosystem. The examples here use Llama locally, in the cloud, and on-prem.

## Table of Contents

- [Llama Recipes: Examples to get started using the Llama models from Meta](#llama-recipes-examples-to-get-started-using-the-llama-models-from-meta)
  - [Table of Contents](#table-of-contents)
  - [Configuration Environment](#configuration-environment)
  - [Getting Started](#getting-started)
    - [Installing](#installing)
      - [Install with pip](#install-with-pip)
      - [Install with optional dependencies](#install-with-optional-dependencies)
      - [Install from source](#install-from-source)
  - [Repository Organization](#repository-organization)
    - [`recipes/`](#recipes)
    - [`src/`](#src)
  - [Supported Features](#supported-features)
  - [Resource](#resource)

## Configuration Environment

```bash
$ python -m venv env
$ source env/bin/activate
$ pip install --upgrade pip -i https://mirrors.aliyun.com/pypi/simple/
$ pip install -e . -i https://mirrors.aliyun.com/pypi/simple/
```

## Getting Started

### Installing

#### Install with pip
```
pip install llama-recipes
```

#### Install with optional dependencies
Llama-recipes offers the installation of optional packages.
To run the unit tests we can install the required dependencies with:
```
pip install llama-recipes[tests]
```
For the vLLM example we need additional requirements that can be installed with:
```
pip install llama-recipes[vllm]
```
To use the sensitive topics safety checker install with:
```
pip install llama-recipes[auditnlg]
```
Some recipes require the presence of langchain. To install the packages follow the recipe description or install with:
```
pip install llama-recipes[langchain]
```
Optional dependencies can also be combined with [option1,option2].

#### Install from source
To install from source e.g. for development use these commands. We're using hatchling as our build backend which requires an up-to-date pip as well as setuptools package.
```
git clone git@github.com:meta-llama/llama-recipes.git
cd llama-recipes
pip install -U pip setuptools
pip install -e .
```
For development and contributing to llama-recipes please install all optional dependencies:
```
git clone git@github.com:meta-llama/llama-recipes.git
cd llama-recipes
pip install -U pip setuptools
pip install -e .[tests,auditnlg,vllm]
```

## Repository Organization
Most of the code dealing with Llama usage is organized across 2 main folders: `recipes/` and `src/`.

### `recipes/`

Contains examples organized in folders by topic:
| Subfolder | Description |
|---|---|
[quickstart](./recipes/quickstart) | The "Hello World" of using Llama, start here if you are new to using Llama.
[use_cases](./recipes/use_cases)|Scripts showing common applications of Meta Llama3
[3p_integrations](./recipes/3p_integrations)|Partner owned folder showing common applications of Meta Llama3
[responsible_ai](./recipes/responsible_ai)|Scripts to use PurpleLlama for safeguarding model outputs
[experimental](./recipes/experimental)|Meta Llama implementations of experimental LLM techniques

### `src/`

Contains modules which support the example recipes:
| Subfolder | Description |
|---|---|
| [configs](src/llama_recipes/configs/) | Contains the configuration files for PEFT methods, FSDP, Datasets, Weights & Biases experiment tracking. |
| [datasets](src/llama_recipes/datasets/) | Contains individual scripts for each dataset to download and process. Note |
| [inference](src/llama_recipes/inference/) | Includes modules for inference for the fine-tuned models. |
| [model_checkpointing](src/llama_recipes/model_checkpointing/) | Contains FSDP checkpoint handlers. |
| [policies](src/llama_recipes/policies/) | Contains FSDP scripts to provide different policies, such as mixed precision, transformer wrapping policy and activation checkpointing along with any precision optimizer (used for running FSDP with pure bf16 mode). |
| [utils](src/llama_recipes/utils/) | Utility files for:<br/> - `train_utils.py` provides training/eval loop and more train utils.<br/> - `dataset_utils.py` to get preprocessed datasets.<br/> - `config_utils.py` to override the configs received from CLI.<br/> - `fsdp_utils.py` provides FSDP  wrapping policy for PEFT methods.<br/> - `memory_utils.py` context manager to track different memory stats in train loop. |


## Supported Features
The recipes and modules in this repository support the following features:

| Feature                                        |   |
| ---------------------------------------------- | - |
| HF support for inference                       | ✅ |
| HF support for finetuning                      | ✅ |
| PEFT                                           | ✅ |
| Deferred initialization ( meta init)           | ✅ |
| Low CPU mode for multi GPU                     | ✅ |
| Mixed precision                                | ✅ |
| Single node quantization                       | ✅ |
| Flash attention                                | ✅ |
| Activation checkpointing FSDP                  | ✅ |
| Hybrid Sharded Data Parallel (HSDP)            | ✅ |
| Dataset packing & padding                      | ✅ |
| BF16 Optimizer (Pure BF16)                     | ✅ |
| Profiling & MFU tracking                       | ✅ |
| Gradient accumulation                          | ✅ |
| CPU offloading                                 | ✅ |
| FSDP checkpoint conversion to HF for inference | ✅ |
| W&B experiment tracker                         | ✅ |

## Resource

1. [an interactive tokenizer tool](https://tiktokenizer.vercel.app/?model=meta-llama%2FMeta-Llama-3-8B)