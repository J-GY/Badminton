# TRIDENT

## Installation

To install the required dependencies, please use the following command to install from `requirement.yml`:

```bash
conda env create -f requirement.yml
```

## Quick Start

To run the complete pipeline, simply execute:

```bash
# badminton dataset
./run_badminton.sh

# tdrive dataset
./run_tdrive.sh

# rome dataset
./run_rome.sh

# chengdu dataset
./run_chengdu.sh

# xi'an dataset
./run_xian.sh
```

This script will execute the entire process including data preprocessing, model training, and evaluation.

## Requirements

- Python 3.10
- Conda package manager
- All dependencies listed in `requirement.yml`
