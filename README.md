# DenseSteer
A simplified pipeline to extract steering vectors from contrastive datasets and evaluate them using `lm-evaluation-harness`.

## Project Structure
* `01_extract_vectors.py`: Extracts activation differences between desired and baseline responses.
* `02_apply_vectors.py`: Automates evaluation across multiple layers and steering strengths.


## Setup
1. Install dependencies:
```bash
cd lm-evaluation-harness
pip install -e .
pip install steering-vectors 
```


## Usage
### Step 1: Extract Steering Vectors

Modify DATA_FILE in 01_extract_vectors.py to point to your data source, then run:

```bash
python 01_extract_vectors.py
```
This will generate steering_vector.pt in the ./vectors_out directory.

### Step 2: Evaluate

Modify LAYERS and LAMBDAS in 02_apply_vectors.py to define your experimental grid, then run:

```bash
python 02_apply_vectors.py
```
where `steer_hf` is the model class in lm-evaluation-harness applying steering vectors.

## Acknowledgements
This repository is built upon [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) and [steering-vectors](https://github.com/steering-vectors/steering-vectors). We would like to thank all contributors for their support.

