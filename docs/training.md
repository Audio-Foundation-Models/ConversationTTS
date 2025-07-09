## Pre-training
The bash script for launching model training is at Step 8 in [egs/pretraining/run.sh](egs/pretraining/run.sh)

### Preparations
To begin with, make sure all your training data has went through [preprocessing](docs/dataset.md). Fill in the path of all resulting metadata at `train_data_jsons` at Line:135.

### Launch Training
1. Set number of nodes, GPUs per node and host IP for distributed training at Line:128~132.
2. run `bash egs/pretraining/run.sh`.

## Post-training
Developing...