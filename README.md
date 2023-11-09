# TODO LIST
- [x] ~~Overlap Inference~~
- [ ] Prectime Structure
- [x] ~~More post-preprocess~~
- [ ] Ensemble
- [x] ~~More Loss Function~~ Now support `bce`, `mse`, `focal`, `kl` and weight for pos labels (controlled by loss.weight, 0 by default)
- [x] ~~More encoders: U-Net++~~
- [ ] More features
- [ ] Feature embedding
- [x] ~~Wandb sweep~~
- [x] scores on val

# Use Wandb sweep
- If needed, modify `sweep.yaml` first.

- Initiate WandB sweep as: `$ wandb sweep wandb_sweep.yaml`

- Run Agent
  Creating a sweep returns a wandb agent command like:
  ![Screenshot showing result returned by wandb sweep command](https://user-images.githubusercontent.com/13994201/153241187-dfa308b6-c52e-4f0a-9f4d-f47b356b1088.png)
- Next invoke the `wandb agent path/to/sweep` command provided in the output of the previous command.
# Child Mind Institute - Detect Sleep States

This repository is for [Child Mind Institute - Detect Sleep States](https://www.kaggle.com/competitions/child-mind-institute-detect-sleep-states/overview)

## Build Environment
### 1. install [rye](https://github.com/mitsuhiko/rye)

[install documentation](https://rye-up.com/guide/installation/#installing-rye)

MacOS
```zsh
curl -sSf https://rye-up.com/get | bash
echo 'source "$HOME/.rye/env"' >> ~/.zshrc
source ~/.zshrc
```

Linux
```bash
curl -sSf https://rye-up.com/get | bash
echo 'source "$HOME/.rye/env"' >> ~/.bashrc
source ~/.bashrc
```

Windows  
see [install documentation](https://rye-up.com/guide/installation/)

### 2. Create virtual environment

```bash
rye sync
```

### 3. Activate virtual environment

```bash
. .venv/bin/activate
```

### Set path
Rewrite run/conf/dir/local.yaml to match your environment

```yaml
data_dir: 
processed_dir: 
output_dir: 
model_dir: 
sub_dir: ./
```

## Prepare Data

### 1. Download data

```bash
cd data
kaggle competitions download -c child-mind-institute-detect-sleep-states
unzip child-mind-institute-detect-sleep-states.zip
```

### 2. Preprocess data

```bash
rye run python -m run/prepare_data.py phase=train,test
```

## Train Model
The following commands are for training the model of LB0.714
```bash
rye run python run/train.py downsample_rate=2 duration=5760 exp_name=exp001 batch_size=32
```

You can easily perform experiments by changing the parameters because [hydra](https://hydra.cc/docs/intro/) is used.
The following commands perform experiments with downsample_rate of 2, 4, 6, and 8.

```bash
rye run python -m run/train.py downsample_rate=2,4,6,8
```


## Upload Model
```bash
rye run python tools/upload_dataset.py
```

## Inference
The following commands are for inference of LB0.714 
```bash
rye run python run/inference.py dir=kaggle exp_name=exp001 weight.run_name=single downsample_rate=2 duration=5760 model.encoder_weights=null post_process.score_th=0.005 post_process.distance=40 phase=test
```
