# 39th Detail Solution  [Child Mind Institute - Detect Sleep States](https://www.kaggle.com/competitions/child-mind-institute-detect-sleep-states)

**Thanks to Kaggle for hosting this meaningful competition. Thanks to all the Kagglers for your discussions and shared perspectives. **

**This was our first time participating in a formal tabular competition, and we've learned a lot from the experience.**

**Team Avengers will always continue the journey on Kaggle.**

Main GitHub Repo: [Here](https://github.com/lullabies777/kaggle-detect-sleep)

PrecTime GitHub Repo: [Here](https://github.com/Lizhecheng02/Kaggle-Detect_Sleep_States)

Original Discussion: [Here](https://www.kaggle.com/competitions/child-mind-institute-detect-sleep-states/discussion/459885)

## Here is the Detail Solution

### Baseline Code

Here, we need to thank [tubotubo](https://www.kaggle.com/tubotubo) for providing the baseline code.  We didn't join the competition from the very beginning, this baseline code provided us with some ideas and basic model structures.

### Dataset Preparation

- We didn't use any methods to handle the dirty data, which might be one reason why we couldn't improve our scores anymore.

- On the evening before the competition ended, my teammate found this [discussion](https://www.kaggle.com/competitions/child-mind-institute-detect-sleep-states/discussion/456177). Consequently, we attempted to clean the data by removing the data for the days where the event was empty. However, due to the time limitation, we didn't make significant progress.  

- We believe data cleaning should be helpful because the model using this method showed a smaller difference in scores on the private leaderboard.

### Feature Engineering

We generated new features using shift, different and rolling window functions. 

The final set we utilized 24 rolling features in addition to the original 4, making a total of 28 features. The use of new features did not significantly improve the model's score, which was somewhat unexpected for us.

Code: [Here](https://github.com/lullabies777/kaggle-detect-sleep/blob/main/run/prepare_data.py)

```
*[pl.col("anglez").diff(i).alias(f"anglez_diff_{i}") for i in range(diff_start, diff_end, diff_step)],
*[pl.col("anglez").shift(i).alias(f"anglez_lag_{i}")
  for i in range(shift_start, shift_end, shift_step) if i != 0],
*[pl.col("anglez").rolling_mean(window_size).alias(
    f"anglez_mean_{window_size}") for window_size in window_steps],
*[pl.col("anglez").rolling_min(window_size).alias(
    f"anglez_min_{window_size}") for window_size in window_steps],
*[pl.col("anglez").rolling_max(window_size).alias(
    f"anglez_max_{window_size}") for window_size in window_steps],
*[pl.col("anglez").rolling_std(window_size).alias(
    f"anglez_std_{window_size}") for window_size in window_steps]
```

### Wandb sweep

Wandb sweep is a hyperparameter optimization tool provided by the Wandb machine learning platform. It allows automatic exploration of different hyperparameter combinations to enhance a model's performance.

Implementation Code: [Here](https://github.com/lullabies777/kaggle-detect-sleep/tree/main/run/sweep)

### Models

- Used overlap - To enhance accuracy in predicting sequence edges, we utilized overlap by using a 10000 length sequence to predict an 8000 length sequence.
- Implementation of PrecTime Model -  You can find details in this [discussion](https://www.kaggle.com/competitions/child-mind-institute-detect-sleep-states/discussion/459616). We also made modifications to its structure, including the addition of transformer architecture and residual connection structures. The experiments had shown that these modifications contribute to a certain improvement in the model's performance.

### Post-preprocessing Trick

We used dynamic programming algorithm to deal with overlap problem.  

Principle behind this method: To achieve a high MAP (Mean Average Precision), three main criteria need to be met: Firstly, the predicted label should be sufficiently close to the actual label. Secondly, within a positive or negative tolerance range around the actual label, there should only be one predicted point. Thirdly, the score of other predicted points outside the actual label range should be lower than those within the range.

```
def get_results_slide_window(pred, gap):
    scores = list(pred)
    stack = [0]
    dp = [-1] * len(scores)
    dp[0] = 0
    for i in range(1,len(scores)):
        if i - stack[-1] < gap:
            if scores[i] >= scores[stack[-1]]:
                stack.pop()
                if i - gap >= 0:
                    if stack:
                        if dp[i - gap] != stack[-1]:
                            while stack and dp[i - gap] - stack[-1] < gap:
                                stack.pop()
                            stack.append(dp[i - gap])
                    else:
                        stack.append(dp[i - gap])
                stack.append(i)
        else:
            stack.append(i)
        dp[i] = stack[-1]
    return stack
```

### Ensemble

Our final ensemble method essentially involved averaging different outputs. With post-processing and this ensemble method combined, our results generally follow the pattern that the more models we use or the greater the variety of models, the higher the score.

Our submissions:

| Models                                                       | LB Score  | PB Score  | Selected |
| ------------------------------------------------------------ | --------- | --------- | -------- |
| ``2 * 5 folds PrecTime + 1 * 5 folds LSTM-Unet``             | ``0.75``  | ``0.8``   | **Yes**  |
| ``2 * 5 folds PrecTime + 2 * 5 folds LSTM-Unet + 10 single models`` | ``0.759`` | ``0.803`` | **Yes**  |
| ``1 * 5 folds PrecTime + 1 fold LSTM-Unet + 10 single models`` | ``0.761`` | ``0.804`` | **No**   |
| ``1 * 5 folds PrecTime + 1 * 5 folds LSTM-Unet + 10 single models`` | ``0.759`` | ``0.803`` | **No**   |

### Other Details

Welcome everyone to check our GitHub code, looking forward to any discussions.

### Conclusion

- Data Cleaning.
- Generate New Features. 
- Use Architecture Like Conv1d, RNN, GRU, LSTM or Transformer.
- Write Post-preprocessing for Special Metrics.

## Thanks to all of my teammates for working together to gain this Silver Medal.



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
- [ ] PrecTime
  - [x] Model
  - [ ] /src/models/PrecTime.py 里面Forward 算loss


# Use Wandb sweep
- If needed, modify `sweep.yaml` first.

- Initiate WandB sweep as: `$ wandb sweep wandb_sweep.yaml`

- Run Agent
  Creating a sweep returns a wandb agent command like:
  ![Screenshot showing result returned by wandb sweep command](https://user-images.githubusercontent.com/13994201/153241187-dfa308b6-c52e-4f0a-9f4d-f47b356b1088.png)
- Next invoke the `wandb agent path/to/sweep` command provided in the output of the previous command.
# Child Mind Institute - Detect Sleep States

This repository is for [Child Mind Institute - Detect Sleep States](https://www.kaggle.com/competitions/child-mind-institute-detect-sleep-states/overview)

## Build Environment (Not necessary)
### 1. Install [rye](https://github.com/mitsuhiko/rye)

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

## Set path
Rewrite run/conf/dir/local.yaml to match your environment

```yaml
data_dir: /kaggle-detect-sleep/data
processed_dir: /kaggle-detect-sleep/data/processed_data
output_dir: /kaggle-detect-sleep/output
model_dir: /kaggle-detect-sleep/output/train
sub_dir: ./
```

## Prepare Data

### 1. Set kaggle environment

```bash
export KAGGLE_USERNAME=your_kaggle_username
export KAGGLE_KEY=your_api_key
```

### 2. Download data

```bash
cd data
kaggle competitions download -c child-mind-institute-detect-sleep-states
unzip child-mind-institute-detect-sleep-states.zip
```

### 2. Preprocess data

```bash
python run/prepare_data.py phase=train,test
```

## Train Model
-  **Basic Model**
```bash
python run/train.py downsample_rate=2 duration=5760 exp_name=exp001 batch_size=32
```

You can easily perform experiments by changing the parameters because [hydra](https://hydra.cc/docs/intro/) is used.
The following commands perform experiments with downsample_rate of 2, 4, 6, and 8.

```bash
python run/train.py downsample_rate=2,4,6,8
```

-  **PrecTime Model**
```bash
python run/train_prectime.py
```
You can select sweep yaml
```bash
sweep_prectime_lstm.yaml
sweep_prectime_r_lstm.yaml
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

