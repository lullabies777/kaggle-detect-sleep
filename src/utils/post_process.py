import numpy as np
import polars as pl
from scipy.signal import find_peaks


def post_process_for_seg(
    keys: list[str], preds: np.ndarray, score_th: float = 0.01, distance: int = 5000
) -> pl.DataFrame:
    """make submission dataframe for segmentation task

    Args:
        keys (list[str]): list of keys. key is "{series_id}_{chunk_id}"
        preds (np.ndarray): (num_series * num_chunks, duration, 2)
        score_th (float, optional): threshold for score. Defaults to 0.5.

    Returns:
        pl.DataFrame: submission dataframe
    """
    series_ids = np.array(list(map(lambda x: x.split("_")[0], keys)))
    unique_series_ids = np.unique(series_ids)

    records = []
    for series_id in unique_series_ids:
        series_idx = np.where(series_ids == series_id)[0]
        this_series_preds = preds[series_idx].reshape(-1, 2)

        for i, event_name in enumerate(["onset", "wakeup"]):
            this_event_preds = this_series_preds[:, i]
            steps = find_peaks(this_event_preds, height=score_th, distance=distance)[0]
            scores = this_event_preds[steps]

            for step, score in zip(steps, scores):
                records.append(
                    {
                        "series_id": series_id,
                        "step": step,
                        "event": event_name,
                        "score": score,
                    }
                )

    if len(records) == 0:  # 一つも予測がない場合はdummyを入れる
        records.append(
            {
                "series_id": series_id,
                "step": 0,
                "event": "onset",
                "score": 0,
            }
        )

    sub_df = pl.DataFrame(records).sort(by=["series_id", "step"])
    row_ids = pl.Series(name="row_id", values=np.arange(len(sub_df)))
    sub_df = sub_df.with_columns(row_ids).select(["row_id", "series_id", "step", "event", "score"])
    return sub_df

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

def post_process_for_seg_v2(
    keys: list[str], preds: np.ndarray,  gap: int = 180, quantile: float = 0.9
)-> pl.DataFrame:
    """make submission dataframe for segmentation task

    Args:
        keys (list[str]): list of keys. key is "{series_id}_{chunk_id}"
        preds (np.ndarray): (num_series * num_chunks, duration, 2)

    Returns:
        pl.DataFrame: submission dataframe
    """
    series_ids = np.array(list(map(lambda x: x.split("_")[0], keys)))
    unique_series_ids = np.unique(series_ids)
    
    records = []
    for series_id in unique_series_ids:
        series_idx = np.where(series_ids == series_id)[0]
        this_series_preds = preds[series_idx].reshape(-1, 2)

        for i, event_name in enumerate(["onset", "wakeup"]):
            tmp_df = []
            this_event_preds = this_series_preds[:, i]
            steps = get_results_slide_window(this_event_preds, gap)
            scores = this_event_preds[steps]

            for step, score in zip(steps, scores):
                tmp_df.append(
                    {
                        "series_id": series_id,
                        "step": step,
                        "event": event_name,
                        "score": score,
                    }
                )
            tmp_df = pl.DataFrame(tmp_df)
            tmp_df = tmp_df.filter(pl.col("score") > pl.col("score").quantile(quantile))
            records.append(tmp_df)
            
    if len(records) == 0:
        records.append(
            pl.DataFrame({
                "series_id": series_id,
                "step": 0,
                "event": "onset",
                "score": 0,
            })
        )
    sub_df = pl.concat(records)
    row_ids = pl.Series(name="row_id", values=np.arange(len(sub_df)))
    sub_df = sub_df.with_columns(row_ids).select(["row_id", "series_id", "step", "event", "score"])
    return sub_df
