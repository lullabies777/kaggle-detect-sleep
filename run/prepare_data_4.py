import sys
sys.path.append('./')
from src.utils.common import trace
import shutil
from pathlib import Path
import hydra
import numpy as np
import polars as pl
from omegaconf import DictConfig
from tqdm import tqdm

diff_start = 1
diff_end = 6
diff_step = 1

new_feature_names = []

for i in range(diff_start, diff_end, diff_step):
    if i != 0:
        name_anglez = f"anglez_diff_{i}"
        new_feature_names.append(name_anglez)
        name_enmo = f"enmo_diff_{i}"
        new_feature_names.append(name_enmo)

print(new_feature_names)
# 'anglez_lag_-24', 'enmo_lag_-24', 'anglez_lag_-12', 'enmo_lag_-12', 'anglez_lag_12', 'enmo_lag_12', 'anglez_lag_24', 'enmo_lag_24', 'anglez_min_12', 'enmo_min_12', 'anglez_max_12', 'enmo_max_12', 'anglez_std_12', 'enmo_std_12', 'anglez_mean_12', 'enmo_mean_12', 'anglez_min_24', 'enmo_min_24', 'anglez_max_24', 'enmo_max_24', 'anglez_std_24', 'enmo_std_24', 'anglez_mean_24', 'enmo_mean_24', 'anglez_min_36', 'enmo_min_36', 'anglez_max_36', 'enmo_max_36', 'anglez_std_36', 'enmo_std_36', 'anglez_mean_36', 'enmo_mean_36', 'anglez_min_48', 'enmo_min_48', 'anglez_max_48', 'enmo_max_48', 'anglez_std_48', 'enmo_std_48', 'anglez_mean_48', 'enmo_mean_48', 'anglez_min_60', 'enmo_min_60', 'anglez_max_60', 'enmo_max_60', 'anglez_std_60', 'enmo_std_60', 'anglez_mean_60', 'enmo_mean_60'

SERIES_SCHEMA = {
    "series_id": pl.Utf8,
    "step": pl.UInt32,
    "anglez": pl.Float32,
    "enmo": pl.Float32,
}


FEATURE_NAMES = [
    "anglez",
    "enmo",
    "step",
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",
    "minute_sin",
    "minute_cos",
    "anglez_sin",
    "anglez_cos",
]

FEATURE_NAMES.extend(new_feature_names)

ANGLEZ_MEAN = -8.810476
ANGLEZ_STD = 35.521877
ENMO_MEAN = 0.041315
ENMO_STD = 0.101829


def to_coord(x: pl.Expr, max_: int, name: str) -> list[pl.Expr]:
    rad = 2 * np.pi * (x % max_) / max_
    x_sin = rad.sin()
    x_cos = rad.cos()

    return [x_sin.alias(f"{name}_sin"), x_cos.alias(f"{name}_cos")]


def deg_to_rad(x: pl.Expr) -> pl.Expr:
    return np.pi / 180 * x


def add_feature(series_df: pl.DataFrame) -> pl.DataFrame:
    series_df = series_df.with_row_count("step").with_columns(
        *to_coord(pl.col("timestamp").dt.hour(), 24, "hour"),
        *to_coord(pl.col("timestamp").dt.month(), 12, "month"),
        *to_coord(pl.col("timestamp").dt.minute(), 60, "minute"),
        pl.col('step') / pl.count('step'),
        pl.col('anglez_rad').sin().alias('anglez_sin'),
        pl.col('anglez_rad').cos().alias('anglez_cos'),
    ).select("series_id", *FEATURE_NAMES)
    return series_df


def save_each_series(this_series_df: pl.DataFrame, columns: list[str], output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    for col_name in columns:
        x = this_series_df.get_column(col_name).to_numpy(zero_copy_only=True)
        np.save(output_dir / f"{col_name}.npy", x)


@hydra.main(config_path="conf", config_name="prepare_data", version_base="1.2")
def main(cfg: DictConfig):
    processed_dir: Path = Path(cfg.dir.processed_dir) / cfg.phase

    # # ディレクトリが存在する場合は削除
    # if processed_dir.exists():
    #     shutil.rmtree(processed_dir)
    #     print(f"Removed {cfg.phase} dir: {processed_dir}")

    with trace("Load series"):
        # scan parquet
        if cfg.phase in ["train", "test"]:
            series_lf = pl.scan_parquet(
                Path(cfg.dir.data_dir) / f"{cfg.phase}_series.parquet",
                low_memory=True,
            )
        elif cfg.phase == "dev":
            series_lf = pl.scan_parquet(
                Path(cfg.dir.processed_dir) / f"{cfg.phase}_series.parquet",
                low_memory=True,
            )
        else:
            raise ValueError(f"Invalid phase: {cfg.phase}")

        # preprocess
        # series_df = (
        #     series_lf.with_columns(
        #         pl.col("timestamp").str.to_datetime("%Y-%m-%dT%H:%M:%S%z"),
        #         deg_to_rad(pl.col("anglez")).alias("anglez_rad"),
        #         (pl.col("anglez") - ANGLEZ_MEAN) / ANGLEZ_STD,
        #         (pl.col("enmo") - ENMO_MEAN) / ENMO_STD,
        #     )
        #     .select(
        #         [
        #             pl.col("series_id"),
        #             pl.col("anglez"),
        #             pl.col("enmo"),
        #             pl.col("timestamp"),
        #             pl.col("anglez_rad")
        #         ]
        #     )
        #     .collect(streaming=True)
        #     .sort(by=["series_id", "timestamp"])
        # )
        # n_unique = series_df.get_column("series_id").n_unique()

        # 先做标准化处理
        grouped_stats = series_lf.group_by('series_id').agg([
            pl.col('anglez').mean().alias('anglez_mean'),
            pl.col('anglez').std().alias('anglez_std'),
            pl.col('enmo').mean().alias('enmo_mean'),
            pl.col('enmo').std().alias('enmo_std')
        ])

        series_lf = series_lf.join(grouped_stats, on='series_id', how="left")

        print(series_lf.columns)

        series_lf = series_lf.with_columns([
            ((pl.col('anglez') - pl.col('anglez_mean')) /
                pl.col('anglez_std')).alias('normalized_anglez'),
            ((pl.col('enmo') - pl.col('enmo_mean')) /
                pl.col('enmo_std')).alias('normalized_enmo')
        ])

        series_lf = series_lf.drop(
            ['anglez', 'enmo', 'anglez_mean', 'anglez_std', 'enmo_mean', 'enmo_std']
        )

        series_lf = series_lf.with_columns([
            pl.col('normalized_anglez').alias('anglez'),
            pl.col('normalized_enmo').alias('enmo')
        ])

        series_lf = series_lf.drop(
            ['normalized_anglez', 'normalized_enmo']
        ).sort(by=['series_id', 'timestamp'])

        print(series_lf.columns)

        # 更新之后 rolling feature只存在于同一个series_id的内容之间
        series_df = (
            series_lf.with_columns(
                pl.col("timestamp").str.to_datetime("%Y-%m-%dT%H:%M:%S%z"),
                deg_to_rad(pl.col("anglez")).alias("anglez_rad"),
            )
            .group_by("series_id")
            .agg([
                pl.col("anglez"),
                pl.col("enmo"),
                pl.col("timestamp"),
                pl.col("anglez_rad"),
                *[pl.col("anglez").diff(i).alias(f"anglez_diff_{i}") for i in range(diff_start, diff_end, diff_step)],
                *[pl.col("enmo").diff(i).alias(f"enmo_diff_{i}") for i in range(diff_start, diff_end, diff_step)],
                # *[pl.col("anglez").shift(i).alias(f"anglez_lag_{i}")
                #   for i in range(shift_start, shift_end, shift_step) if i != 0],
                # *[pl.col("enmo").shift(i).alias(f"enmo_lag_{i}")
                #   for i in range(shift_start, shift_end, shift_step) if i != 0],
                # *[pl.col("anglez").rolling_mean(window_size).alias(
                #     f"anglez_mean_{window_size}") for window_size in window_steps],
                # *[pl.col("anglez").rolling_min(window_size).alias(
                #     f"anglez_min_{window_size}") for window_size in window_steps],
                # *[pl.col("anglez").rolling_max(window_size).alias(
                #     f"anglez_max_{window_size}") for window_size in window_steps],
                # *[pl.col("anglez").rolling_std(window_size).alias(
                #     f"anglez_std_{window_size}") for window_size in window_steps],
                # *[pl.col("enmo").rolling_mean(window_size).alias(
                #     f"enmo_mean_{window_size}") for window_size in window_steps],
                # *[pl.col("enmo").rolling_min(window_size).alias(
                #     f"enmo_min_{window_size}") for window_size in window_steps],
                # *[pl.col("enmo").rolling_max(window_size).alias(
                #     f"enmo_max_{window_size}") for window_size in window_steps],
                # *[pl.col("enmo").rolling_std(window_size).alias(
                #     f"enmo_std_{window_size}") for window_size in window_steps],
            ])
            .explode([
                "anglez", "enmo", "timestamp", "anglez_rad",
                *[f"anglez_diff_{i}" for i in range(diff_start, diff_end, diff_step)],
                *[f"enmo_diff_{i}" for i in range(diff_start, diff_end, diff_step)],
                # *[f"anglez_lag_{i}" for i in range(shift_start, shift_end, shift_step) if i != 0],
                # *[f"enmo_lag_{i}" for i in range(shift_start, shift_end, shift_step) if i != 0],
                # *[f"anglez_{stat}_{window_size}" for stat in
                #   ["mean", "min", "max", "std"] for window_size in window_steps],
                # *[f"enmo_{stat}_{window_size}" for stat in
                #   ["mean", "min", "max", "std"] for window_size in window_steps],
            ])
            .collect(streaming=True)
            .sort(by=["series_id", "timestamp"])
        )
        series_df = series_df.fill_null(0.0)
        print(series_df.head())
        n_unique = series_df.get_column("series_id").n_unique()

    with trace("Save features"):
        for series_id, this_series_df in tqdm(series_df.group_by("series_id"), total=n_unique):
            # 特徴量を追加
            this_series_df = add_feature(this_series_df)

            # 特徴量をそれぞれnpyで保存
            series_dir = processed_dir / series_id  # type: ignore
            save_each_series(this_series_df, FEATURE_NAMES, series_dir)


if __name__ == "__main__":
    main()
