import argparse
import wandb
import os
from time import gmtime, strftime
from src.modelmodule.seg_prectime import SegModel_prectime
from src.datamodule.seg import SegDataModule
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    RichModelSummary,
    RichProgressBar,
)
from pytorch_lightning import Trainer, seed_everything
from omegaconf import DictConfig
import omegaconf
import torch
import hydra
import logging
from pathlib import Path
import yaml
import sys
sys.path.append('./')

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s:%(name)s - %(message)s"
)
LOGGER = logging.getLogger(Path(__file__).name)


@hydra.main(config_path="conf", config_name="train_prectime", version_base="1.2")
def main(cfg: DictConfig):  # type: ignore
    seed_everything(cfg.seed)
    cfg.sequence_length = cfg.duration + 2 * cfg.overlap_interval
    # using simple format of showing time
    s = strftime("%a_%d_%b_%H_%M", gmtime())

    wandb.init(project="kaggle-sleep-sweep")
    # init experiment logger
    pl_logger = WandbLogger(
        project="kaggle-sleep-sweep",
    )
    pl_logger.log_hyperparams(cfg)

    datamodule = SegDataModule(cfg)
    LOGGER.info("Set Up DataModule")
    model = SegModel_prectime(
        cfg, datamodule.valid_event_df, len(
            cfg.features), len(cfg.labels), cfg.duration
    )
    print(model)
    # x = torch.rand((cfg.batch_size,len(cfg.features),cfg.sequence_length))
    # print(model(x))
    # set callbacks
    checkpoint_cb = ModelCheckpoint(
        verbose=True,
        monitor=cfg.monitor,
        mode=cfg.monitor_mode,
        save_top_k=1,
        save_last=False,
    )
    lr_monitor = LearningRateMonitor("epoch")
    progress_bar = RichProgressBar()
    model_summary = RichModelSummary(max_depth=2)

    early_stop_callback = EarlyStopping(
        monitor="val_score2", patience=100, verbose=False, mode="max")

    trainer = Trainer(
        # env
        default_root_dir=Path.cwd(),
        # num_nodes=cfg.training.num_gpus,
        accelerator=cfg.accelerator,
        precision=16 if cfg.use_amp else 32,
        # training
        fast_dev_run=cfg.debug,  # run only 1 train batch and 1 val batch
        max_epochs=cfg.epoch,
        max_steps=cfg.epoch * len(datamodule.train_dataloader()),
        gradient_clip_val=cfg.gradient_clip_val,
        accumulate_grad_batches=cfg.accumulate_grad_batches,
        callbacks=[checkpoint_cb, lr_monitor, progress_bar,
                   model_summary, early_stop_callback],
        logger=pl_logger,
        # resume_from_checkpoint=resume_from,
        num_sanity_val_steps=0,
        log_every_n_steps=int(len(datamodule.train_dataloader())*0.1),
        sync_batchnorm=True,
        # val_check_interval=0.5,
        check_val_every_n_epoch=3,
        # strategy='ddp_find_unused_parameters_true',
    )

    trainer.fit(model, datamodule=datamodule)

    print("best model path: ", checkpoint_cb.best_model_path)

    # load best weights
    model = SegModel_prectime.load_from_checkpoint(
        checkpoint_cb.best_model_path,
        cfg=cfg,
        val_event_df=datamodule.valid_event_df,
        feature_dim=len(cfg.features),
        num_classes=len(cfg.labels),
        duration=cfg.duration,
    )

    save_path = os.path.join(cfg.dir.output_dir, "train", cfg.exp_name, "single", "/".join(
        checkpoint_cb.best_model_path.split('/')[:-2]), "best_model_weights.pth")
    cfg_save_path = os.path.join(cfg.dir.output_dir, "train", cfg.exp_name, "single",
                                 "/".join(checkpoint_cb.best_model_path.split('/')[:-2]), "best_cfg.pkl")
    cfg_yaml_save_path = os.path.join(cfg.dir.output_dir, "train", cfg.exp_name, "single", "/".join(
        checkpoint_cb.best_model_path.split('/')[:-2]), "best_cfg.yaml")
    # weights_path = str("model_weights.pth")  # type: ignore
    LOGGER.info(f"Extracting and saving best weights: {save_path}")

    torch.save(model.model.state_dict(), save_path)
    torch.save(cfg, cfg_save_path)
    with open(cfg_yaml_save_path, 'w') as f:
        yaml.dump(cfg, f)

    wandb.finish()
    return


# kill wandb: ps aux|grep wandb|grep -v grep | awk '{print $2}'|xargs kill -9
if __name__ == "__main__":
    main()
