from typing import Union

import torch.nn as nn
from omegaconf import DictConfig

from src.models.decoder.lstmdecoder import LSTMDecoder
from src.models.decoder.mlpdecoder import MLPDecoder
from src.models.decoder.transformerdecoder import TransformerDecoder
from src.models.decoder.unet1ddecoder import UNet1DDecoder
from src.models.decoder.PredictionRefinement import PredictionRefinement
from src.models.feature_extractor.cnn import CNNSpectrogram
from src.models.feature_extractor.lstm import LSTMFeatureExtractor
from src.models.feature_extractor.panns import PANNsFeatureExtractor
from src.models.feature_extractor.spectrogram import SpecFeatureExtractor
from src.models.feature_extractor.PrecFeatureExtractor import PrecFeatureExtractor
from src.models.encoder.ContextDetection import ContextDetection
from src.models.spec1D import Spec1D
from src.models.spec2Dcnn import Spec2DCNN

import segmentation_models_pytorch as smp

FEATURE_EXTRACTORS = Union[
    CNNSpectrogram, PANNsFeatureExtractor, LSTMFeatureExtractor, SpecFeatureExtractor
]
DECODERS = Union[UNet1DDecoder, LSTMDecoder, TransformerDecoder, MLPDecoder]
MODELS = Union[Spec1D, Spec2DCNN]




def get_feature_extractor(
    cfg: DictConfig, feature_dim: int, num_timesteps: int
) -> FEATURE_EXTRACTORS:
    feature_extractor: FEATURE_EXTRACTORS
    if cfg.feature_extractor.name == "CNNSpectrogram":
        feature_extractor = CNNSpectrogram(
            in_channels=feature_dim,
            base_filters=cfg.feature_extractor.base_filters,
            kernel_sizes=cfg.feature_extractor.kernel_sizes,
            stride=cfg.feature_extractor.stride,
            sigmoid=cfg.feature_extractor.sigmoid,
            output_size=num_timesteps,
            conv=nn.Conv1d,
            reinit=cfg.feature_extractor.reinit,
        )
    elif cfg.feature_extractor.name == "PANNsFeatureExtractor":
        feature_extractor = PANNsFeatureExtractor(
            in_channels=feature_dim,
            base_filters=cfg.feature_extractor.base_filters,
            kernel_sizes=cfg.feature_extractor.kernel_sizes,
            stride=cfg.feature_extractor.stride,
            sigmoid=cfg.feature_extractor.sigmoid,
            output_size=num_timesteps,
            conv=nn.Conv1d,
            reinit=cfg.feature_extractor.reinit,
            win_length=cfg.feature_extractor.win_length,
        )
    elif cfg.feature_extractor.name == "LSTMFeatureExtractor":
        feature_extractor = LSTMFeatureExtractor(
            in_channels=feature_dim,
            hidden_size=cfg.feature_extractor.hidden_size,
            num_layers=cfg.feature_extractor.num_layers,
            bidirectional=cfg.feature_extractor.bidirectional,
            out_size=num_timesteps,
        )
    elif cfg.feature_extractor.name == "SpecFeatureExtractor":
        feature_extractor = SpecFeatureExtractor(
            in_channels=feature_dim,
            height=cfg.feature_extractor.height,
            hop_length=cfg.feature_extractor.hop_length,
            win_length=cfg.feature_extractor.win_length,
            out_size=num_timesteps,
        )
    elif cfg.feature_extractor.name == "PrecFeatureExtractor":
        feature_extractor = PrecFeatureExtractor(
            in_channels=feature_dim,
            hidden_channels=cfg.feature_extractor.hidden_channels,
            kernel_size=cfg.feature_extractor.kernel_size,
            padding=cfg.feature_extractor.padding,
            stride=cfg.feature_extractor.stride,
            dilation=cfg.feature_extractor.dilation
        )
    else:
        raise ValueError(f"Invalid feature extractor name: {cfg.feature_extractor.name}")

    return feature_extractor

def get_encoder(cfg: DictConfig, feature_extractor: FEATURE_EXTRACTORS):
    if cfg.encoder.name == 'UNet':
        encoder = smp.Unet(
            encoder_name=cfg.model.encoder_name,
            encoder_weights=cfg.model.encoder_weights,
            in_channels=feature_extractor.out_chans,
            classes=1,
        )
    elif cfg.encoder.name == 'UNet++':
        encoder = smp.UnetPlusPlus(
            encoder_name=cfg.model.encoder_name,
            encoder_weights=cfg.model.encoder_weights,
            in_channels=feature_extractor.out_chans,
            classes=1,
        )
    elif cfg.encoder.name == 'ContextDetection':
        encoder = ContextDetection(
            input_size= cfg.encoder.input_size,
            hidden_size= cfg.encoder.hidden_size,
            num_layers= cfg.encoder.num_layers,
            bidirectional= cfg.encoder.bidirectional,
        )
    else:
        raise ValueError(f"Invalid encoder name: {cfg.encoder.name}")
    
    return encoder

def get_decoder(cfg: DictConfig, n_channels: int, n_classes: int, num_timesteps: int) -> DECODERS:
    decoder: DECODERS
    if cfg.decoder.name == "UNet1DDecoder":
        decoder = UNet1DDecoder(
            n_channels=n_channels,
            n_classes=n_classes,
            duration=num_timesteps,
            bilinear=cfg.decoder.bilinear,
            se=cfg.decoder.se,
            res=cfg.decoder.res,
            scale_factor=cfg.decoder.scale_factor,
            dropout=cfg.decoder.dropout,
        )
    elif cfg.decoder.name == "LSTMDecoder":
        decoder = LSTMDecoder(
            input_size=n_channels,
            hidden_size=cfg.decoder.hidden_size,
            num_layers=cfg.decoder.num_layers,
            dropout=cfg.decoder.dropout,
            bidirectional=cfg.decoder.bidirectional,
            n_classes=n_classes,
        )
    elif cfg.decoder.name == "TransformerDecoder":
        decoder = TransformerDecoder(
            input_size=n_channels,
            hidden_size=cfg.decoder.hidden_size,
            num_layers=cfg.decoder.num_layers,
            dropout=cfg.decoder.dropout,
            nhead=cfg.decoder.nhead,
            n_classes=n_classes,
        )
    elif cfg.decoder.name == "MLPDecoder":
        decoder = MLPDecoder(n_channels=n_channels, n_classes=n_classes)
    elif cfg.decoder.name == 'PredictionRefinement':
        decoder = PredictionRefinement(
            in_channels= cfg.decoder.in_channels,
            out_channels= cfg.decoder.out_channels,
            kernel_size= cfg.decoder.kernel_size,
            padding= cfg.decoder.padding,
            stride= cfg.decoder.stride,
            dilation= cfg.decoder.dilation,
            if_maxpool= cfg.decoder.if_maxpool,
            if_dropout= cfg.decoder.if_dropout,
            mode= cfg.decoder.mode
        )

    else:
        raise ValueError(f"Invalid decoder name: {cfg.decoder.name}")

    return decoder

def get_model(cfg: DictConfig, feature_dim: int, n_classes: int, num_timesteps: int) -> MODELS:
    model: MODELS
    if cfg.model.name == "Spec2DCNN":
        feature_extractor = get_feature_extractor(cfg, feature_dim, num_timesteps)
        encoder = get_encoder(cfg, feature_extractor)
        decoder = get_decoder(cfg, feature_extractor.height, n_classes, num_timesteps)
        model = Spec2DCNN(
            feature_extractor=feature_extractor,
            decoder=decoder,
            encoder=encoder,
            cfg = cfg,
            mixup_alpha=cfg.augmentation.mixup_alpha,
            cutmix_alpha=cfg.augmentation.cutmix_alpha
        )
    elif cfg.model.name == "Spec1D":
        feature_extractor = get_feature_extractor(cfg, feature_dim, num_timesteps)
        decoder = get_decoder(cfg, feature_extractor.height, n_classes, num_timesteps)
        model = Spec1D(
            feature_extractor=feature_extractor,
            decoder=decoder,
            cfg = cfg,
            mixup_alpha=cfg.augmentation.mixup_alpha,
            cutmix_alpha=cfg.augmentation.cutmix_alpha,
        )
    elif cfg.model.name == "PrecTime":
        pass
    else:
        raise NotImplementedError

    return model
