from typing import Union

import torch.nn as nn
from omegaconf import DictConfig

from src.models.decoder.lstmdecoder import LSTMDecoder
from src.models.decoder.mlpdecoder import MLPDecoder
from src.models.decoder.transformerdecoder import TransformerDecoder
from src.models.decoder.unet1ddecoder import UNet1DDecoder
from src.models.decoder.PredictionRefinement import PredictionRefinement
from src.models.decoder.cnn_transformer import cnn_transformer
from src.models.feature_extractor.cnn import CNNSpectrogram
from src.models.feature_extractor.lstm import LSTMFeatureExtractor
from src.models.feature_extractor.panns import PANNsFeatureExtractor
from src.models.feature_extractor.spectrogram import SpecFeatureExtractor
from src.models.feature_extractor.PrecFeatureExtractor import PrecFeatureExtractor
from src.models.encoder.ContextDetection import ContextDetection
from src.models.encoder.Residual_lstm import Residual_lstm
from src.models.spec1D import Spec1D
from src.models.spec2Dcnn import Spec2DCNN
from src.models.PrecTime import PrecTime
from src.models.Combined import Combined

import segmentation_models_pytorch as smp

FEATURE_EXTRACTORS = Union[
    CNNSpectrogram, PANNsFeatureExtractor, LSTMFeatureExtractor, SpecFeatureExtractor, PrecFeatureExtractor]
DECODERS = Union[UNet1DDecoder, LSTMDecoder, TransformerDecoder, MLPDecoder, PredictionRefinement,cnn_transformer]
MODELS = Union[Spec1D, Spec2DCNN, PrecTime, Combined]




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
            input_channels=feature_dim,
            left_hidden_channels=cfg.feature_extractor.left_hidden_channels,
            right_hidden_channels=cfg.feature_extractor.right_hidden_channels,
            left_fe_kernel_size=cfg.feature_extractor.left_fe_kernel_size,
            right_fe_kernel_size=cfg.feature_extractor.right_fe_kernel_size,
            left_fe_padding=cfg.feature_extractor.left_fe_padding,
            right_fe_padding=cfg.feature_extractor.right_fe_padding,
            left_fe_stride=cfg.feature_extractor.left_fe_stride,
            right_fe_stride=cfg.feature_extractor.right_fe_stride,
            left_fe_dilation=cfg.feature_extractor.left_fe_dilation,
            right_fe_dilation=cfg.feature_extractor.right_fe_dilation
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
            fe_fc_dimension= cfg.encoder.fe_fc_dimension,
            lstm_dimensions= cfg.encoder.lstm_dimensions,
            num_layers= cfg.encoder.num_layers,
            sequence_length= cfg.sequence_length,
            chunks= cfg.chunks,
            n_head= cfg.encoder.n_head,
            dropout= cfg.encoder.dropout,
            num_encoder_layers= cfg.encoder.num_encoder_layers,
            dim_feedforward= cfg.encoder.dim_feedforward,
            r_lstm_layers= cfg.encoder.r_lstm_layers,
            encoder_type= cfg.encoder.encoder_type,
            activation= cfg.encoder.activation
        )
    elif cfg.encoder.name == 'Residual_lstm':
        encoder = Residual_lstm(
            input_dimension= cfg.encoder.input_dimension,
            num_layers= cfg.encoder.num_layers,
            dropout= cfg.encoder.dropout,
            names= cfg.encoder.names
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
        if cfg.encoder.encoder_type=='lstm':
            cfg.decoder.in_channels = cfg.feature_extractor.left_hidden_channels[-1] + cfg.feature_extractor.right_hidden_channels[-1] + cfg.encoder.lstm_dimensions[-1] * 2
            
        elif cfg.encoder.encoder_type=='transformer':
            cfg.decoder.in_channels = cfg.feature_extractor.left_hidden_channels[-1] + cfg.feature_extractor.right_hidden_channels[-1] + cfg.encoder.fe_fc_dimension
        elif cfg.encoder.encoder_type=='r_lstm':
            cfg.decoder.in_channels = cfg.feature_extractor.left_hidden_channels[-1] + cfg.feature_extractor.right_hidden_channels[-1] + cfg.encoder.fe_fc_dimension
        decoder = PredictionRefinement(
            in_channels= cfg.decoder.in_channels,
            out_channels= cfg.decoder.out_channels,
            kernel_size= cfg.decoder.kernel_size,
            padding= cfg.decoder.padding,
            stride= cfg.decoder.stride,
            dilation= cfg.decoder.dilation,
            scale_factor= cfg.decoder.scale_factor,
            mode= cfg.decoder.mode
        )
    elif cfg.decoder.name == 'cnn_transformer':
        decoder = cnn_transformer(
            input_dimension= cfg.encoder.input_dimension,
            num_layers= cfg.decoder.num_layers,
            nheads= cfg.decoder.nheads,
            dropout= cfg.decoder.dropout,
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
        feature_extractor = get_feature_extractor(cfg, feature_dim, num_timesteps)
        encoder = get_encoder(cfg, feature_extractor)
        decoder = get_decoder(cfg, None, n_classes, num_timesteps)
        model = PrecTime(
            feature_extractor=feature_extractor,
            encoder=encoder,
            decoder=decoder,
            cfg = cfg,
            mixup_alpha=cfg.augmentation.mixup_alpha,
            cutmix_alpha=cfg.augmentation.cutmix_alpha,
        )
    elif cfg.model.name == "Combined":
        feature_extractor = get_feature_extractor(cfg, feature_dim, num_timesteps)
        encoder = get_encoder(cfg, feature_extractor)
        decoder = get_decoder(cfg, None, n_classes, num_timesteps)
        model = Combined(
            feature_extractor=feature_extractor,
            encoder=encoder,
            decoder=decoder,
            cfg = cfg,
            mixup_alpha=cfg.augmentation.mixup_alpha,
            cutmix_alpha=cfg.augmentation.cutmix_alpha,
        )

    else:
        raise NotImplementedError
    return model
