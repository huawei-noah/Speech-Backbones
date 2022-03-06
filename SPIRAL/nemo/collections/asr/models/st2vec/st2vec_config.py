# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional, List, Any

from dataclasses import field, dataclass
from omegaconf import MISSING

from nemo.collections.asr.models.spec2vec.spec2vec_config import FeatureEncoderConfig, ProjectorConfig, \
    NoisePerturbConfig
from nemo.collections.asr.models.wav2vec.wav2vec_config import LossConfig, Wav2VecTransformerConfig, \
    Wav2VecMaskingConfig, QuantizerConfig
from nemo.collections.asr.modules.audio_preprocessing import AudioToMelSpectrogramPreprocessorConfig
from nemo.core.config.modelPT import ModelConfig


@dataclass
class ShiftPerturbConfig:
    dist: str = 'uniform'
    shift_prob: float = MISSING
    max_ratio: float = 0.5
    unit: int = MISSING
    max: Optional[int] = None
    min: Optional[int] = None
    mean: Optional[float] = None
    std: Optional[float] = None
    truncate: bool = True


@dataclass
class ST2VecEncoderConfig:
    preprocessor: AudioToMelSpectrogramPreprocessorConfig = MISSING

    feature_encoder: FeatureEncoderConfig = FeatureEncoderConfig()
    freeze_feature_encoder: bool = False
    noise_mix_ratio: Optional[float] = None
    masking: Optional[Wav2VecMaskingConfig] = None
    shifting: Optional[ShiftPerturbConfig] = None
    target_shifting: Optional[ShiftPerturbConfig] = None
    target_masking: Optional[Wav2VecMaskingConfig] = None
    target_compute_perturb: bool = False

    target_momentum: float = 0.99
    target_momentum_final: Optional[float] = None
    target_momentum_steps: Optional[int] = None
    target_momentum_type: Optional[str] = None
    projector: Optional[ProjectorConfig] = None
    predictor: Optional[ProjectorConfig] = None

    quantizer: Optional[QuantizerConfig] = None

    n_negatives: int = field(
        default=100, metadata={'help': 'Number of negatives to sample from the same audio sample'}
    )
    cross_sample_negatives: int = field(
        default=0, metadata={'help': 'Number of negatives to sample from any sample in the batch'}
    )
    codebook_negatives: int = field(default=0, metadata={'help': 'Number of negative examples in codebook'})
    negatives_from_everywhere: bool = field(
        default=False, metadata={'help': 'Sample negatives from everywhere, not just masked states'}
    )
    negatives_from_noisy_features: bool = False


@dataclass
class FeatST2VecEncoderConfig:
    preprocessor: AudioToMelSpectrogramPreprocessorConfig = MISSING

    feature_encoder: FeatureEncoderConfig = FeatureEncoderConfig()
    context_net: Wav2VecTransformerConfig = MISSING
    masking: Optional[Wav2VecMaskingConfig] = None
    target_masking: Optional[Wav2VecMaskingConfig] = None

    target_momentum: float = 0.99
    predictor: Optional[ProjectorConfig] = None

    n_negatives: int = field(
        default=100, metadata={'help': 'Number of negatives to sample from the same audio sample'}
    )
    cross_sample_negatives: int = field(
        default=0, metadata={'help': 'Number of negatives to sample from any sample in the batch'}
    )
    codebook_negatives: int = field(default=0, metadata={'help': 'Number of negative examples in codebook'})
    negatives_from_everywhere: bool = field(
        default=False, metadata={'help': 'Sample negatives from everywhere, not just masked states'}
    )
    negatives_from_noisy_features: bool = False


@dataclass
class ST2VecPretrainModelConfig(ModelConfig):
    encoder_type: str = 'st'
    st2vec_encoder: Any = MISSING

    noise_perturb: Optional[NoisePerturbConfig] = None

    loss_type: str = 'wav2vec'
    logit_temp: float = field(default=0.1, metadata={'help': 'Temperature to divide logits by'})
    loss: LossConfig = LossConfig()

    expected_gpu_num: int = 1
