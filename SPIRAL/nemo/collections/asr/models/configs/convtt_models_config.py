# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
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
#
from dataclasses import dataclass
from typing import List, Optional

from omegaconf import MISSING

from nemo.collections.asr.modules.audio_preprocessing import (
    AudioToMelSpectrogramPreprocessorConfig,
    SpectrogramAugmentationConfig,
)
from nemo.core.config import modelPT as model_cfg

from .common_config import *
from .transducer_config import *
from .conv_transformer_config import *


@dataclass
class ConvTTModel(model_cfg.ModelConfig):
    labels: List[str] = MISSING
    tokenizer: Optional[Tokenizer] = None

    train_ds: DatasetConfig = MISSING
    validation_ds: DatasetConfig = MISSING
    test_ds: DatasetConfig = MISSING

    expected_gpu_num: int = MISSING
    optim: Optional[OptimConfig] = MISSING

    preprocessor: AudioToMelSpectrogramPreprocessorConfig = MISSING
    spec_augment: Optional[SpectrogramAugmentationConfig] = MISSING

    encoder: ConvTransformerEncoder = MISSING
    decoder: TransformerTDecoder = MISSING
    joint: RNNTJoint = MISSING
    model_defaults: ModelDefaults = MISSING

    decoding: Decoding = MISSING


@dataclass
class ConvTTConfig(model_cfg.ModelPTConfig):
    model: ConvTTModel = ConvTTModel()
