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

from nemo.collections.asr.models.configs.common_config import AdamWParams, DatasetConfig, Tokenizer, \
    PolynomialHoldDecayAnnealingParams, Conv1dNormAct, ProjUpsampling
from nemo.collections.asr.models.spec2vec.spec2vec_config import ST2VecCTCFinetuneModelConfig
from nemo.collections.asr.models.configs.ctc_models_config import ConvASRDecoderConfig
from nemo.core.config import TrainerConfig
from nemo.core.config.modelPT import ModelPTConfig
from nemo.utils.exp_manager import ExpManagerConfig, CallbackParams

config_name = 'st2vec_ctc'

sample_rate = 16000
num_features = 128

model = ST2VecCTCFinetuneModelConfig()

LABELS = [" ", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m",
          "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "'"]
model.labels = LABELS
model.tokenizer = None
model.add_end_space = True

from .spiral_large_pretrain_librilight import st2vec_encoder
encoder = st2vec_encoder
# encoder.freeze_feature_encoder = False
encoder.masking.mask_prob = 0.5
encoder.masking.mask_length = 4
encoder.masking.mask_channel_prob = 0.3
encoder.masking.mask_channel_length = 20
transformer0 = encoder.feature_encoder.conv_transformer_blocks[-2].transformer_block
transformer0.encoder.encoder_layerdrop = 0.1
transformer0.encoder.activation_dropout = 0.1
transformer0.encoder.dropout = 0.1
transformer = encoder.feature_encoder.conv_transformer_blocks[-1].transformer_block
transformer.encoder.encoder_layerdrop = 0.1
transformer.encoder.activation_dropout = 0.1
transformer.encoder.dropout = 0.1

model.encoder = encoder

model.freeze_finetune_updates = 2000

enc_output_dim = transformer.encoder.embedding_dim

model.decoder = ConvASRDecoderConfig(
    feat_in=enc_output_dim,
    proj_upsampling=ProjUpsampling(rate=4, filters=512, kernel_size=(5,), norm_type='ln', act_func='relu', dropout=0.1),
    conv_layers=[Conv1dNormAct(filters=512, kernel_size=(5,), stride=(1,),
                               norm_type='ln', dropout=0.1,
                               act_func='relu'),
                 Conv1dNormAct(filters=512, kernel_size=(5,), stride=(1,),
                               norm_type='ln', dropout=0.1,
                               act_func='relu'),
             Conv1dNormAct(filters=512, kernel_size=(5,), stride=(1,),
                               norm_type='ln', dropout=0.1,
                               act_func='relu'),
             ],
    vocabulary=LABELS,
    blank_pos='after_vocab_last'
)

model.train_ds = DatasetConfig(
    manifest_filepath='manifest_json/librivox-train-clean-100.json',
    labels=LABELS,
    sample_rate=sample_rate,
    batch_size=18,
    shuffle=True,
    max_duration=42.0,
    num_workers=4,
    pin_memory=True,
)
model.validation_ds = DatasetConfig(
    manifest_filepath='manifest_json/librivox-dev-other.json',
    labels=LABELS,
    sample_rate=sample_rate,
    batch_size=18,
    shuffle=False,
    num_workers=4,
)
model.test_ds = DatasetConfig(
    manifest_filepath='manifest_json/librivox-test-clean.json',
    labels=LABELS,
    sample_rate=sample_rate,
    batch_size=18,
    shuffle=False,
    num_workers=4,
)
model.expected_gpu_num = 8
lr = 0.00003
model.optim = AdamWParams(
    lr=lr,
    eps=1e-6,
    betas=[0.9, 0.98],
    weight_decay=0.01,
    sched=PolynomialHoldDecayAnnealingParams(
        min_lr=lr * 0.05,
        warmup_ratio=0.1,
        hold_ratio=0.4,
        max_steps=80000,
    ),
)
trainer = TrainerConfig(
    gpus=1,
    max_epochs=393,
    accelerator='ddp',
    accumulate_grad_batches=1,
    checkpoint_callback=False, # Provided by exp_manager
    logger=False,  # Provided by exp_manager
    log_every_n_steps=50,
    progress_bar_refresh_rate=50,
    num_sanity_val_steps=0,
    check_val_every_n_epoch=1
)
exp_manager = ExpManagerConfig(
    name=config_name,
    create_checkpoint_callback=True,
    checkpoint_callback_params=CallbackParams(
        monitor="val_wer",
        mode="min",
        save_top_k=5
    )
)
cfg = ModelPTConfig(
    name=config_name,
    model=model,
    trainer=trainer,
    exp_manager=exp_manager
)