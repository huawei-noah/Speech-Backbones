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
# This source code contains parts from
#
#     wav2vec/wav2vec_model.py
# 
# with the following copyright statement.
#
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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


# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) Facebook, Inc. and its affiliates.
#

import logging
from math import ceil
from typing import Dict, Optional, Union

import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer

from nemo.collections.asr.data import audio_to_text_dataset
from nemo.collections.asr.losses.similarityloss import NegativeCosineSimilarityLoss
from nemo.collections.asr.losses.wav2vecloss import Wav2VecLoss
from nemo.collections.asr.models.st2vec.st2vec_model import ST2VecEncoder
from nemo.collections.asr.parts.perturb import process_augmentations, RandomNoisePerturbation, AudioAugmentor
from nemo.core import ModelPT
from nemo.core.classes.common import PretrainedModelInfo


def buffered_arange(max):
    if not hasattr(buffered_arange, "buf"):
        buffered_arange.buf = torch.LongTensor()
    if max > buffered_arange.buf.numel():
        buffered_arange.buf.resize_(max)
        torch.arange(max, out=buffered_arange.buf)
    return buffered_arange.buf[:max]


class ST2VecPretrainModel(ModelPT):
    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        # Get global rank and total number of GPU workers for IterableDataset partitioning, if applicable
        self.global_rank = 0
        self.world_size = 1
        self.local_rank = 0
        if trainer is not None:
            self.global_rank = (trainer.node_rank * trainer.num_gpus) + trainer.local_rank
            self.world_size = trainer.num_nodes * trainer.num_gpus
            self.local_rank = trainer.local_rank

        super().__init__(cfg=cfg, trainer=trainer)

        if isinstance(cfg, dict):
            cfg = OmegaConf.create(cfg)
        elif not isinstance(cfg, DictConfig):
            raise ValueError(f"cfg was type: {type(cfg)}. Expected either a dict or a DictConfig")

        if cfg.encoder_type == 'st':
            self.st2vec_encoder = ST2VecEncoder(cfg.st2vec_encoder)
        else:
            raise ValueError('unknown encoder type: {}'.format(cfg.encoder_type))

        self.loss_type = cfg.loss_type
        if self.loss_type == 'neg_cos_sim':
            self.loss = NegativeCosineSimilarityLoss()
        else:
            assert self.loss_type == 'wav2vec'
            self.loss = Wav2VecLoss(
                feature_loss_weight=0.0,
                prob_ppl_weight=cfg.loss.prob_ppl_weight,
                logit_temp=cfg.logit_temp,
            )

        self._prev_log_step = -1

    def training_step(self, batch, batch_idx):
        loss, contrastive_loss, prob_ppl_loss, cur_temp, prob_ppl, _ = self._step(batch)

        if self.global_step > self._prev_log_step:
            self._prev_log_step = self.global_step
            tensorboard = self.logger.experiment
            tensorboard.add_scalar('loss', loss, self.global_step)
            if prob_ppl_loss is not None:
                tensorboard.add_scalar('contrastive_loss', contrastive_loss, self.global_step)
                tensorboard.add_scalar('prob_ppl_loss', prob_ppl_loss, self.global_step)
                tensorboard.add_scalar('temp', cur_temp, self.global_step)
                tensorboard.add_scalar('prob_ppl', prob_ppl, self.global_step)
            tensorboard.add_scalar('learning_rate', self._optimizer.param_groups[0]['lr'], self.global_step)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        loss, contrastive_loss, prob_ppl_loss, _, prob_ppl, accuracy = self._step(batch)
        self.log('val_loss', loss, prog_bar=True, on_epoch=True, sync_dist=True)
        if prob_ppl is not None:
            self.log('val_contrastive_loss', contrastive_loss, prog_bar=False, on_step=False, on_epoch=True, sync_dist=False)
            self.log('val_prob_ppl', prob_ppl, prog_bar=False, on_step=False, on_epoch=True, sync_dist=False)
        if accuracy is not None:
            self.log('val_accuracy', accuracy, prog_bar=True, on_step=False, on_epoch=True, sync_dist=False)

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        loss, contrastive_loss, prob_ppl_loss, _, _, accuracy = self._step(batch)
        self.log('test_loss', loss, prog_bar=True, on_epoch=True, sync_dist=True)
        if accuracy is not None:
            self.log('test_accuracy', accuracy, prog_bar=True, on_step=False, on_epoch=True, sync_dist=False)

    def _step(self, batch):
        if len(batch) == 4:
            audio_signal, audio_lengths, p_audio_signal, p_audio_lengths = batch
        else:
            audio_signal, audio_lengths = batch
            p_audio_signal, p_audio_lengths = None, None

        logits, targets, sampled_negatives, _, prob_ppl_loss, cur_temp, prob_ppl = self(
            source=audio_signal, source_lens=audio_lengths, p_source=p_audio_signal, p_source_lens=p_audio_lengths
        )
        if self.loss_type == 'neg_cos_sim':
            loss = self.loss(predictions=logits, targets=targets)
            contrastive_loss, prob_ppl_loss, accuracy = None, None, None
        else:
            assert self.loss_type == 'wav2vec'
            loss, contrastive_loss, _, prob_ppl_loss, accuracy = self.loss(
                logits=logits,
                targets=targets,
                negatives=sampled_negatives,
                prob_ppl_loss=prob_ppl_loss,
                feature_loss=None,
                compute_accuracy=not self.training
            )
        return loss, contrastive_loss, prob_ppl_loss, cur_temp, prob_ppl, accuracy

    @classmethod
    def list_available_models(cls) -> Optional[PretrainedModelInfo]:
        return None

    def forward(self, source, source_lens, p_source, p_source_lens, mask=True, features_only=False) -> tuple:
        return self.st2vec_encoder(source, source_lens, p_source, p_source_lens, mask=mask, features_only=features_only,
                                   global_step=self.global_step)

    def setup_training_data(self, train_data_config: Optional[Union[DictConfig, Dict]]):
        if 'shuffle' not in train_data_config:
            train_data_config['shuffle'] = True

        # preserve config
        self._update_dataset_config(dataset_name='train', config=train_data_config)

        self._train_dl = self._setup_dataloader_from_config(config=train_data_config, noise_perturb_config=self._cfg['noise_perturb'])

        # Need to set this because if using an IterableDataset, the length of the dataloader is the total number
        # of samples rather than the number of batches, and this messes up the tqdm progress bar.
        # So we set the number of steps manually (to the correct number) to fix this.
        if 'is_tarred' in train_data_config and train_data_config['is_tarred']:
            # We also need to check if limit_train_batches is already set.
            # If it's an int, we assume that the user has set it to something sane, i.e. <= # training batches,
            # and don't change it. Otherwise, adjust batches accordingly if it's a float (including 1.0).
            if isinstance(self._trainer.limit_train_batches, float):
                self._trainer.limit_train_batches = int(
                    self._trainer.limit_train_batches
                    * ceil((len(self._train_dl.dataset) / self.world_size) / train_data_config['batch_size'])
                )

    def setup_validation_data(self, val_data_config: Optional[Union[DictConfig, Dict]]):
        if 'shuffle' not in val_data_config:
            val_data_config['shuffle'] = False

        # preserve config
        self._update_dataset_config(dataset_name='validation', config=val_data_config)

        self._validation_dl = self._setup_dataloader_from_config(config=val_data_config, noise_perturb_config=None)

    def setup_test_data(self, test_data_config: Optional[Union[DictConfig, Dict]]):
        if 'shuffle' not in test_data_config:
            test_data_config['shuffle'] = False

        # preserve config
        self._update_dataset_config(dataset_name='test', config=test_data_config)

        self._test_dl = self._setup_dataloader_from_config(config=test_data_config, noise_perturb_config=None)

    def _setup_dataloader_from_config(self, config: Optional[Dict], noise_perturb_config):

        if noise_perturb_config is not None:
            noise_perturb = RandomNoisePerturbation(**noise_perturb_config)
            augmentor = AudioAugmentor(perturbations=[(1.0, noise_perturb)])
            return_both = True
        else:
            augmentor = None
            return_both = False

        shuffle = config['shuffle']

        if 'manifest_filepath' in config and config['manifest_filepath'] is None:
            logging.warning(f"Could not load dataset as `manifest_filepath` was None. Provided config : {config}")
            return None

        dataset = audio_to_text_dataset.get_audio_dataset(config=config, augmentor=augmentor, return_both=return_both)

        return torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=config['batch_size'],
            collate_fn=dataset.collate_fn,
            drop_last=config.get('drop_last', False),
            shuffle=shuffle,
            num_workers=config.get('num_workers', 0),
            pin_memory=config.get('pin_memory', False),
        )
