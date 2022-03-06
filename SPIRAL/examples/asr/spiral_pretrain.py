# 2020.03.04 modified from examples/asr/wav2vec_pretrain.py
#            - Supports for SPIRAL and ctc_finetune
#            Huawei Technologies Co., Ltd. <foss@huawei.com>
# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.
#
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
import glob
import itertools
import os

import numpy as np
import pandas
import pytorch_lightning as pl
from omegaconf import OmegaConf

from nemo.collections.asr.parts import compute_wer
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager


"""
Pre-train a wav2vec 2.0 transformer model on audio. Uses a contrastive loss function to pre-train on unlabelled audio,
using a task similar to masked language modeling in NLP. In wav2vec, we mask portions of the audio 
and the model is trained by minimising the distance of the ground truth for the masked section, 
using the ground truth quantized codebook representation. Distractors are obtained from other time steps.
See :class:`Wav2VecCriterion` for more information.

Reference: https://arxiv.org/abs/2006.11477

    python examples/asr/experimental/wav2vec/wav2vec_pretrain.py \
        model.train_ds.manifest_filepath="./examples/asr/train.tsv" \
        model.validation_ds.manifest_filepath="./examples/asr/valid.tsv" \
        trainer.gpus=1 \
        trainer.max_epochs=100
        
Basic run (on CPU for 50 epochs):
    python examples/asr/experimental/wav2vec/wav2vec_pretrain.py \
        model.train_ds.manifest_filepath="./examples/asr/train.tsv" \
        model.validation_ds.manifest_filepath="./examples/asr/valid.tsv" \
        trainer.gpus=1 \
        trainer.max_epochs=50

Using wav2vec-large with mixed precision:
    python examples/asr/experimental/wav2vec/wav2vec_pretrain.py \
        --config-name=wav2vec_pretrain_large \
        model.train_ds.manifest_filepath="./examples/asr/train.tsv" \
        model.validation_ds.manifest_filepath="./examples/asr/valid.tsv" \
        trainer.gpus=1 \
        trainer.max_epochs=100 \
        trainer.precision=16

Add PyTorch Lightning Trainer arguments from CLI:
    python wav2vec.py \
        ... \
        +trainer.fast_dev_run=true

Hydra logs will be found in "$(./outputs/$(date +"%y-%m-%d")/$(date +"%H-%M-%S")/.hydra)"
PTL logs will be found in "$(./outputs/$(date +"%y-%m-%d")/$(date +"%H-%M-%S")/lightning_logs)"

Override some args of optimizer:
    python examples/asr/experimental/wav2vec/wav2vec_pretrain.py \
        model.train_ds.manifest_filepath="./examples/asr/train.tsv" \
        model.validation_ds.manifest_filepath="./examples/asr/valid.tsv" \
        trainer.gpus=2 \
        trainer.max_epochs=2 \
        model.optim.args.params.betas=[0.8,0.5] \
        model.optim.args.params.weight_decay=0.0001

Override optimizer entirely
    python examples/asr/experimental/wav2vec/wav2vec_pretrain.py \
        model.train_ds.manifest_filepath="./examples/asr/train.tsv" \
        model.validation_ds.manifest_filepath="./examples/asr/valid.tsv" \
        trainer.gpus=2 \
        trainer.max_epochs=2 \
        ~model.optim.args \
        +model.optim.args.betas=[0.8,0.5]\
        +model.optim.args.weight_decay=0.0005

"""


def main(cfg, args):
    logging.info("Application config\n" + OmegaConf.to_yaml(cfg))

    if args.run_mode == 'test':
        gpu = 1 if cfg.trainer.gpus != 0 else 0
        trainer = pl.Trainer(gpus=gpu, precision=cfg.trainer.precision)
    else:
        trainer = pl.Trainer(**cfg.trainer)
        exp_manager(trainer, cfg.get("exp_manager", None))

    if args.model_type == 'spiral':
        from nemo.collections.asr.models.st2vec.st2vec_pretrain import ST2VecPretrainModel
        model = ST2VecPretrainModel(cfg=cfg.model, trainer=trainer)
    elif args.model_type == 'ctc_finetune':
        from nemo.collections.asr.models.spec2vec.ctc_finetune import CTCFinetuneModel
        model = CTCFinetuneModel(cfg=cfg.model, trainer=trainer)
    else:
        raise ValueError('Unknown model type: {}'.format(args.model_type))
    print('\nmodel structure:')
    print(model)
    print()

    if args.run_mode == 'test':
        assert args.init_chkpt_dir and args.init_chkpt_file
        assert not args.init_chkpt_file.endswith('.nemo')
        chkpt_path = get_ckpt_path(args.init_chkpt_dir, args.init_chkpt_file)
        model.load_state_from_checkpoint(model, chkpt_path, map_location=None, strict=True)

    if args.run_mode == 'train':
        trainer.fit(model)
    else:
        assert args.run_mode == 'test'
        assert model.prepare_test(trainer)

        test_results = trainer.test(model)

        if isinstance(test_results, list):
            test_results = test_results[-1]
        if test_results is not None and 'decode_results' in test_results:
            if not os.path.exists(args.model_save_dir):
                os.mkdir(args.model_save_dir)
            test_output_dir = os.path.join(args.model_save_dir, 'test_results')
            assert not os.path.exists(test_output_dir)
            os.mkdir(test_output_dir)

            references, hypotheses = test_results['decode_results']
            log_analyze_wer(list(zip(itertools.count(1), references, hypotheses)),
                            output_dir=test_output_dir, output_fname_prefix='decode_results', cfg=cfg)

        if 'test_logprob' in test_results and args.save_logits:
            assert 'test_logprob_len' in test_results
            test_logp_fp = os.path.join(args.model_save_dir, 'test_logp.npy')
            assert not os.path.exists(test_logp_fp)
            test_logits_fp = os.path.join(args.model_save_dir, 'test_logits.npy')
            assert not os.path.exists(test_logits_fp)

            test_logp_list = []
            test_logits_list = []
            for logprob_b, logits_b, logprob_b_len in zip(test_results['test_logprob'], test_results['test_logits'], test_results['test_logprob_len']):
                for i, len_i in enumerate(logprob_b_len):
                    test_logp_list.append(logprob_b[i, :len_i, :])
                    test_logits_list.append(logits_b[i, :len_i, :])

            print('save logp of {} entries'.format(len(test_logp_list)))
            with open(test_logp_fp, 'wb') as f:
                np.save(f, test_logp_list)
            print('save logits of {} entries'.format(len(test_logits_list)))
            with open(test_logits_fp, 'wb') as f:
                np.save(f, test_logits_list)


def get_ckpt_path(ckpt_dir, ckpt_name):
  ckpt_path = os.path.join(ckpt_dir, ckpt_name)
  if '*' not in ckpt_path:
    return ckpt_path

  ckpt_fs = glob.glob(ckpt_path)
  if len(ckpt_fs) != 1:
    raise ValueError('expect 1 ckpt file, but got {}'.format(len(ckpt_fs)))
  return ckpt_fs[0]


def log_analyze_wer(inputs, output_dir, output_fname_prefix, cfg):
    df = pandas.DataFrame(inputs, columns=['wav_filename', 'transcript', 'predicted_transcript'])
    df.to_csv(os.path.join(output_dir, output_fname_prefix + '.csv'), index=False)

    print()
    (str_summary, str_details), _ = compute_wer.analyze(inputs, output_html_path=os.path.join(output_dir, output_fname_prefix + '_diagnosis.html'))
    print()
    print(str_details)
    print(str_summary)

    with open(os.path.join(output_dir, 'wer.log'), mode='w') as wer_f:
        wer_f.write(str_details)
        wer_f.write('\n')
        wer_f.write(str_summary)
