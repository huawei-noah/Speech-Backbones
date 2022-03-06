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

import argparse
import glob
import os
import sys
from importlib import import_module

from hydra.experimental import compose, initialize
from omegaconf import OmegaConf, ValidationError


def main(argv=None):
    parser = argparse.ArgumentParser(description='Run training',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_dir', type=str, help='Dataset Path')
    parser.add_argument('--manifest_dir', type=str, default='')
    parser.add_argument('--model_save_dir', type=str, help='model save dir')
    parser.add_argument('--tensorboard_dir', type=str, help='tensorboard dir')
    parser.add_argument('--log_dir', type=str, help='log dir')
    parser.add_argument('--chkpt_dir', type=str, default='', help='log dir')
    parser.add_argument('--config_path', type=str, help='task config path')
    parser.add_argument('--config_name', type=str, help='task config name')
    parser.add_argument('--structured_config', type=str2bool, default=True)
    parser.add_argument('--num_gpus', type=int, default=0, help='number of gpus')
    parser.add_argument('--num_nodes', type=int, default=1, help='number of nodes')
    parser.add_argument('--use_horovod', type=str2bool, default=False)
    parser.add_argument('--resume_if_exists', type=str2bool, default=True)
    parser.add_argument('--run_mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--test_mode', type=str, default='multi_gpu', help='run mode')
    parser.add_argument('--init_chkpt_dir', type=str, default='')
    parser.add_argument('--init_chkpt_file', type=str, default='')
    parser.add_argument('--init_model_partial', type=str2bool, default=False)
    parser.add_argument('--use_chkpt_hparams', type=str2bool, default=False)
    parser.add_argument('--load_model_skip_var', type=str, default='')
    parser.add_argument('--test_manifest', type=str, default='')
    parser.add_argument('--model_type', type=str, default='spiral', choices=['spiral', 'ctc_finetune'])
    parser.add_argument('--finetune_from_scratch', type=str2bool, default=False)
    parser.add_argument('--dev_data_dup_factor', type=int, default=0)
    parser.add_argument('--use_teacher_encoder', type=str2bool, default=False)
    parser.add_argument('--save_logits', type=str2bool, default=False)

    args = parser.parse_args(args=argv)
    print('training args: {}'.format(args))

    manifest_dir = args.data_dir
    if args.manifest_dir:
        manifest_dir = args.manifest_dir

    if args.structured_config:
        cfg_module = import_module(os.path.join(args.config_path, args.config_name).replace('/', '.'))
        cfg = cfg_module.cfg

        cfg.exp_manager.explicit_log_dir = args.model_save_dir
        cfg.exp_manager.resume_if_exists = args.resume_if_exists

        cfg.model.train_ds.manifest_dir = manifest_dir
        cfg.model.validation_ds.manifest_dir = manifest_dir
        cfg.model.train_ds.data_dir = args.data_dir
        cfg.model.validation_ds.data_dir = args.data_dir
        cfg.model.test_ds.manifest_dir = manifest_dir
        cfg.model.test_ds.data_dir = args.data_dir
        if args.test_manifest:
            cfg.model.test_ds.manifest_filepath = args.test_manifest
        if args.dev_data_dup_factor > 0:
            cfg.model.validation_ds.dup_factor = args.dev_data_dup_factor

        if hasattr(cfg.model, 'tokenizer') and cfg.model.tokenizer is not None:
            cfg.model.tokenizer.dir = manifest_dir

        if args.use_horovod:
            cfg.trainer.accelerator = 'horovod'
            cfg.trainer.gpus = 1  # number of GPUs/machines provided on command-line
            cfg.model.optim.lr /= cfg.model.expected_gpu_num  # lightning will scale up the lr by number of GPUs
        else:
            cfg.trainer.gpus = args.num_gpus
            cfg.trainer.num_nodes = args.num_nodes

        if args.model_type == 'ctc_finetune':
            if args.run_mode == 'test':
                cfg.model.pretrain_chkpt_path = None
            elif args.finetune_from_scratch:
                assert args.init_chkpt_dir == '' and args.init_chkpt_file == ''
                cfg.model.pretrain_chkpt_path = None
            else:
                assert args.init_chkpt_dir != '' and args.init_chkpt_file != ''
                cfg.model.pretrain_chkpt_path = get_ckpt_path(args.init_chkpt_dir, args.init_chkpt_file)

            if args.use_teacher_encoder:
                cfg.model.use_teacher_encoder = True

        try:
            cfg = OmegaConf.structured(cfg)
            OmegaConf.set_struct(cfg, True)
        except ValidationError:
            print('found type error in config', file=sys.stderr, flush=True)
            raise

        print('train config:', flush=True)
        print(OmegaConf.to_yaml(cfg), flush=True)
    else:
        overrides = {'+exp_manager.explicit_log_dir': args.model_save_dir,
                     'model.train_ds.manifest_dir': manifest_dir,
                     'model.validation_ds.manifest_dir': manifest_dir,
                     'model.train_ds.data_dir': args.data_dir,
                     'model.validation_ds.data_dir': args.data_dir,
                     'trainer.gpus': args.num_gpus,
                     }
        overrides['model.test_ds.manifest_dir'] = manifest_dir
        overrides['model.test_ds.data_dir'] = args.data_dir

        if args.use_horovod:
            overrides['trainer.accelerator'] = 'horovod'
            overrides['trainer.gpus'] = 1

        overrides_str = ['{}={}'.format(k, v) for k, v in overrides.items()]
        cfg = get_hydra_config(config_path=args.config_path, config_name=args.config_name, overrides=overrides_str)
        if args.use_horovod:
            cfg.model.optim.lr /= cfg.model.expected_gpu_num  # lightning will scale up the lr by number of GPUs
    run(cfg, args)


def get_ckpt_path(ckpt_dir, ckpt_name):
  ckpt_path = os.path.join(ckpt_dir, ckpt_name)
  if '*' not in ckpt_path:
    return ckpt_path

  ckpt_fs = glob.glob(ckpt_path)
  if len(ckpt_fs) != 1:
    raise ValueError('expect 1 ckpt file, but got {}'.format(len(ckpt_fs)))
  return ckpt_fs[0]


def str2bool(s):
    s = s.lower()
    if s == 'true':
        return True
    elif s == 'false':
        return False
    else:
        raise ValueError('invalid value: {}, must be true or false'.format(s))


def get_hydra_config(config_path, config_name, overrides):
    initialize(config_path=config_path, job_name="run_speech_to_text")
    return compose(config_name=config_name, overrides=overrides)


def run(cfg, args):
    from examples.asr import spiral_pretrain
    spiral_pretrain.main(cfg, args)


if __name__ == '__main__':
    main()
