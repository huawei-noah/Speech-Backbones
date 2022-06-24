# SPIRAL
This repo is the official implementation of ["SPIRAL: Self-supervised Perturbation-Invariant Representation Learning for Speech Pre-Training"](https://arxiv.org/abs/2201.10207).
This code is based on the repository developed by Nvidia: ["Nemo"](https://github.com/NVIDIA/NeMo)

## Installation

```
pip install -r requirements/requirements.txt
pip install -r requirements/requirements_asr.txt
```

## Pre-training

### Data preparing
For pre-training, you should prepare a directory containing wav files, we recommend splitting each wav into 10 to 20 seconds.
And then prepare a json manifest file in the following format.
```
{"audio_filepath": "train-clean-100-wav/374-180298-0000.wav", "duration": 14.53, "text": "chapter sixteen i might have told you of the beginning of this liaison in a few lines but i wanted you to see every step by which we came i to agree to whatever marguerite wished"}
{"audio_filepath": "train-clean-100-wav/374-180298-0001.wav", "duration": 16.085, "text": "marguerite to be unable to live apart from me it was the day after the evening when she came to see me that i sent her manon lescaut from that time seeing that i could not change my mistress's life i changed my own"}
```
The text field is not required for pre-training.
You can use scripts/get_librispeech_data.py to prepare Lirbripseech data.

### Pre-train SPIRAL base model
Pre-training of SPIRAL base model on Librispeech 960 with 2 * 8 gpus
```
python run_spiral.py \
--config_name=spiral_base_pretrain_ls960 \
--config_path=examples/asr/conf/spiral \
--model_type=st2vec \
--num_nodes=2 \
--num_gpus=8 \
--data_dir=DIRECTORY_OF_TRAIN_DATA \
--model_save_dir=DIRECTORY_FOR_CHECKPOINTS
```

Before launch the training, the following environment variables should be defined on each node.
* MASTER_PORT - required; has to be a free port on machine with NODE_RANK 0
* MASTER_ADDR - required (except for NODE_RANK 0); address of NODE_RANK 0 node
* WORLD_SIZE - required; how many nodes are in the cluster
* NODE_RANK - required; id of the node in the cluster
For more information for multi-node multi-gpu training, please refer to https://pytorch-lightning.readthedocs.io/en/stable/clouds/cluster.html.

if you want to use horovod instead of pytorch DDP for distributed training, add the following arguments
```
--use_horovod=true \
```

### Pre-train SPIRAL large model
Pre-training of SPIRAL large model on Libri-Light  with 4 * 8 gpus, substitute the following arguments of the above command.
```
--config_name=spiral_large_pretrain_librilight \
--num_nodes=4 \
--num_gpus=8 \
```


## Fine-tune a pre-trained model with CTC

Fine-tuning SPIRAL base with Librispeech clean 100 subset with 1 * 8 gpus.
```
python run_spiral.py \
--config_name=spiral_base_finetune_ls100_subword \
--config_path=examples/asr/conf/spiral \
--model_type=ctc_finetune \
--num_nodes=1 \
--num_gpus=8 \
--data_dir=DIRECTORY_OF_TRAIN_DATA \
--model_save_dir=DIRECTORY_FOR_CHECKPOINTS \
--init_chkpt_dir=DIRECTORY_FOR_PRETRAINED_CHECKPOINTS \
--init_chkpt_file=checkpoints/st2vec-last.ckpt
```

Fine-tuning SPIRAL large with Librispeech clean 100 subset with 1 * 8 gpus, substitute the following flags of the above command.
```
--config_name=spiral_large_finetune_ls100_subword \
```

Fine-tuning SPIRAL large with Librispeech 960  with 2 * 8 gpus, substitute the following flags of the above command.
```
--config_name=spiral_large_finetune_ls960_subword \
--num_nodes=2 \
--num_gpus=8 \
```


## Evaluation


Evaluate a fine-tuned SPIRAL base model,

```
python run_spiral.py \
--config_name=spiral_base_finetune_ls100_subword \
--config_path=examples/asr/conf/spiral \
--model_type=ctc_finetune \
--run_mode=test \
--num_nodes=1 \
--num_gpus=1 \
--data_dir=DIRECTORY_OF_EVALUATION_DATA \
--model_save_dir=DIRECTORY_FOR_CHECKPOINTS \
--init_chkpt_dir=DIRECTORY_FOR_PRETRAINED_CHECKPOINTS \
--init_chkpt_file=checkpoints/BEST.ckpt \
--test_manifest=path/to/evaluation_manifest.json
```

You can add the following arguments to save the logits for the test data, it can be used to evaluate with a externel language model.
```
--save_logits=true \
```

## Pre-trained models

We have released the **pretrained** models in [Hugging Face 🤗](https://huggingface.co/huawei-noah). Please enjoy!

| Model | Pretrained Data | Download |
| ----------- | ----------- | -------- |
| SPIRAL-base | 960 hrs LibriSpeech (LS960) | [huawei-noah/SPIRAL-base](https://huggingface.co/huawei-noah/SPIRAL-base) |
| SPIRAL-base-MCT| noise-robustness training with LS960 <br> and [ICASSP 2021 DNS Challenge](https://github.com/microsoft/DNS-Challenge/tree/icassp2021-final) noise dataset | [huawei-noah/SPIRAL-base-MCT](https://huggingface.co/huawei-noah/SPIRAL-base-MCT) |
| SPIRAL-Large | 60k hrs Libri-Light| [huawei-noah/SPIRAL-Large](https://huggingface.co/huawei-noah/SPIRAL-Large) |


## License
This repository is released under the Apache 2.0 license as found in the [LICENSE](LICENSE) file.

## Citation
If you find SPIRAL useful for your research, we would appreciate a citation via
```
@inproceedings{huang2022spiral,
  title={{SPIRAL}: Self-supervised Perturbation-Invariant Representation Learning for Speech Pre-Training},
  author={Wenyong Huang and Zhenhe Zhang and Yu Ting Yeung and Xin Jiang and Qun Liu},
  booktitle={International Conference on Learning Representations},
  year={2022},
  url={https://openreview.net/forum?id=TBpg4PnXhYH}
}
```
