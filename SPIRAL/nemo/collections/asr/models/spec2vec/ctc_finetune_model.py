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
# This file contains code fragments from 
#
#     ../ctc_bpe_models.py
#     ../ctc_models.py
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
#
import contextlib
import torch

from nemo.core import Serialization


class CTCFinetuneModel(torch.nn.Module):
    def __init__(self, encoder, cfg):
        super().__init__()

        self.encoder = encoder

        self.decoder = Serialization.from_config_dict(cfg.decoder)

        self.freeze_finetune_updates = cfg.freeze_finetune_updates

    def forward(self, input_signal, input_signal_length, global_step):
        ft = False if global_step is None else self.freeze_finetune_updates <= global_step
        with torch.no_grad() if not ft else contextlib.suppress():
            encoded, encoded_len = self.encoder(input_signal, input_signal_length, None, None,
                                                   mask=self.training, features_only=True)

        # [B, T, D] => [B, D, T]
        encoded = encoded.transpose(1, 2)

        # Ensure that shape mismatch does not occur due to padding
        # Due to padding and subsequent downsampling, it may be possible that
        # max sequence length computed does not match the actual max sequence length
        max_output_len = encoded_len.max()
        if encoded.shape[2] != max_output_len:
            encoded = encoded.narrow(dim=2, start=0, length=max_output_len).contiguous()

        logits, encoded_len = self.decoder(encoder_output=encoded, lens=encoded_len, log_prob=False)
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

        with torch.no_grad():
            greedy_predictions = log_probs.argmax(dim=-1, keepdim=False)

        return log_probs, encoded_len, greedy_predictions, logits

