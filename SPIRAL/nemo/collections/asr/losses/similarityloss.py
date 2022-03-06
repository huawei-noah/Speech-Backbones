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
#

import torch

from nemo.core import Loss


class NegativeCosineSimilarityLoss(Loss):

    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        assert reduction == 'mean'
        self.reduction = reduction

    def forward(self, predictions: torch.tensor, targets: torch.tensor):
        similarity_scores = torch.cosine_similarity(predictions.float(), targets.float(), dim=-1).type_as(predictions)
        loss = 1.0 - similarity_scores.mean()
        return loss
