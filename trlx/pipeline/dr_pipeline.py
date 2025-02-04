import json
import os
import time
from functools import partial
from typing import Iterable

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from trlx.data.dr_types import DRRLBatch, DRRLElement
from trlx.pipeline import BaseRolloutStore


class DRRolloutStorage(BaseRolloutStore):
    """
    Rollout storage for training dr
    """

    def __init__(self, pad_token_id, padding_side):
        super().__init__()

        self.pad_token_id = pad_token_id
        self.padding_side = padding_side
        self.history: Iterable[DRRLElement] = [None]

    def push(self, exps: Iterable[DRRLElement]):
        self.history += exps

    def clear_history(self):
        self.history = []

    def export_history(self, location: str, only_text=True):
        assert os.path.exists(location)

        fpath = os.path.join(location, f"epoch-{str(time.time())}.json")

        def exp_to_dict(exp):
            return {k: v.cpu().tolist() for k, v in exp.__dict__.items()}

        def filter_text(d, only_text):
            if only_text:
                keys = list(d.keys())
                for key in keys:
                    if key != "query_tensor" and key != "response_tensor":
                        d.pop(key)
            return d

        data = [filter_text(exp_to_dict(exp), only_text) for exp in self.history]
        with open(fpath, "w") as f:
            f.write(json.dumps(data, indent=2))

    def __getitem__(self, index: int) -> DRRLElement:
        return self.history[index]

    def __len__(self) -> int:
        return len(self.history)

    def create_loader(
        self,
        batch_size: int,
        shuffle: bool,
    ) -> DataLoader:
        def collate_fn(elems: Iterable[DRRLElement]):
            if self.padding_side == "right":
                # Right padding of already right-padded queries
                query_tensors = pad_sequence(
                    [elem.query_tensor for elem in elems],
                    padding_value=self.pad_token_id,
                    batch_first=True,
                )
            else:
                # Left padding of already left-padded queries
                query_tensors = pad_sequence(
                    [elem.query_tensor.flip(0) for elem in elems],
                    padding_value=self.pad_token_id,
                    batch_first=True,
                ).flip(1)

            num_samples = len(elems)
            pad_response_tensors = pad_sequence(
                [elem.response_tensor[0] for elem in elems] + [elem.response_tensor[1] for elem in elems],
                padding_value=self.pad_token_id,
                batch_first=True,
            )
            pad_response_tensors = torch.stack([pad_response_tensors[:num_samples], pad_response_tensors[num_samples:]])
            pad_logprobs = pad_sequence(
                [elem.logprobs[0] for elem in elems] + [elem.logprobs[1] for elem in elems],
                padding_value=0.0,
                batch_first=True,
            )
            pad_logprobs = torch.stack([pad_logprobs[:num_samples], pad_logprobs[num_samples:]])
            pad_chosed = torch.stack([elem.chosen for elem in elems]).transpose(0, 1)
            pad_rewards = pad_sequence([elem.rewards for elem in elems]).transpose(0, 1)
            return DRRLBatch(
                query_tensors,
                pad_response_tensors,
                pad_logprobs,
                pad_chosed,
                pad_rewards,
            )

        return DataLoader(self, batch_size, shuffle=shuffle, collate_fn=collate_fn)