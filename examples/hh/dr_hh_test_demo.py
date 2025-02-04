import json
import math
import os
import sys
sys.path.insert(0, "/home/kyle/Documents/lab/trlx")  
from itertools import islice

import numpy as np
import torch
import tritonclient.grpc as client_util
from datasets import load_dataset
from huggingface_hub import snapshot_download
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from tritonclient.utils import np_to_triton_dtype

import trlx
from trlx.data.default_configs import (
    ModelConfig,
    OptimizerConfig,
    DRConfig,
    SchedulerConfig,
    TokenizerConfig,
    TrainConfig,
    TRLConfig,
)

def main(hparams={}):
  

    dataset = load_dataset("Dahoas/rm-static")
    print(dataset)
    print(dataset["train"][1])
    prompts = [{"prompt": x["prompt"], "original_output": x["chosen"]} for x in dataset["train"]]
    eval_prompts = [{"prompt": x["prompt"], "original_output": x["chosen"]} for x in islice(dataset["test"], 280)]
    print("------------------------------------------------------")
    print(prompts[1])




if __name__ == "__main__":
    hparams = {} if len(sys.argv) == 1 else json.loads(sys.argv[1])
    main(hparams)