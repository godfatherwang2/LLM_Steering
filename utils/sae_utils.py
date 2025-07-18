import sys
sys.path.append("..")

import torch
import numpy as np
from typing import List, Tuple, Dict
from torch import Tensor, nn
from jaxtyping import Float, Int, Bool
from collections import defaultdict
from dataclasses import dataclass
from transformers import AutoTokenizer
from tqdm import tqdm
import html
from dataclasses import field
from sae_lens import SAE
from IPython.display import HTML, display
import random
from huggingface_hub import hf_hub_download
from utils.utils import k_largest_indices, value_to_color_rgb

# Neuronpedia layer sparsity widths for Gemma 2 2B
# SAEs in Neuronpedia
layer_sparisity_widths = {"gemma-2-2b": {
    0: {'16k': 105},
    1: {'16k': 102},
    2: {'16k': 142},
    3: {'16k': 59},
    4: {'16k': 124},
    5: {'16k': 68},
    6: {'16k': 70},
    7: {'16k': 69},
    8: {'16k': 71},
    9: {'16k': 73},
    10: {'16k': 77},
    11: {'16k': 80},
    12: {'16k': 82},
    13: {'16k': 84},
    14: {'16k': 84},
    15: {'16k': 78},
    16: {'16k': 78},
    17: {'16k': 77},
    18: {'16k': 74},
    19: {'16k': 73},
    20: {'16k': 71},
    21: {'16k': 70},
    22: {'16k': 72},
    23: {'16k': 74},
    24: {'16k': 73},
    25: {'16k': 116},
},
    "gemma-2-9b": {
    0: {'16k': 129},
    1: {'16k': 69},
    2: {'16k': 67},
    3: {'16k': 90},
    4: {'16k': 91},
    5: {'16k': 77},
    6: {'16k': 93},
    7: {'16k': 92},
    8: {'16k': 99},
    9: {'16k': 100},
    10: {'16k': 113},
    11: {'16k': 118},
    12: {'16k': 130},
    13: {'16k': 132},
    14: {'16k': 67},
    15: {'16k': 131},
    16: {'16k': 75},
    17: {'16k': 73},
    18: {'16k': 71},
    19: {'16k': 132},
    20: {'16k': 68},
    21: {'16k': 129},
    22: {'16k': 123},
    23: {'16k': 120},
    24: {'16k': 114},
    25: {'16k': 114},
    26: {'16k': 116},
    27: {'16k': 118},
    28: {'16k': 119},
    29: {'16k': 119},
    30: {'16k': 120},
    31: {'16k': 114},
    32: {'16k': 111},
    33: {'16k': 114},
    34: {'16k': 114},
    35: {'16k': 120},
    36: {'16k': 120},
    37: {'16k': 124},
    38: {'16k': 128},
    39: {'16k': 131},
    40: {'16k': 125},
    41: {'16k': 113},
    },
    "gemma-2-9b-it": {
        9: {'16k': 88},
        20: {'16k': 91},
        31: {'16k': 76},
    }
}

class JumpReLUSAE(nn.Module):
  def __init__(self, d_model, d_sae):
    # Note that we initialise these to zeros because we're loading in pre-trained weights.
    super().__init__()
    self.W_enc = nn.Parameter(torch.zeros(d_model, d_sae))
    self.W_dec = nn.Parameter(torch.zeros(d_sae, d_model))
    self.threshold = nn.Parameter(torch.zeros(d_sae))
    self.b_enc = nn.Parameter(torch.zeros(d_sae))
    self.b_dec = nn.Parameter(torch.zeros(d_model))

  def encode(self, input_acts):
    pre_acts = input_acts @ self.W_enc + self.b_enc
    mask = (pre_acts > self.threshold)
    acts = mask * torch.nn.functional.relu(pre_acts)
    return acts

  def decode(self, acts):
    return acts @ self.W_dec + self.b_dec
  
  def get_pre_acts(self, input_acts):
      pre_acts = input_acts @ self.W_enc + self.b_enc
      return pre_acts

  def forward(self, acts):
    acts = self.encode(acts)
    recon = self.decode(acts)
    return recon
  
def load_sae(repo_id, sae_id):
    # Load from SAE Lens
    if repo_id == 'gemma-2b-it-res-jb':
        sae, cfg_dict, sparsity = SAE.from_pretrained(
        release = repo_id, # see other options in sae_lens/pretrained_saes.yaml
        sae_id = "blocks.12.hook_resid_post", # won't always be a hook point
        device = "cuda"
        )
    elif repo_id == 'llama_scope_lxr_8x':
        sae, cfg_dict, sparsity = SAE.from_pretrained(
        release = repo_id, # see other options in sae_lens/pretrained_saes.yaml
        sae_id = sae_id, # won't always be a hook point
        device = "cuda"
        )
    # Load Gemma Scope SAEsfrom Hugging Face Hub
    else:
        path_to_params = hf_hub_download(
        repo_id=repo_id,
        filename=f"{sae_id}/params.npz",
        force_download=False,
        )
        params = np.load(path_to_params)
        pt_params = {k: torch.from_numpy(v).cuda() for k, v in params.items()}
        sae = JumpReLUSAE(params['W_enc'].shape[0], params['W_enc'].shape[1])
        sae.load_state_dict(pt_params)
    sae.to('cuda')
    return sae