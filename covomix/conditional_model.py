import time
from math import ceil
import warnings
from typing import List, Tuple, Union

import torch
import pytorch_lightning as pl
from torch_ema import ExponentialMovingAverage

from covomix.util.inference import evaluate_acoustic_predictor_hubert,  evaluate_text2semantic, evaluate_acoustic_predictor_hubert_2input_2output, evaluate_acoustic_predictor_hubert_2input_1output
from covomix.util.other import pad_spec
import numpy as np
from functools import partial
from tqdm import tqdm
from torchvision.utils import make_grid
from einops import rearrange, repeat
from diffusers import AutoencoderKL, DDPMScheduler
import matplotlib.pyplot as plt
import os
from covomix.util.other import energy_ratios
from torch import nn
import torch.nn.functional as F

from covomix.util.DDPM_utils import log_txt_as_img, exists, default, ismap, isimage, mean_flat, count_params, instantiate_from_config
from covomix.util.DDPM_utils import make_beta_schedule, extract_into_tensor, noise_like
from typing import Optional
import numpy as np
import torchaudio
import onnxruntime as ort
import torchaudio.compliance.kaldi as kaldi
import random

from covomix.covomix_model.acoustic import CoVoMix, ConditionalFlowMatcherWrapper
from covomix.covomix_model.text2semantic import TextToSemantic,TextToSemanticWrapper


class CoVoMixModel(pl.LightningModule):
    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument("--lr", type=float, default=1e-4, help="The learning rate (1e-4 by default)")
        parser.add_argument("--ema_decay", type=float, default=0.999, help="The parameter EMA decay constant (0.999 by default)")
        parser.add_argument("--t_eps", type=float, default=0.03, help="The minimum time (3e-2 by default)")
        parser.add_argument("--num_eval_files", type=int, default=20, help="Number of files for speech enhancement performance evaluation during training. Pass 0 to turn off (no checkpoints based on evaluation metrics will be generated).")
        parser.add_argument("--loss_type", type=str, default="mse", choices=("mse", "mae"), help="The type of loss function to use.")
        parser.add_argument("--classifier_free", type=str, default="no", choices=("yes", "no"), help="Use classifier free guidance or not")
        parser.add_argument("--CoVoMix_dim", type=int, default=80,  help="CoVoMix feature dim, 80 for melspectrogram")
        parser.add_argument("--CoVoMix_num_phoneme_tokens", type=int, default=256,  help="CoVoMix_num_phoneme_tokens")
        parser.add_argument("--CoVoMix_depth", type=int, default=2,  help="CoVoMix_depth")
        parser.add_argument("--CoVoMix_dim_head", type=int, default=64,  help="CoVoMix_dim_head")
        parser.add_argument("--CoVoMix_heads", type=int, default=16,  help="CoVoMix_heads")
        parser.add_argument("--inference_step", type=int, default=10, help="inference step for CEM")
        parser.add_argument("--cond_drop_prob", type=float, default=0.0, help="probability of drop condition")
        parser.add_argument("--CoVoMix_model", type=str, default="acoustic", help="CoVoMix Model")
        parser.add_argument("--CoVoMix_dp_loss_target", type=str, default="log", help="CoVoMix Duration predictor loss in normal scale or in log scale")
        parser.add_argument("--CoVoMix_dp_flow",  action='store_true', help="CoVoMix Duration predictor is trained on flow loss or regression loss")
        parser.add_argument("--CoVoMix_dim_transformer", type=int, default=1024, help="CoVoMix transformer hidden dimension ")
        parser.add_argument("--lr_scheduler",  action='store_true', help="lr scheduler")
        parser.add_argument("--text2semantic",  action='store_true', help="the model to trained is a text-to-semantic model")
        parser.add_argument("--twocondition_twooutput",  action='store_true', help="configure of CoVoMix acoustic model (2 inputs and 2 individual outputs)")
        parser.add_argument("--twocondition_oneoutput",  action='store_true', help="configure of CoVoMix acoustic model (2 inputs and 1 mixed outputs)")
        parser.add_argument("--text2semantic_tokens",  type=int, default=513,  help="number of text2semantic semntic token vocabulary")
        parser.add_argument("--text2semantic_target_depth",  type=int, default=4,  help="text2semantic decoder arch")
        parser.add_argument("--text2semantic_source_depth",  type=int, default=4,  help="text2semantic encoder arch")
        parser.add_argument("--text2semantic_head",  type=int, default=8,  help="text2semantic head number")
        parser.add_argument("--no_source_transformer",  action='store_true',  help="do not use text2semantic encoder")
        parser.add_argument("--text2semantic_two_output",  action='store_true',  help="text2semantic has two output sequence ")
        parser.add_argument("--num_text_token_ids",  type=int, default=30530,   help="text2semantic text tokenizer vocabulary size ")
        parser.add_argument("--target_transformer_dim",  type=int, default=512,   help="text2semantic tokenizer vocabulary size ")
        parser.add_argument("--t2s_batch_size",  type=int, default=5,   help="text2semantic batch size ")
        parser.add_argument("--speechturn_refiner",  action='store_true',  help="whether use 2 input for text2semantic model")
        return parser

    def __init__(
        self, lr=1e-4, ema_decay=0.999, t_eps=3e-2,
        num_eval_files=20, loss_type='mse', data_module_cls=None, 
        CoVoMix_dim = 512, CoVoMix_num_phoneme_tokens = 256, 
        CoVoMix_depth=2,CoVoMix_dim_head=64,CoVoMix_heads=16,
        cond_drop_prob = 0.0,CoVoMix_model = "acoustic",
        CoVoMix_dp_loss_target="log",CoVoMix_dp_flow=False,
        CoVoMix_dim_transformer = 1024,
        medium_file_save_dir = "/tmpdir/CoVoMix_fisher_results/",
        lr_scheduler= False,total_epochs=500,wake_up_epochs = 15, decay_start_epoch=30,
        text2semantic=False,twocondition_oneoutput = False,twocondition_twooutput = False,
        text2semantic_tokens= 513, 
        text2semantic_target_depth=4,  text2semantic_head= 8, no_source_transformer = False, 
        text2semantic_two_output = False,  
        num_text_token_ids = 30530, target_transformer_dim = None,t2s_batch_size = 5,
        speechturn_refiner =  False, text2semantic_source_depth  = 4,
        **kwargs
    ):
        super().__init__()
        # Initialize Backbone DNN
        self.CoVoMix_model = CoVoMix_model
        self.text2semantic = text2semantic
        self.text2semantic_source_depth = text2semantic_source_depth
        print("CoVoMix_model",CoVoMix_model,"text2semantic",text2semantic)
        if not self.text2semantic:
            if self.CoVoMix_model == "acoustic":
                model = CoVoMix(
                    dim = CoVoMix_dim_transformer,
                    dim_in = CoVoMix_dim,
                    num_phoneme_tokens = CoVoMix_num_phoneme_tokens,
                    depth = CoVoMix_depth,
                    dim_head = CoVoMix_dim_head,
                    heads = CoVoMix_heads,
                    twocondition_twooutput=twocondition_twooutput,
                    twocondition_oneoutput=twocondition_oneoutput,
                )
            else:
                raise error("CoVoMix_model should be acoustic or duration_predictor")
            self.cfm_wrapper = ConditionalFlowMatcherWrapper(
                CoVoMix = model,
                use_torchode = False,   # by default will use torchdiffeq with midpoint as in paper, but can use the promising torchode package too
                cond_drop_prob = cond_drop_prob,
            )
        
        elif self.text2semantic:
            print("init text2semantic model")
            print("text2semantic_two_output",text2semantic_two_output)
            if target_transformer_dim == None:
                target_transformer_dim = CoVoMix_dim_transformer
            model = TextToSemantic(
                dim = CoVoMix_dim_transformer,
                source_depth=text2semantic_source_depth,
                target_depth=text2semantic_target_depth,
                semantic_pad_id = -1,
                text_pad_id=0,
                heads = text2semantic_head,
                num_text_token_ids=num_text_token_ids,
                num_semantic_token_ids=text2semantic_tokens,
                no_source_transformer = no_source_transformer,
                two_output = text2semantic_two_output,
                two_input = speechturn_refiner,
                target_transformer_dim = target_transformer_dim,
                )
            self.cfm_wrapper = TextToSemanticWrapper(model = model)
        else: 
            raise error("Only support CoVoMix and text2semantic")
        self.target_transformer_dim = target_transformer_dim
        self.twocondition_oneoutput = twocondition_oneoutput
        self.twocondition_twooutput = twocondition_twooutput
        self.lr = lr
        self.ema_decay = ema_decay
        self.ema = ExponentialMovingAverage(self.parameters(), decay=self.ema_decay)
        self._error_loading_ema = False
        self.t_eps = t_eps
        self.loss_type = loss_type
        self.num_eval_files = num_eval_files

        self.save_hyperparameters(ignore=['no_wandb'])
        if data_module_cls is not None:
           self.data_module = data_module_cls(**kwargs, gpu=kwargs.get('gpus', 0) > 0)
        else: 
           self.data_module = None
        self.CoVoMix_model = CoVoMix_model
        self.medium_file_save_dir = medium_file_save_dir
        self.lr_scheduler = lr_scheduler
        self.total_epochs = total_epochs
        self.wake_up_epochs = wake_up_epochs
        self.decay_start_epoch = decay_start_epoch

        self.text2semantic_two_output = text2semantic_two_output
        self.num_text_token_ids = num_text_token_ids
        self.t2s_batch_size = t2s_batch_size

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def on_train_epoch_start(self):
        current_epoch = self.current_epoch
        optimizer = self.optimizers()

        if self.lr_scheduler:
            if current_epoch < self.wake_up_epochs:
                lr = self.lr * (current_epoch + 1) / self.wake_up_epochs
            elif current_epoch < self.decay_start_epoch:
                lr = self.lr
            else:
                lr = self.lr * (1 - (current_epoch - self.decay_start_epoch) / (self.total_epochs - self.decay_start_epoch))

            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            self.log('learning_rate', lr, on_epoch=True, prog_bar=True, logger=True)

    def optimizer_step(self, *args, **kwargs):
        # Method overridden so that the EMA params are updated after each optimizer step
        super().optimizer_step(*args, **kwargs)
        self.ema.update(self.parameters())

    # on_load_checkpoint / on_save_checkpoint needed for EMA storing/loading
    def on_load_checkpoint(self, checkpoint):
        ema = checkpoint.get('ema', None)
        if ema is not None:
            self.ema.load_state_dict(checkpoint['ema'])
        else:
            self._error_loading_ema = True
            warnings.warn("EMA state_dict not found in checkpoint!")

    def on_save_checkpoint(self, checkpoint):
        checkpoint['ema'] = self.ema.state_dict()

    def train(self, mode, no_ema=False):
        res = super().train(mode)  # call the standard `train` method with the given mode
        if not self._error_loading_ema:
            if mode == False and not no_ema:
                # eval
                self.ema.store(self.parameters())        # store current params in EMA
                self.ema.copy_to(self.parameters())      # copy EMA parameters over current params for evaluation
            else:
                # train
                if self.ema.collected_params is not None:
                    self.ema.restore(self.parameters())  # restore the EMA weights (if stored)
        return res

    def eval(self, no_ema=False):
        return self.train(False, no_ema=no_ema)

    def _loss(self, err):
        if self.loss_type == 'mse':
            losses = torch.square(err.abs())
        elif self.loss_type == 'mae':
            losses = err.abs()
        # taken from reduce_op function: sum over channels and position and mean over batch dim
        # presumably only important for absolute loss number, not for gradients
        loss = torch.mean(0.5*torch.sum(losses.reshape(losses.shape[0], -1), dim=-1))
        return loss

    def _step(self, batch, batch_idx):
        x, phonemes, mask, cond, prompt_mel = batch
        
        if self.text2semantic:
            loss = self.cfm_wrapper(
                grapheme_token_ids=phonemes,
                semantic_token_ids=x,
                prompt_mel = prompt_mel,
            )
        else: 
            if self.twocondition_oneoutput and cond == None:
                loss = self.cfm_wrapper(
                    x[:,:,-80:],
                    phoneme_ids = phonemes,
                    cond = x[:,:,:-80], 
                    mask = mask,
                )
            elif self.twocondition_oneoutput and cond != None:
                loss = self.cfm_wrapper(
                    x[:,:,-80:],
                    phoneme_ids = phonemes,
                    cond = cond,
                    mask = mask,
                )
            else:    
                loss = self.cfm_wrapper(
                    x,
                    phoneme_ids = phonemes,
                    cond = x, 
                    mask = mask,
                )
        return loss
    

    def training_step(self, batch, batch_idx):
        loss = self._step(batch, batch_idx)
        self.log('train_loss', loss, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, phonemes, mask, cond, prompt_mel = batch
        loss = self._step(batch, batch_idx)
        self.log('valid_loss', loss, on_step=False, on_epoch=True)
            
        if batch_idx == 0 and self.num_eval_files != 0:
            if self.CoVoMix_model == "acoustic":
                if self.twocondition_twooutput :
                    accuracy,l2  = evaluate_acoustic_predictor_hubert_2input_2output(self, self.num_eval_files)
                    self.log('accuracy', accuracy, on_step=False, on_epoch=True)
                    self.log('l2', l2, on_step=False, on_epoch=True)
                elif self.twocondition_oneoutput :
                    accuracy,l2  = evaluate_acoustic_predictor_hubert_2input_1output(self, self.num_eval_files)
                    self.log('accuracy', accuracy, on_step=False, on_epoch=True)
                    self.log('l2', l2, on_step=False, on_epoch=True)
                else:
                    accuracy, l2 = evaluate_acoustic_predictor_hubert(self, self.num_eval_files)
                    self.log('accuracy', accuracy, on_step=False, on_epoch=True)
                    self.log('l2', l2, on_step=False, on_epoch=True)
            elif self.text2semantic or self.CoVoMix_model == "text2semantic":
                accuracy, l2 = evaluate_text2semantic(self, self.num_eval_files)
                self.log('accuracy', accuracy, on_step=False, on_epoch=True)
                self.log('l2', l2, on_step=False, on_epoch=True)
            else: 
                raise error("CoVoMix_model should be acoustic or duration_predictor")
        return loss

    def synthesis_sample(self, phoneme_ids, cond, mask, cond_scale):
        sampled = self.cfm_wrapper.sample(
            phoneme_ids = phoneme_ids,
            cond = cond,
            mask = mask,
            cond_scale = cond_scale
        ) 
        return sampled
    
    def synthesis_sample_e3tts(self, phoneme_ids, cond, mask, cond_scale):
        sampled = self.cfm_wrapper.sample(
            phoneme_ids = phoneme_ids,
            cond = cond,
            mask = mask,
            cond_scale = cond_scale
        ) 
        return sampled
    
    def synthesis_sample_text2semantic(self, grapheme_token_ids, temprature = 1.0,cond_scale = 1.0, beam_search_decode = False, prompt_mel = None):
        sampled = self.cfm_wrapper.sample(
            grapheme_token_ids = grapheme_token_ids,
            temperature=temprature,
            cond_scale=cond_scale,
            beam_search_decode = beam_search_decode,
            prompt_mel=prompt_mel,
        ) 
        return sampled
    
    def synthesis_regression_duration(self, phoneme_ids, cond, mask, cond_scale):
        sampled = self.cfm_wrapper.sample_regression(
            phoneme_ids = phoneme_ids,
            cond = cond,
            mask = mask,
            cond_scale = cond_scale
        ) 
        return sampled
    
    
    def synthesis_flow_duration(self, phoneme_ids, cond, mask, cond_scale):
        sampled = self.cfm_wrapper.sample(
            phoneme_ids = phoneme_ids,
            cond = cond,
            mask = mask,
            cond_scale = cond_scale
        ) 
        return sampled
    
    def to(self, *args, **kwargs):
        """Override PyTorch .to() to also transfer the EMA of the model weights"""
        self.ema.to(*args, **kwargs)
        return super().to(*args, **kwargs)

    

    def train_dataloader(self):
        return self.data_module.train_dataloader()

    def val_dataloader(self):
        return self.data_module.val_dataloader()

    def test_dataloader(self):
        return self.data_module.test_dataloader()

    def setup(self, stage=None):
        return self.data_module.setup(stage=stage)

    def to_audio(self, spec, length=None):
        return self._istft(self._backward_transform(spec), length)

    def _forward_transform(self, spec):
        return self.data_module.spec_fwd(spec)

    def _backward_transform(self, spec):
        return self.data_module.spec_back(spec)

    def _stft(self, sig):
        return self.data_module.stft(sig)

    def _istft(self, spec, length=None):
        return self.data_module.istft(spec, length)

    def generate(self, phonemes, x, mask, **kwargs
    ):
        
        sampled = self.cfm_wrapper.sample(
            phoneme_ids = phonemes,
            cond = x,
            mask = mask
        )
        


