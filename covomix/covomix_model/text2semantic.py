import math
from pathlib import Path
from functools import partial
from random import random

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch import Tensor, nn, einsum, IntTensor, LongTensor
from torch.nn import Module, ModuleList
from torch.utils.data import Dataset

from einops import rearrange, repeat, pack, reduce
from einops.layers.torch import Rearrange

from covomix.covomix_model.rotary_embedding_torch import RotaryEmbedding

from beartype import beartype
from beartype.door import is_bearable
from beartype.typing import Optional, Union, Callable, Literal, Tuple, List

from covomix.covomix_model.attend_t2s import Attend
from covomix.covomix_model.t2s_distributed import all_gather

from tqdm import tqdm
from transformers import BertTokenizer, BertModel


# types

FloatTensor = Union[
    torch.FloatTensor,
    torch.cuda.FloatTensor
]

# helpers

def prob_mask_like(shape, prob, device):
    if prob == 1:
        return torch.ones(shape, device = device, dtype = torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device = device, dtype = torch.bool)
    else:
        return torch.zeros(shape, device = device).float().uniform_(0, 1) < prob


def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def empty(t: Tensor):
    return t.numel() == 0

def l2norm(t):
    return F.normalize(t, dim = -1)

def set_eos_id(t: Tensor, eos_id: int, pad_id: int):
    eos_indices = ((t == pad_id).cumsum(dim = -1) == 0).sum(dim = -1, keepdim = True).long()

    batch_range = torch.arange(t.shape[0], device = t.device, dtype = torch.long)
    batch_range = rearrange(batch_range, '... -> ... 1')

    t = F.pad(t, (0, 1), value = pad_id)
    t[batch_range, eos_indices] = eos_id
    return t

def batch_unique_consecutive(t, pad_value = 0.):
    unique_arr = [torch.unique_consecutive(el) for el in t.unbind(dim = 0)]
    return pad_sequence(unique_arr, batch_first = True, padding_value = pad_value)

def mask_after_eos(target, eos_id, pad_id):
    mask = (target == eos_id).cumsum(dim = -1) > 0
    mask = F.pad(mask, (1, -1), value = False)
    return target.masked_fill(mask, pad_id)

def safe_div(num, den, eps = 1e-10):
    return num / max(den, eps)

def find_first_true_index(bool_tensor, dim = -1):
    return (bool_tensor.cumsum(dim = dim) == 0).sum(dim = dim)

# freezing and unfreezing helpers
def set_requires_grad_(module: Module, requires_grad: bool):
    for p in module.parameters():
        p.requires_grad = requires_grad

def freeze(module: Module):
    set_requires_grad_(module, False)

def unfreeze(module: Module):
    set_requires_grad_(module, True)

# sampling helpers
def eval_decorator(fn):
    def inner(self, *args, **kwargs):
        was_training = self.training
        self.eval()
        out = fn(self, *args, **kwargs)
        self.train(was_training)
        return out
    return inner

def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))

def gumbel_sample(t, temperature = 1., dim = -1):
    return ((t / max(temperature, 1e-10)) + gumbel_noise(t)).argmax(dim = dim)

def argmax_sample(t, dim = -1):
    return t.argmax(dim = dim)

def top_p(logits, thres = 0.9):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = F.pad(cum_probs > thres, (1, -1), value = 0)
    sorted_logits[sorted_indices_to_remove] = float('-inf')
    sorted_logits = sorted_logits.scatter(-1, sorted_indices, sorted_logits)
    return sorted_logits

def top_k(logits, thres = 0.1, k = None):
    if not exists(k):
        k = math.ceil(thres * logits.shape[-1])
    val, ind = torch.topk(logits, k, dim = -1)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(-1, ind, val)
    return probs

# residual wrapper
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

# rmsnorm
class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.normalize(x, dim = -1) * self.scale * self.gamma

# feedforward

class GEGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim = -1)
        return F.gelu(gate) * x

def FeedForward(dim, mult = 4, dropout = 0.):
    dim_inner = int(dim * mult * 2 / 3)
    return nn.Sequential(
        RMSNorm(dim),
        nn.Linear(dim, dim_inner * 2),
        GEGLU(),
        nn.Dropout(dropout),
        nn.Linear(dim_inner, dim)
    )


# attention

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        *,
        dim_head = 64,
        heads = 8,
        kv_heads = None,
        causal = False,
        dim_context = None,
        dropout = 0.,
        rotary_emb: Optional[RotaryEmbedding] = None,
        flash = False,
        add_null_kv = False
    ):
        super().__init__()
        dim_context = default(dim_context, dim)

        self.heads = heads
        self.kv_heads = default(kv_heads, heads)
        assert (self.heads % self.kv_heads) == 0, 'number of key value heads must be divisible by query heads'

        self.scale = dim_head ** -0.5
        dim_query_inner = heads * dim_head
        dim_kv_inner = self.kv_heads * dim_head

        self.rotary_emb = rotary_emb

        self.attend = Attend(
            causal = causal,
            flash = flash,
            dropout = dropout
        )

        self.norm = RMSNorm(dim)
        self.attn_dropout = nn.Dropout(dropout)

        self.to_q = nn.Sequential(
            nn.Linear(dim, dim_query_inner, bias = False),
            Rearrange('b n (h d) -> b h n d', h = self.heads)
        )

        self.to_kv = nn.Sequential(
            nn.Linear(dim_context, dim_kv_inner * 2, bias = False),
            Rearrange('b n (kv h d) -> kv b h n d', kv = 2, h = self.kv_heads)
        )

        self.to_out = nn.Linear(dim_query_inner, dim, bias = False)

        self.add_null_kv = add_null_kv
        if add_null_kv:
            self.null_kv = nn.Parameter(torch.randn(2, self.kv_heads, 1, dim_head))

    def forward(
        self,
        x,
        context = None,
        mask = None,
        cache = None,
        return_cached_key_values = False
    ):
        has_context = exists(context)
        b = x.shape[0]

        x = self.norm(x)

        context = default(context, x)

        q, k, v = (self.to_q(x), *self.to_kv(context))

        if exists(cache):
            ck, cv = cache.unbind(dim = 1)
            k = torch.cat((ck, k), dim = -2)
            v = torch.cat((cv, v), dim = -2)

        new_cache = torch.stack((k, v), dim = 1)

        if exists(self.rotary_emb):
            assert not has_context
            q, k = self.rotary_emb.rotate_queries_with_cached_keys(q, k)

        if self.add_null_kv:
            assert not exists(self.rotary_emb)
            nk, nv = map(lambda t: repeat(t, 'h 1 d -> b h 1 d', b = b), self.null_kv)
            k = torch.cat((nk, k), dim = -2)
            v = torch.cat((nv, v), dim = -2)

            if exists(mask):
                mask = F.pad(mask, (1, 0), value = True)

        out = self.attend(q, k, v, mask = mask)

        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)

        if not return_cached_key_values:
            return out

        return out, new_cache

# transformer

class Transformer(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth,
        dim_context = None,
        dim_head = 64,
        heads = 8,
        kv_heads = None,
        causal = False,
        attn_dropout = 0.,
        ff_mult = 4,
        ff_dropout = 0.,
        cross_attend = False,
        attn_flash = False
    ):
        super().__init__()

        rotary_emb = RotaryEmbedding(dim_head)

        self.layers = nn.ModuleList([])
        
        dim_context = default(dim_context, dim)

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim = dim, causal = causal, dim_head = dim_head, heads = heads, kv_heads = kv_heads, dropout = attn_dropout, rotary_emb = rotary_emb, flash = attn_flash),
                Attention(dim = dim, dim_head = dim_head, heads = heads, dropout = attn_dropout, flash = attn_flash, add_null_kv = True, dim_context = dim_context) if cross_attend else None,
                FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout)
            ]))

        self.final_norm = RMSNorm(dim)

    def forward(
        self,
        x,
        mask = None,
        context = None,
        context_mask = None,
        cache = None,
        return_cache = False,
        return_hiddens = False,
        early_exit_at_layer = None,
        seq_start_pos = None
    ):
        has_context = exists(context)

        if exists(seq_start_pos):
            assert not exists(mask)
            seq_len = x.shape[-2]
            seq_arange = torch.arange(seq_len, device = x.device, dtype = torch.long)
            mask = seq_arange >= seq_start_pos[..., None]

        if exists(cache):
            #print("cache",cache.shape, "x",x.shape)
            cached_length, seq_len = cache.shape[-2], x.shape[-2]
            assert seq_len > cached_length
            x = x[:, cached_length:]

        new_cache = []
        hiddens = []

        if exists(cache):
            iter_cache = iter(cache.unbind(dim = 1))
        else:
            iter_cache = iter([])

        for ind, (self_attn, maybe_cross_attn, ff) in enumerate(self.layers):
            layer = ind + 1

            residual = x
            attn_out, key_values = self_attn(x, mask = mask, cache = next(iter_cache, None), return_cached_key_values = True)
            x = attn_out + residual

            new_cache.append(key_values)

            if exists(maybe_cross_attn):
                assert has_context
                x = maybe_cross_attn(x, context = context, mask = context_mask) + x

            x = ff(x) + x
            hiddens.append(x)

            if exists(early_exit_at_layer) and early_exit_at_layer == layer:
                break

        if exists(early_exit_at_layer):
            if return_cache:
                return x, torch.stack(new_cache, dim = 1)
            return x

        out = self.final_norm(x)

        if return_hiddens:
            assert not return_cache
            return out, torch.stack(hiddens)

        if not return_cache:
            return out

        return out, torch.stack(new_cache, dim = 1)

# class

SpeechOrTextLiteral = Union[
    Literal['speech'],
    Literal['text']
]


class empty_identity_encoder(Module):
    def __init__(self):
        super().__init__()

    def forward(self, 
        x,
        mask = None,
        context = None,
        context_mask = None,
        cache = None,
        return_cache = False,
        return_hiddens = False,
        early_exit_at_layer = None,
        seq_start_pos = None):
        
        return x


class TextToSemantic(Module):
    @beartype
    def __init__(
        self,
        dim,
        *,
        source_depth,
        target_depth,
        num_text_token_ids = None,
        tokenizer_encode: Optional[Callable] = None,
        use_openai_tokenizer = False,
        wav2vec = None,
        num_semantic_token_ids = None,
        dim_head = 64,
        heads = 8,
        target_kv_heads = None,  # for grouped query attention, saving memory on decoder inference
        attn_dropout = 0.,
        ff_mult = 4,
        ff_dropout = 0.,
        semantic_pad_id = -1,
        text_pad_id = 0,
        autoset_semantic_eos_id = True,
        autoset_text_eos_id = True,
        attn_flash = False,
        cond_drop_prob = 0.,
        target_early_exit_layer = None,
        detach_early_exit_embed = False,
        align_reg_loss_weight = 0.1,
        align_reg_use_logsumexp_pool = True,
        align_reg_logsumexp_pool_temp = 0.1,
        text2semantic_bert_encoder = False,
        text2semantic_t5_encoder = False,
        no_source_transformer = False,
        two_output = False,
        two_input = False,
        target_transformer_dim = None,
        classifier_free_guidance = False,
    ):
        super().__init__()
        if text2semantic_bert_encoder:
            dim = 768
        elif text2semantic_t5_encoder:
            dim = 512
        self.dim = dim
        self.wav2vec = wav2vec
        if target_transformer_dim == None:
            self.target_transformer_dim = self.dim
        else: 
            self.target_transformer_dim = target_transformer_dim
        #print("text2semantic model: self.target_transformer_dim=",target_transformer_dim)

        if exists(self.wav2vec):
            freeze(self.wav2vec)

        self.tokenizer_encode = tokenizer_encode

        if use_openai_tokenizer:
            assert not exists(tokenizer_encode)
            assert not exists(num_text_token_ids)
            self.tokenizer_encode = tokenizer.tokenize
            num_text_token_ids = tokenizer.vocab_size
        else:
            assert exists(num_text_token_ids), 'num_text_token_ids not specified'

        num_semantic_token_ids = wav2vec.codebook_size if exists(wav2vec) else num_semantic_token_ids
        assert exists(num_semantic_token_ids), 'you need to either pass in a wav2vec model from audiolm-pytorch, or specify the number of semantic token ids with num_semantic_token_ids'

        self.num_semantic_token_ids = num_semantic_token_ids
        self.num_text_token_ids = num_text_token_ids

        # padding id, for deriving attention mask automatically if not passed in

        self.semantic_pad_id = semantic_pad_id
        self.text_pad_id = text_pad_id

        self.pad_id = dict(
            speech = semantic_pad_id,
            text = text_pad_id
        )

        # eos id

        self.autoset_eos_id = dict(
            speech = autoset_semantic_eos_id,
            text = autoset_text_eos_id
        )

        self.eos_id = dict(
            speech = num_semantic_token_ids,
            text = num_text_token_ids
        )

                

        # embedding
        num_semantic_token_ids_with_eos = num_semantic_token_ids + int(autoset_semantic_eos_id)
        num_text_token_ids_with_eos = num_text_token_ids + int(autoset_text_eos_id)
        self.text2semantic_bert_encoder = text2semantic_bert_encoder
        self.text2semantic_t5_encoder = text2semantic_t5_encoder
        if text2semantic_bert_encoder or text2semantic_t5_encoder:
            # self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            # self.bert_tokenizer.add_tokens(['[laughter]'])
            # self.bert_model = BertModel.from_pretrained('bert-base-uncased')
            # self.bert_model.resize_token_embeddings(len( self.bert_tokenizer))
            text_token_emb = nn.Identity()
        elif two_input:
            text_token_emb = nn.Embedding(num_text_token_ids_with_eos, dim//2)
        else:
            text_token_emb = nn.Embedding(num_text_token_ids_with_eos, dim)

        if not two_output:
            semantic_token_emb = nn.Embedding(num_semantic_token_ids_with_eos, self.target_transformer_dim)
        else: 
            semantic_token_emb = nn.Embedding(num_semantic_token_ids_with_eos, self.target_transformer_dim//2)

        self.semantic_token_emb = semantic_token_emb

        self.token_emb = nn.ModuleDict(dict(
            speech = semantic_token_emb,
            text = text_token_emb
        ))

        # respective start tokens
        
        self.start_token = nn.ParameterDict(dict(
            speech = nn.Parameter(torch.randn(target_transformer_dim)),
            text = nn.Parameter(torch.randn(dim))
        ))
        

        # projection to logits

        if not two_output:
            to_semantic_logit = nn.Linear(self.target_transformer_dim, num_semantic_token_ids, bias = False)
        else: 
            to_semantic_logit = nn.Linear(self.target_transformer_dim//2, num_semantic_token_ids, bias = False)
        
        if not two_input:
            to_text_logit = nn.Linear(dim, num_text_token_ids, bias = False)
        else: 
            to_text_logit = nn.Linear(dim//2, num_text_token_ids, bias = False)

        to_semantic_logit.weight = semantic_token_emb.weight
        if not text2semantic_bert_encoder and not text2semantic_t5_encoder:
            to_text_logit.weight = text_token_emb.weight

        self.to_logits = nn.ModuleDict(dict(
            speech = to_semantic_logit,
            text = to_text_logit
        ))

        # source and target attention layers

        self.no_source_transformer = no_source_transformer
        if self.no_source_transformer:
            self.source_transformer = empty_identity_encoder()
        else:
            self.source_transformer = Transformer(
                dim = dim,
                dim_head = dim_head,
                heads = heads,
                depth = source_depth,
                attn_dropout = attn_dropout,
                ff_mult = ff_mult,
                ff_dropout = ff_dropout,
                causal = False,
                attn_flash = attn_flash
            )


        self.target_transformer = Transformer(
            dim = self.target_transformer_dim,
            dim_head = dim_head,
            heads = heads,
            kv_heads = target_kv_heads,
            depth = target_depth,
            attn_dropout = attn_dropout,
            ff_mult = ff_mult,
            ff_dropout = ff_dropout,
            causal = True,
            cross_attend = True,
            attn_flash = attn_flash,
            dim_context=dim,
        )

        # classifier free guidance - prob of dropping condition

        assert 0 <= cond_drop_prob < 1
        self.cond_drop_prob = cond_drop_prob
        self.classifier_free_guidance = classifier_free_guidance
        if classifier_free_guidance:
            self.null_source_embedding = nn.Parameter(torch.zeros(dim))

        self.align_reg_loss_weight = align_reg_loss_weight # lambda for weight of regularization loss in https://arxiv.org/abs/2309.08773
        self.align_reg_use_logsumexp_pool = align_reg_use_logsumexp_pool
        self.align_reg_logsumexp_pool_temp = align_reg_logsumexp_pool_temp

        # for speculative decoding, to speed up text-to-speech decoding and make real-time TTS approach more feasible with spear-tts
        # using early exist strategy so one can train just the same model

        self.target_has_early_exit = exists(target_early_exit_layer)
        self.early_exit_layer = target_early_exit_layer

        if self.target_has_early_exit:
            assert 0 < target_early_exit_layer <= target_depth, f'the early exit layer for the speech transformer must be between 1 and {target_depth}'

            self.detach_early_exit_embed = detach_early_exit_embed

            self.to_early_exit_semantic_logits = nn.Sequential(
                Residual(FeedForward(dim)),
                RMSNorm(dim),
                nn.Linear(dim, num_semantic_token_ids_with_eos, bias = False)
            )
            
        self.two_output = two_output
        self.two_input = two_input
        #print("two_output",two_output)
        
              
    @property
    def device(self):
        return next(self.parameters()).device


    def unfreeze_all(self):
        unfreeze(self)

    def freeze_encoder(self):
        freeze(self.source_transformer)

    def freeze_encoder_below_layer(self, layer: int):
        """
        for the final training of text-to-semantic on pseudo-labelled dataset
        they freeze the encoder part way up to a certain layer
        """
        unfreeze(self.source_transformer)

        for ind, module in enumerate(self.source_transformer.layers):
            current_layer = ind + 1

            if current_layer <= layer:
                freeze(module)

    def freeze_decoder(self):
        freeze(self.target_transformer)

    def freeze_speech_emb(self):
        freeze(self.token_emb['speech'])
        self.start_token['speech'].requires_grad = False

    def freeze_text_emb(self):
        freeze(self.token_emb['text'])
        self.start_token['text'].requires_grad = False

    # sampling function

    @torch.no_grad()
    @eval_decorator
    @beartype
    def generate(
        self,
        source,
        *,
        source_type: SpeechOrTextLiteral,
        target_type: SpeechOrTextLiteral,
        temperature = 1.,
        filter_logits_fn = top_k,
        filter_fn_kwargs: dict = dict(),
        source_mask: Optional[Tensor] = None,
        max_length = 2048,
        beam_search_decode = False,
        spec_decode = False,
        spec_decode_gamma = 5,
        spec_decode_lenience = 1.,
        beam_size = 10, # default 4
        return_source = False,
        return_target_mask = False,
        cond_scale = 1.,
        prompt_mel = None,
    ):
        assert cond_scale >= 1.
        assert not (cond_scale > 1 and self.cond_drop_prob == 0), 'you need to train with conditional drop probability greater than 0 to use classifier free guidance at inference, and it needs to be the right source to target pair'
        
        if self.two_input:
            source2 = source[:,:,1].squeeze(dim=-1)
            source = source[:,:,0].squeeze(dim=-1)
        
        if is_bearable(source, FloatTensor) and source_type == 'speech':
            assert exists(self.wav2vec), 'wav2vec should be passed in, if generating with source as raw soundwave'

            with torch.no_grad():
                self.wav2vec.eval()
                source = source.to(self.device)
                source = self.wav2vec(source)
        
        if self.text2semantic_bert_encoder or self.text2semantic_t5_encoder:
            source_emb, source_mask = source
            source_emb = source_emb.to(self.device)
            source_mask = source_mask.to(self.device)
        
        
        if not  self.text2semantic_bert_encoder and not self.text2semantic_t5_encoder:
            if source.shape[-1] > max_length :
                source_token_emb = source[:,:,:max_length]
        source_token_emb = self.token_emb[source_type]
        source_pad_id = self.pad_id[source_type]

        # all target modules and parameters

        target_token_emb = self.token_emb[target_type]
        target_start_token = self.start_token[target_type]
        target_to_logit = self.to_logits[target_type]
        target_pad_id = self.pad_id[target_type]
        target_eos_id = self.eos_id[target_type]

        # auto set eos i
        if self.autoset_eos_id[source_type] and not self.text2semantic_bert_encoder and not self.text2semantic_t5_encoder:
            source_eos_id = self.eos_id[source_type]
            source = set_eos_id(source, source_eos_id, pad_id = source_pad_id)

        # if source mask is not passed in
        # automatically derive by the padding id of the modality

        if not exists(source_mask) and source.dtype == torch.long:
            source_mask = source != source_pad_id
        
        # source embedding
        if not self.text2semantic_bert_encoder and not self.text2semantic_t5_encoder:
            if not self.two_input:
                source_emb = source_token_emb(source)
            else: 
                source_emb = source_token_emb(source)
                source_emb2 = source_token_emb(source2)
                source_emb = torch.cat((source_emb,source_emb2),dim=-1)
        batch = source_emb.shape[0]
        
        source_emb = self.source_transformer(source_emb, mask = source_mask)
        # decode target

        target = torch.empty((batch, 0), dtype = torch.long, device = self.device)
        target2 = torch.empty((batch, 0), dtype = torch.long, device = self.device)
        start_token = repeat(target_start_token, 'd -> b 1 d', b = batch)

        # loop to decode

        assert not (beam_search_decode and spec_decode), 'you must choose either beam decode or speculative decoding, but not both'
        if not beam_search_decode and not spec_decode:
            cache = None
            null_cache = None

            for _ in range(max_length):
                if not self.two_output:
                    target_emb = target_token_emb(target)
                else: 
                    target_emb = torch.cat((target_token_emb(target),target_token_emb(target2)),dim=-1)
                target_emb = torch.cat((start_token, target_emb), dim = 1)

                # target attention

                attended_target_emb, cache = self.target_transformer(target_emb, context = source_emb, context_mask = source_mask, cache = cache, return_cache = True)
                
                # decoder logits
                if  not self.two_output:
                    logits = target_to_logit(attended_target_emb)
                    logits = logits[:, -1]
                else :
                    half = attended_target_emb.shape[-1] // 2
                    attended_target_emb1 = attended_target_emb[:,:,:half]
                    attended_target_emb2 = attended_target_emb[:,:,half:]
                    
                    logits = target_to_logit(attended_target_emb1)
                    logits = logits[:, -1]
                    logits2 = target_to_logit(attended_target_emb2)
                    logits2 = logits2[:, -1]
                
                # handle classifier free guidance

                if cond_scale > 1.:
                    null_source_mask = source_mask.float().zero_().bool()

                    attended_null_target_emb, null_cache = self.target_transformer(target_emb, context = source_emb, context_mask = null_source_mask, cache = null_cache, return_cache = True)

                    null_logits = target_to_logit(attended_null_target_emb)
                    null_logits = null_logits[:, -1]
                    
                    

                    logits = null_logits + (logits - null_logits) * cond_scale
                    if self.two_output:
                        logits2 = null_logits + (logits2 - null_logits) * cond_scale
                        
                # filter logits

                logits = filter_logits_fn(logits, **filter_fn_kwargs)
                

                sampled = gumbel_sample(logits, temperature = temperature)
                target, _ = pack((target, sampled), 'b *')
                                
                if self.two_output:
                    logits2 = filter_logits_fn(logits2, **filter_fn_kwargs)
                    sampled2 = gumbel_sample(logits2, temperature = temperature)
                    target2, _ = pack((target2, sampled2), 'b *')
                    
                if not self.autoset_eos_id[target_type]:
                    continue

                is_eos = target == target_eos_id
                all_eos = is_eos.any(dim = -1).all()

                
                if not self.two_output:
                    if not all_eos:
                        continue
                target = mask_after_eos(target, target_eos_id, target_pad_id)

                if self.two_output:
                    is_eos2 = target2 == target_eos_id
                    all_eos2 = is_eos2.any(dim = -1).all()

                    if (not all_eos2) and (not all_eos):
                        continue
                    target2 = mask_after_eos(target2, target_eos_id, target_pad_id)                
                break
            
        # whether to return the target mask
        # for variable lengthed generation output
        
        if self.two_output:
            target = torch.cat((target,target2),dim=1)
        
        if return_target_mask:
            target_mask = target != target_pad_id

        # 4 different types of return cases

        if not return_source:
            if not return_target_mask:
                return target

            return target, target_mask

        if not return_target_mask:
            return source, target

        return source, target, target_mask

    @beartype
    def forward(
        self,
        source,
        target,
        *,
        source_type: SpeechOrTextLiteral,
        target_type: SpeechOrTextLiteral,
        source_mask: Optional[Tensor] = None,
        target_mask: Optional[Tensor] = None,
        return_loss = False,
        return_logits = False,
        cond_drop_prob: Optional[float] = None,
        should_sim_regularize = True,
        return_early_exit_loss = False,
        prompt_mel = None,
    ):  
        
        if self.two_output: 
            target2 = target[:,:,1].squeeze(dim=-1)
            target = target[:,:,0].squeeze(dim=-1)
        else: 
            if target.dim() == 3:
                target = target.squeeze(dim=-1) 

        if self.two_input:
            source2 = source[:,:,1].squeeze(dim=-1)
            source = source[:,:,0].squeeze(dim=-1)
            

        if self.text2semantic_bert_encoder or self.text2semantic_t5_encoder:
            source_emb, source_mask = source
            source_emb = source_emb.to(self.device)
            source_mask = source_mask.to(self.device)
        
        cond_drop_prob = default(cond_drop_prob, self.cond_drop_prob)
        drop_cond = cond_drop_prob > 0 and random() < cond_drop_prob

        if is_bearable(source, FloatTensor) and source_type == 'speech':
            assert exists(self.wav2vec), 'wav2vec should be passed in, if generating with source as raw soundwave'

            with torch.no_grad():
                self.wav2vec.eval()
                source = self.wav2vec(source)

        if is_bearable(target, List[str]):
            assert exists(self.tokenizer_encode)
            target = self.tokenizer_encode(target)
            target = target.to(self.device)

        # assert source.shape[0] == target.shape[0]
        batch = target.shape[0]
        
        source_token_emb = self.token_emb[source_type]
        source_pad_id = self.pad_id[source_type]

        # all target modules and parameters

        target_token_emb = self.token_emb[target_type]
        target_start_token = self.start_token[target_type]
        target_to_logit = self.to_logits[target_type]
        target_pad_id = self.pad_id[target_type]

        # auto set eos id

        if self.autoset_eos_id[source_type] and not self.text2semantic_bert_encoder and not self.text2semantic_t5_encoder:
            source_eos_id = self.eos_id[source_type]
            source = set_eos_id(source, source_eos_id, pad_id = source_pad_id)
            if self.two_input:
                source2 = set_eos_id(source2, source_eos_id, pad_id = source_pad_id)

        if self.autoset_eos_id[target_type] and return_loss:
            target_eos_id = self.eos_id[target_type]
            target = set_eos_id(target, target_eos_id, pad_id = target_pad_id)
            
            if self.two_output:
                target2 = set_eos_id(target2, target_eos_id, pad_id = target_pad_id)

        # if source/target mask is not passed in
        # automatically derive by the padding id of the modality

        if not exists(source_mask) and source.dtype == torch.long and not self.text2semantic_bert_encoder and not self.text2semantic_t5_encoder:
            source_mask = source != source_pad_id

        if not exists(target_mask) and target.dtype == torch.long:
            target_mask = target != target_pad_id

            # attend to bos
            target_mask = F.pad(target_mask, (1, 0), value = True)

        # embedding
        if not self.text2semantic_bert_encoder and not self.text2semantic_t5_encoder:
            if not self.two_input:
                source_emb = source_token_emb(source)
            else: 
                source_emb = source_token_emb(source)
                source_emb2 = source_token_emb(source2)
                source_emb = torch.cat((source_emb,source_emb2),dim=-1)

        
        if not self.two_output:
            target_emb = target_token_emb(target)
        else: 
            target_emb = target_token_emb(target)
            target2_emb = target_token_emb(target2)
            target_emb = torch.cat((target_emb,target2_emb),dim=-1)
        start_token = repeat(target_start_token, 'd -> b 1 d', b = batch)

        target_emb = torch.cat((start_token, target_emb), dim = 1)
        
        # source attention
        source_emb = self.source_transformer(source_emb, source_mask)

        assert source_emb.shape[0] == target_emb.shape[0], 'source and target token embeddings must have the same batch size'


        # whether to drop condition, for CFG

        context_mask = source_mask
        if self.classifier_free_guidance and random() < 0.1:
            cond_drop_mask = prob_mask_like(source_emb.shape[:1], cond_drop_prob, self.device)
            source_emb = torch.where(
                rearrange(cond_drop_mask, '... -> ... 1 1'),
                self.null_source_embedding,
                source_emb
            )

        # target attention
        target_emb, target_hiddens = self.target_transformer(
            target_emb,
            mask = target_mask,
            context = source_emb,
            context_mask = context_mask,
            return_hiddens = True
        )




        # decoder logits
        
        if not self.two_output:
            logits = target_to_logit(target_emb)
        else: 
            half = target_emb.shape[-1]//2
            target_embedding1 = target_emb[:,:,:half]
            target_embedding2 = target_emb[:,:,half:]
            logits = target_to_logit(target_embedding1)
            logits2 = target_to_logit(target_embedding2)

        if not return_loss:
            return logits

        assert (self.training and not empty(target)) or not self.training

        if not self.two_output:
            logits = rearrange(logits[:, :-1], 'b n c -> b c n')
            loss = F.cross_entropy(
                logits,
                target,
                ignore_index = target_pad_id
            )
            
        else:
            logits = rearrange(logits[:, :-1], 'b n c -> b c n')
            logits2 = rearrange(logits2[:, :-1], 'b n c -> b c n')
            loss = F.cross_entropy(
                logits,
                target,
                ignore_index = target_pad_id
            ) + F.cross_entropy(logits2,
                target2,
                ignore_index = target_pad_id
            )
            
               
        if return_early_exit_loss:
            assert self.target_has_early_exit, 'you need to set the `target_early_exit_layer` in order to train a predictor on an earlier hidden dimension for speculative decoding'
            assert source_type == 'text' and target_type == 'speech'

            early_layer_index = self.early_exit_layer - 1
            early_embed = target_hiddens[early_layer_index]

            if self.detach_early_exit_embed:
                # a way to train the early exit head without affecting the main loss
                early_embed = early_embed.detach()

            early_exit_logits = self.to_early_exit_semantic_logits(early_embed)
            early_exit_logits = rearrange(early_exit_logits[:, :-1], 'b n c -> b c n')

            early_exit_loss = F.cross_entropy(
                early_exit_logits,
                target,
                ignore_index = target_pad_id
            )

            loss = loss + early_exit_loss

        if should_sim_regularize and source_type != target_type and drop_cond and self.align_reg_loss_weight > 0:
            # regularizer proposed in https://arxiv.org/abs/2309.08773, alternative to contrastive loss when unconditional
            # supposedly fixes CFG for encoder / decoder transformers

            source_emb, batch_sizes = all_gather(source_emb, 0, None)
            target_emb, _           = all_gather(target_emb, 0, batch_sizes)

            mask_value = -torch.finfo(source_emb.dtype).max

            if exists(source_mask):
                source_emb = source_emb.masked_fill(~source_mask[..., None], mask_value)

            if exists(target_mask):
                target_emb = target_emb.masked_fill(~target_mask[..., None], mask_value)

            # they found that max pool worked best
            # also offer logsumexp pool (smooth max)

            batch, device = source_emb.shape[0], source_emb.device

            if self.align_reg_use_logsumexp_pool:
                temp = self.align_reg_logsumexp_pool_temp
                source_emb, target_emb = map(lambda t: t / temp, (source_emb, target_emb))
                source_emb = reduce(source_emb, 'b n d -> b d', torch.logsumexp)
                target_emb = reduce(target_emb, 'b n d -> b d', torch.logsumexp)
                source_emb, target_emb = map(lambda t: t * temp, (source_emb, target_emb))
            else:
                source_emb = reduce(source_emb, 'b n d -> b d', 'max')
                target_emb = reduce(target_emb, 'b n d -> b d', 'max')

            source_emb, target_emb = map(l2norm, (source_emb, target_emb))

            source_sim, target_sim = map(lambda t: einsum('i d, j d -> i j', t, t), (source_emb, target_emb))
            diag_mask = torch.eye(batch, device = device, dtype = torch.bool)

            align_reg_loss = F.mse_loss(source_sim[~diag_mask], target_sim[~diag_mask])
            loss = loss + align_reg_loss * self.align_reg_loss_weight

        if not return_logits:
            return loss

        return loss, logits

# pretraining modules

def get_mask_subset_prob(mask, prob, min_mask = 0):
    batch, seq, device = *mask.shape, mask.device
    num_to_mask = (mask.sum(dim = -1, keepdim = True) * prob).clamp(min = min_mask)
    logits = torch.rand((batch, seq), device = device)
    logits = logits.masked_fill(~mask, -1)

    randperm = logits.argsort(dim = -1).float()

    num_padding = (~mask).sum(dim = -1, keepdim = True)
    randperm -= num_padding

    subset_mask = randperm < num_to_mask
    subset_mask.masked_fill_(~mask, False)
    return subset_mask

class SpeechSpeechPretrainWrapper(nn.Module):
    @beartype
    def __init__(
        self,
        model: TextToSemantic,
        wav2vec = None,
        deletion_prob: float = 0.6,
        reconstruct_seq: bool = False,
        mask_id = None
    ):
        super().__init__()

        self.model = model
        self.wav2vec = default(wav2vec, model.wav2vec)

        self.deletion_prob = deletion_prob
        self.reconstruct_seq = reconstruct_seq # whether to reconstruct the entire sequence, or just output the deleted ones in order
        self.mask_id = mask_id

    def forward(
        self,
        x,
        return_early_exit_loss = False
    ):
        is_raw_audio = x.dtype == torch.float

        if is_raw_audio:
            assert exists(self.wav2vec)
            
            with torch.no_grad():
                self.wav2vec.eval()
                x = self.wav2vec(x, flatten = False)

        batch = x.shape[0]

        mask = torch.ones_like(x, dtype = torch.bool, device = self.model.device)

        if exists(self.mask_id):
            assert self.reconstruct_seq, 'reconstruct_seq must be true if mask id is provided'
            
            mask = mask.masked_fill(x == self.model.semantic_pad_id, False)
            delete_mask = get_mask_subset_prob(mask, self.deletion_prob)

            source = x.masked_fill(delete_mask, self.mask_id)
        else:
            delete_mask = get_mask_subset_prob(mask, self.deletion_prob)

            source = rearrange(x[~delete_mask], '(b n) -> b n', b = batch)

        if self.reconstruct_seq:
            target = x
        else:
            target = rearrange(x[delete_mask], '(b n) -> b n', b = batch)

        loss, logits = self.model(
            source, target,
            source_type = 'text',
            target_type = 'speech',
            return_loss = True,
            return_logits = True,
            return_early_exit_loss = return_early_exit_loss,
        )

        return loss, logits

# wrapper for backtranslation task

class SemanticToTextWrapper(nn.Module):
    @beartype
    def __init__(
        self,
        model: TextToSemantic
    ):
        super().__init__()

        self.model = model

    def forward(
        self,
        semantic_token_ids,
        grapheme_token_ids,
    ):
        source = semantic_token_ids
        target = grapheme_token_ids

        loss, logits = self.model(
            source, target,
            source_type = 'speech',
            target_type = 'text',
            return_loss = True,
            return_logits = True
        )

        return loss, logits

# wrapper for text to semantic task

class TextToSemanticWrapper(nn.Module):
    @beartype
    def __init__(
        self,
        model: TextToSemantic
    ):
        super().__init__()

        self.model = model

    def forward(
        self,
        grapheme_token_ids,
        semantic_token_ids,
        return_early_exit_loss = False,
        prompt_mel = None,
    ):
        source = grapheme_token_ids
        target = semantic_token_ids

        loss, logits = self.model(
            source, target,
            source_type = 'text',
            target_type = 'speech',
            return_loss = True,
            return_logits = True,
            return_early_exit_loss = return_early_exit_loss,
            prompt_mel = prompt_mel,
        )

        return loss
    
    def sample(self, grapheme_token_ids, temperature = 1., cond_scale = 1., beam_search_decode = False, prompt_mel = None):
        source = grapheme_token_ids
        target, target_mask = self.model.generate(source, 
                      source_type = 'text',
                      target_type = 'speech',
                      return_target_mask = True,
                      return_source = False,
                      temperature = temperature,
                      beam_search_decode = beam_search_decode,
                      cond_scale = cond_scale,
                      prompt_mel = prompt_mel)
        # return non-masked part of target
        target = target[target_mask]
        
        return target
        
