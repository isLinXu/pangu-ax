'''
    code by TaeHwan Jung(@graykode)
    Original Paper and repository here : https://github.com/openai/gpt-2
    GPT2 Pytorch Model : https://github.com/huggingface/pytorch-pretrained-BERT
'''
import copy
import torch
import math
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch import functional as F

def gelu(x):
    pi = torch.div(2,torch.FloatTensor([math.pi]).to(x.device))
    return 0.5 * x * (1 + torch.tanh(
        torch.sqrt(pi) * (x + 0.044715 * torch.pow(x, 3))))

class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias

class Conv1D(nn.Module):
    def __init__(self, nf, nx, transpose=False):
        super(Conv1D, self).__init__()
        self.transpose = transpose
        self.nf = nf
        w = torch.empty(nx, nf)
        nn.init.normal_(w, std=0.02)
        self.weight = Parameter(w.t()) if transpose else Parameter(w)
        self.bias = Parameter(torch.zeros(nf))

    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)),
                        self.weight.t() if self.transpose else self.weight)
        x = x.view(size_out)
        return x

class Attention(nn.Module):
    def __init__(self, nx, n_ctx, config, scale=False):
        super(Attention, self).__init__()
        n_state = nx  # in Attention: n_state=768 (nx=n_embd)
        # [switch nx => n_state from Block to Attention to keep identical to TF implem]
        assert n_state % config.n_head == 0
        self.register_buffer("bias", torch.tril(torch.ones(n_ctx, n_ctx)).view(1, 1, n_ctx, n_ctx))
        self.n_head = config.n_head
        self.split_size = n_state
        self.scale = scale
        self.query = Conv1D(nx, nx, transpose=True)
        self.key = Conv1D(nx, nx, transpose=True)
        self.value = Conv1D(nx, nx, transpose=True)
        self.dense = Conv1D(n_state, nx, transpose=True)
        self.softmax = nn.Softmax(dim=-1)

    def _attn(self, q, k, v):
        w = torch.matmul(q, k)
        if self.scale:
            w = torch.div(w,torch.sqrt(
                torch.FloatTensor([float(v.size(-1))]).to(v.device)))
        nd, ns = w.size(-2), w.size(-1)
        b = self.bias[:, :, ns-nd:ns, :ns]
        w = w * b - 1e10 * (1 - b)
        w = self.softmax(w)

        return torch.matmul(w, v)

    def merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
        return x.view(new_x_shape)  # in Tensorflow implem: fct merge_states

    def split_heads(self, x, k: bool=False):
        new_x_shape = x.size()[:-1] + (self.n_head, x.size(-1) // self.n_head)
        x = x.view(new_x_shape)  # in Tensorflow implem: fct split_states
        if k:
            return x.permute(0, 2, 3, 1)  # (batch, head, head_features, seq_length)
        else:
            return x.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)

    def forward(self, x, layer_past,query):
        if query is not None:
            query = self.query(query)
        else:
            query = self.query(x)
        key = self.key(x)
        value = self.value(x)

        query = self.split_heads(query)
        key = self.split_heads(key, k=True)
        value = self.split_heads(value)
        if layer_past is not None:
            past_key, past_value = layer_past[0].transpose(-2, -1), layer_past[1]  # transpose back cf below
            key = torch.cat((past_key, key), dim=-1)
            value = torch.cat((past_value, value), dim=-2)
        present = torch.stack((key.transpose(-2, -1), value))  # transpose to have same shapes for stacking
        a = self._attn(query, key, value)
        a = self.merge_heads(a)
        a = self.dense(a)
        return a, present

class MLP(nn.Module):
    def __init__(self, n_state, config):  # in MLP: n_state=3072 (4 * n_embd)
        super(MLP, self).__init__()
        nx = config.n_embd
        self.dense_h_to_4h = Conv1D(n_state, nx, transpose=True)
        self.dense_4h_to_h = Conv1D(nx, n_state, transpose=True)
        self.act = gelu

    def forward(self, x):
        h = self.act(self.dense_h_to_4h(x))
        h2 = self.dense_4h_to_h(h)
        return h2

class Att_block(nn.Module):
    def __init__(self, n_ctx, config, scale=False):
        super(Att_block, self).__init__()
        nx = config.n_embd
        self.input_layernorm = LayerNorm(nx, eps=config.layer_norm_epsilon)
        self.attention = Attention(nx, n_ctx, config, scale)
        self.post_attention_layernorm = LayerNorm(nx, eps=config.layer_norm_epsilon)
        self.mlp = MLP(4 * nx, config)

    def forward(self, x, layer_past, query):
        x_ln = self.input_layernorm(x)
        a, present = self.attention(x_ln, layer_past=layer_past, query=query)
        x = x + a
        m = self.mlp(self.post_attention_layernorm(x))
        x = x + m
        return x, present


class Embendding(nn.Module):
    def __init__(self, config):
        super(Embendding, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embeddings = nn.Embedding(config.n_positions, config.n_embd)
        self.top_query_embeddings = nn.Embedding(config.n_positions, config.n_embd)

    def forward(self, input_ids, position_ids):
        inputs_embeds = self.word_embeddings(input_ids)
        position_embeds = self.position_embeddings(position_ids)
        top_query_embeddins = self.top_query_embeddings(position_ids)
        return inputs_embeds, position_embeds, top_query_embeddins


class Transformer(nn.Module):
    def __init__(self, config):
        super(Transformer, self).__init__()
        self.n_layer = config.n_layer
        self.n_embd = config.n_embd
        self.n_vocab = config.vocab_size
        block = Att_block(config.n_ctx, config, scale=True)
        self.layers = nn.ModuleList([copy.deepcopy(block) for _ in range(config.n_layer-1)])
        self.topQueryLayer = Att_block(config.n_ctx, config, scale=True)
        self.final_layernorm = LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)

    def set_embeddings_weights(self, model_embeddings_weights):
        embed_shape = model_embeddings_weights.shape
        self.decoder = nn.Linear(embed_shape[1], embed_shape[0], bias=False)
        self.decoder.weight = model_embeddings_weights  # Tied weights

    def forward(self, input_ids, position_ids, token_type_ids, past, embendding):
        if past is None:
            past_length = 0
            past = [None] * self.n_layer
        else:
            past_length = past[0][0].size(-2)
        if position_ids is None:
            position_ids = torch.arange(past_length, input_ids.size(-1) + past_length, dtype=torch.long,
                                        device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_ids.size(-1))
        position_ids = position_ids.view(-1, position_ids.size(-1))

        inputs_embeds, position_embeds, top_query_embeddins = embendding(input_ids, position_ids)
        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1))
            token_type_embeds = self.word_embenddings(token_type_ids)
        else:
            token_type_embeds = 0
        hidden_states = inputs_embeds + position_embeds + token_type_embeds
        presents = []
        for i,block in enumerate(self.layers):
            layer_past = past[i]
            hidden_states, present = block(hidden_states, layer_past, query=None)
            presents.append(present)

        hidden_states = self.final_layernorm(hidden_states)

        layer_past = past[i+1]
        hidden_states, present = self.topQueryLayer(hidden_states, layer_past, query=top_query_embeddins)
        presents.append(present)

        output_shape = input_shape + (hidden_states.size(-1),)
        return hidden_states.view(output_shape), torch.stack(presents)

class Pangu(nn.Module):
    def __init__(self, config):
        super(Pangu, self).__init__()
        self.embedding = Embendding(config)
        self.transformer = Transformer(config)
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-1)

    def forward(self, input_ids,
                position_ids,
                token_type_ids,
                lm_labels,
                past):

        hidden_states, presents = self.transformer(input_ids, position_ids, token_type_ids, past,self.embedding)
        lm_logits = torch.matmul(hidden_states, self.embedding.word_embeddings.weight.t())
        if lm_labels is not None:
            loss = self.loss_fct(lm_logits.view(-1, lm_logits.size(-1)), lm_labels.view(-1))
            return loss
        return lm_logits, presents