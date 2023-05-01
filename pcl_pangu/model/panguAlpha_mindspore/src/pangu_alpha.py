# Copyright 2022 PCL, Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""PanguAlpha model"""
import math
import os
import numpy as np
import mindspore.nn as nn
from mindspore.common.tensor import Tensor
from mindspore.common.parameter import Parameter
import mindspore.common.dtype as mstype
from mindspore.common.initializer import initializer, Normal, TruncatedNormal
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore import context
from mindspore.common.seed import _get_graph_seed
from mindspore._checkparam import Validator


class Dropout(nn.Cell):
    r"""
        A Dropout Implements with P.DropoutGenMask and  P.DropoutDoMask for parallel training.
    """

    def __init__(self, keep_prob=0.5, dtype=mstype.float32):
        super(Dropout, self).__init__()
        if keep_prob <= 0 or keep_prob > 1:
            raise ValueError(
                "dropout probability should be a number in range (0, 1], but got {}".format(
                    keep_prob))
        Validator.check_subclass("dtype", dtype, mstype.number_type, self.cls_name)
        Validator.check_value_type('keep_prob', keep_prob, [float], self.cls_name)
        self.keep_prob = keep_prob
        self.is_ascend = context.get_context('device_target') in ["Ascend"]
        if self.is_ascend:
            seed0, seed1 = _get_graph_seed(0, "dropout")
            self.seed0 = seed0
            self.seed1 = seed1
            self.dtype = dtype
            self.get_shape = P.Shape()
            self.dropout_gen_mask = P.DropoutGenMask(Seed0=self.seed0, Seed1=self.seed1)
            self.dropout_do_mask = P.DropoutDoMask()
            self.cast = P.Cast()
        else:
            self.dropout = P.Dropout(keep_prob)

    def construct(self, x):
        r"""
           Input: a tensor
           Returns: a tensor
        """
        if not self.training:
            return x

        if not self.is_ascend:
            out, _ = self.dropout(x)
            return out

        if self.keep_prob == 1:
            return x

        shape = self.get_shape(x)
        dtype = P.DType()(x)
        keep_prob = self.cast(self.keep_prob, dtype)
        output = self.dropout_gen_mask(shape, keep_prob)
        return self.dropout_do_mask(x, output, keep_prob)

    def extend_repr(self):
        return 'keep_prob={}, dtype={}'.format(self.keep_prob, self.dtype)

    def shard(self, strategy):
        if self.is_ascend:
            self.dropout_gen_mask.shard(strategy)
            self.dropout_do_mask.shard(strategy)
        else:
            self.dropout.shard(strategy)


class LayerNorm(nn.Cell):
    r"""
        A self-defined layer norm operation using reduce sum and reduce mean
    """

    def __init__(self, normalized_shape, dp=4, eps=1e-5, parallel_optimizer=False):
        super(LayerNorm, self).__init__()
        self.gamma = Parameter(initializer('ones', normalized_shape), name="gamma",
                               parallel_optimizer=parallel_optimizer)
        self.beta = Parameter(initializer('zeros', normalized_shape), name="beta",
                              parallel_optimizer=parallel_optimizer)
        self.mean = P.ReduceMean(keep_dims=True).shard(((dp, 1, 1),))
        self.square = P.Square().shard(((dp, 1, 1),))
        self.sqrt = P.Sqrt().shard(((dp, 1, 1),))
        self.sub1 = P.Sub().shard(((dp, 1, 1), (dp, 1, 1)))
        self.add = P.TensorAdd().shard(((dp, 1, 1), ()))
        self.mul = P.Mul().shard(((dp, 1, 1), (1,)))
        self.add2 = P.TensorAdd().shard(((dp, 1, 1), (1,)))
        self.real_div = P.RealDiv().shard(((dp, 1, 1), (dp, 1, 1)))
        self.eps = eps

    def construct(self, x):
        mean = self.mean(x, -1)
        diff = self.sub1(x, mean)
        variance = self.mean(self.square(diff), -1)
        variance_eps = self.sqrt(self.add(variance, self.eps))
        output = self.real_div(diff, variance_eps)
        output = self.add2(self.mul(output, self.gamma), self.beta)
        return output


class Mapping(nn.Cell):
    """
    A mapping function with a 3d input
    Args:
        input_size: the size of the last dimension of the input tensor
        output_size: the desired size of the last dimension of the output tensor
        dtype: the compute datatype
        scale: the scale factor for initialization
    Inputs:
        x: the 3d input
    Returns:
        output: Tensor, a 3d tensor after projection
    """

    # 优化：matmul，dtype, mapping_output
    def __init__(self, config, input_size, output_size, scale=1.0):
        super(Mapping, self).__init__()
        self.output_size = output_size
        self.input_size = input_size
        self.weight = Parameter(initializer(Normal(sigma=0.02 * scale),
                                            [input_size, output_size], config.param_init_type),
                                name="mapping_weight")
        self.bias = Parameter(initializer("zeros", [
            output_size,
        ], config.param_init_type),
                              name="mapping_bias",
                              parallel_optimizer=False)
        self.dtype = config.compute_dtype
        self.cast = P.Cast()
        self.add = P.TensorAdd().shard(((config.dp, 1), (1,)))
        self.matmul = P.MatMul().shard(
            ((config.dp, config.mp), (config.mp, 1)))

    def construct(self, x):
        out_shape = P.Shape()(x)[:-1] + (self.output_size,)
        x = P.Reshape()(x, (-1, self.input_size))
        weight = self.cast(self.weight, self.dtype)
        x = self.matmul(x, weight)
        x = self.add(x, self.cast(self.bias, self.dtype))
        output = P.Reshape()(x, out_shape)
        return output


class MappingOutput(nn.Cell):
    """
    A mapping function with a 3d input
    Args:
        input_size: the size of the last dimension of the input tensor
        output_size: the desired size of the last dimension of the output tensor
        dtype: the compute datatype
        scale: the scale factor for initialization
    Inputs:
        x: the 3d input
    Returns:
        output: Tensor, a 3d tensor after projection
    """

    def __init__(self, config, input_size, output_size, scale=1.0):
        super(MappingOutput, self).__init__()
        self.output_size = output_size
        self.input_size = input_size
        self.weight = Parameter(initializer(Normal(sigma=0.02 * scale),
                                            [input_size, output_size],
                                            config.param_init_type),
                                name="mapping_weight")
        self.bias = Parameter(initializer("zeros", [
            output_size,
        ], config.param_init_type),
                              name="mapping_bias")
        self.dtype = config.compute_dtype
        self.cast = P.Cast()
        self.add = P.TensorAdd().shard(((config.dp, config.mp), (config.mp,)))
        self.matmul = P.MatMul().shard(((config.dp, 1), (1, config.mp)))

    def construct(self, x):
        out_shape = P.Shape()(x)[:-1] + (self.output_size,)
        x = P.Reshape()(x, (-1, self.input_size))
        weight = self.cast(self.weight, self.dtype)
        x = self.matmul(x, weight)
        x = self.add(x, self.cast(self.bias, self.dtype))
        output = P.Reshape()(x, out_shape)
        return output


class FeedForwardLayer(nn.Cell):
    """
    The output mapping module for each layer
    Args:
        config(PanguAlphaConfig): the config of network
        scale: scale factor for initialization
    Inputs:
        x: output of the self-attention module
    Returns:
        output: Tensor, the output of this layer after mapping
    """

    def __init__(self, config, scale=1.0):
        super(FeedForwardLayer, self).__init__()
        input_size = config.embedding_size
        output_size = config.embedding_size * config.expand_ratio
        # Project to expand_ratio*embedding_size
        self.mapping = MappingOutput(config, input_size, output_size)
        # Project back to embedding_size
        self.projection = Mapping(config, output_size, input_size, scale)
        self.activation = nn.GELU()
        self.activation.gelu.shard(((config.dp, 1, config.mp),))
        self.dropout = Dropout(1 - config.dropout_rate)
        self.dropout.shard(((config.dp, 1, 1),))

    def construct(self, x):
        # [bs, seq_length, expand_ratio*embedding_size]
        hidden = self.activation(self.mapping(x))
        output = self.projection(hidden)
        # [bs, seq_length, expand_ratio]
        output = self.dropout(output)
        return output


class EmbeddingLookup(nn.Cell):
    """
    The embedding lookup table for vocabulary
    Inputs:
        input_ids: the tokenized inputs with datatype int32
    Returns:
        output: Tensor, the embedding vector for the input with shape (batch_size,
        seq_length, embedding_size)
        self.embedding_table: Tensor, the embedding table for the vocabulary
    """

    def __init__(self):
        super(EmbeddingLookup, self).__init__()
        self.gather = P.GatherV2()

    def construct(self, input_ids, table):
        output = self.gather(table, input_ids, 0)
        return output


class Attention(nn.Cell):
    """
    Self-Attention module for each layer

    Args:
        config(PanguAlphaConfig): the config of network
        scale: scale factor for initialization
        layer_idx: current layer index
    """

    def __init__(self, config, scale=1.0, layer_idx=None):
        super(Attention, self).__init__()
        # Output layer
        self.projection = Mapping(config, config.embedding_size,
                                  config.embedding_size, scale)
        self.transpose = P.Transpose().shard(((config.dp, 1, config.mp, 1),))
        self.merger_head_transpose = P.Transpose().shard(
            ((config.dp, config.mp, 1, 1),))
        self.reshape = P.Reshape()
        self.n_head = config.num_heads
        # embedding size per head
        self.size_per_head = config.embedding_size // self.n_head
        self.concat_k = P.Concat(axis=3)
        self.concat_v = P.Concat(axis=2)
        self.multiply_data = Tensor([
            -10000.0,
        ], dtype=mstype.float32)
        self.batch_matmul = P.BatchMatMul().shard(
            ((config.dp, config.mp, 1, 1), (config.dp, config.mp, 1, 1)))
        self.scale = scale
        self.real_div = P.RealDiv().shard(((config.dp, config.mp, 1, 1), ()))
        self.sub = P.Sub().shard(
            ((1,), (config.dp, 1, 1, 1)))
        self.mul = P.Mul().shard(
            ((config.dp, 1, 1, 1), (1,)))
        self.add = P.TensorAdd().shard(
            ((config.dp, 1, 1, 1), (config.dp, config.mp, 1, 1)))
        # Normalize factor for attention, sqrt(dk) as widely used
        if self.scale:
            self.scale_factor = Tensor(math.sqrt(self.size_per_head))
        if layer_idx is not None:
            self.coeff = math.sqrt(layer_idx * math.sqrt(self.size_per_head))
            self.coeff = Tensor(self.coeff)
        self.use_past = config.use_past
        self.dropout = Dropout(1 - config.dropout_rate)
        self.dropout.shard(((config.dp, 1, 1),))
        self.prob_dropout = Dropout(1 - config.dropout_rate)
        self.prob_dropout.shard(((config.dp, config.mp, 1, 1),))
        self.softmax = nn.Softmax()
        self.softmax.softmax.shard(((config.dp, config.mp, 1),))
        self.expand_dims = P.ExpandDims().shard(((config.dp, 1, 1),))

        dense_shape = [config.embedding_size, config.embedding_size]
        bias_shape = [config.embedding_size]
        # Query
        self.dense1 = nn.Dense(config.embedding_size,
                               config.embedding_size,
                               weight_init=initializer(init='normal', shape=dense_shape,
                                                       dtype=config.param_init_type),
                               bias_init=initializer(init='zeros', shape=bias_shape,
                                                     dtype=config.param_init_type)).to_float(config.compute_dtype)
        self.dense1.matmul.shard(((config.dp, 1), (config.mp, 1)))
        self.dense1.bias_add.shard(((config.dp, config.mp), (config.mp,)))
        # Key
        self.dense2 = nn.Dense(config.embedding_size,
                               config.embedding_size,
                               weight_init=initializer(init='normal',
                                                       shape=dense_shape,
                                                       dtype=config.param_init_type),
                               bias_init=initializer(init='zeros',
                                                     shape=bias_shape,
                                                     dtype=config.param_init_type)).to_float(config.compute_dtype)
        self.dense2.matmul.shard(((config.dp, 1), (config.mp, 1)))
        self.dense2.bias_add.shard(((config.dp, config.mp), (config.mp,)))
        # Value
        self.dense3 = nn.Dense(config.embedding_size,
                               config.embedding_size,
                               weight_init=initializer(init='normal',
                                                       shape=dense_shape,
                                                       dtype=config.param_init_type),
                               bias_init=initializer(init='zeros',
                                                     shape=bias_shape,
                                                     dtype=config.param_init_type)).to_float(config.compute_dtype)
        self.dense3.matmul.shard(((config.dp, 1), (config.mp, 1)))
        self.dense3.bias_add.shard(((config.dp, config.mp), (config.mp,)))

        self.is_first_iteration = True
        self.dtype = config.compute_dtype
        self.use_past = config.use_past
        if self.use_past:
            # operators used for state reuse
            seq_range = np.arange(config.seq_length).reshape(1, 1, -1)
            self.range = Tensor(np.tile(seq_range, (config.batch_size, 1, 1)), mstype.int32)
            self.seq_length = config.seq_length
            self.attention_mask = Tensor(np.tril(np.ones(shape=(self.seq_length, self.seq_length))), mstype.int32)
            self.slice = P.StridedSlice().shard(((1, 1, 1, 1),))
            self.not_equal = P.NotEqual().shard(((1, 1, 1, 1), ()))
            self.reducesum = P.ReduceSum().shard(((1, 1, 1, 1),))
            self.expand_dims = P.ExpandDims().shard(((1, 1, 1),))
            self.tensor_le = P.LessEqual().shard(((1, 1, 1), (1, 1, 1)))
            self.add = P.TensorAdd().shard(((1, 1, 1, 1), (1, 1, 1, 1)))
            self.equal = P.Equal().shard(((1, 1, 1), (1, 1, 1)))
            self.sub1 = P.Sub().shard(((1,), ()))
            self.tile = P.Tile().shard(((1, 1, 1, 1),))
            self.less = P.Less().shard(((1, 1, 1), (1, 1, 1)))
            self.mul1 = P.Mul().shard(((1, 1, 1, 1), (1, 1, 1, 1)))

    def construct(self, x, attention_mask, key_past=None, value_past=None, batch_valid_length=None):
        """
        self-attention

        Inputs:
            x: output of previous layer
            attention_mask: the attention mask matrix with shape (batch_size, 1,
            seq_length, seq_length)
            key_past: previous saved key state
            value_past: previous saved value state
            batch_valid_length: the valid input seq_length without padding

        Returns:
            output: Tensor, the output logit of this layer
            layer_present: Tensor, the feature map of current layer
        """

        original_shape = F.shape(x)
        x = F.reshape(x, (-1, original_shape[-1]))
        # Self attention: query, key, value are derived from the same inputs
        query = self.dense1(x)
        key = self.dense2(x)
        value = self.dense3(x)
        # [bs, num_heads, seq_length, size_per_head]
        query = self.transpose(
            F.reshape(
                query,
                (-1, original_shape[1], self.n_head, self.size_per_head)),
            (0, 2, 1, 3))
        # [bs, num_heads, size_per_head, seq_length]
        key = self.transpose(
            F.reshape(
                key, (-1, original_shape[1], self.n_head, self.size_per_head)),
            (0, 2, 3, 1))
        # [bs, num_heads, seq_length, size_per_head]
        value = self.transpose(
            F.reshape(
                value,
                (-1, original_shape[1], self.n_head, self.size_per_head)),
            (0, 2, 1, 3))

        # key and value for current token(s)
        key_present = key
        value_present = value
        if self.use_past:
            # The first graph with the input size of (bs, seq_length)
            if self.is_first_iteration:
                # Get the valid input length without padding
                valid_length_vector = F.cast(self.less(self.range, batch_valid_length.view(1, 1, -1)), self.dtype)
                # Cover the key and value numbers corresponding to the padding position
                key_present = self.mul1(key, self.expand_dims(valid_length_vector, 2))
                value_present = self.mul1(value, self.expand_dims(valid_length_vector, 3))
            # The second graph with the inpus size of (bs, 1)
            # the shape of query is (bs, num_heads, 1, size_per_head)
            # the shape of key is   (bs, num_heads, size_per_head, 1)
            # the shape of value is (bs, num_heads, 1, size_per_head)
            else:
                # Get the current token position index
                valid_length = self.reducesum(F.cast(self.not_equal(self.slice(key_past, (0, 0, 0, 0),
                                                                               (F.shape(x)[0], 1, 1, self.seq_length),
                                                                               (1, 1, 1, 1)),
                                                                    0), mstype.float32), (1, 2, 3))
                valid_length = F.reshape(valid_length, (-1, 1, 1))
                valid_length_vector = F.cast(self.equal(valid_length, self.range), self.dtype)
                # Pad the key and value to seq_length with only the position index not zero
                current_key = self.mul1(self.tile(key, (1, 1, 1, self.seq_length)),
                                        self.expand_dims(valid_length_vector, 2))
                current_value = self.mul1(self.tile(value, (1, 1, self.seq_length, 1)),
                                          self.expand_dims(valid_length_vector, 3))
                # Concat the previous saved state and current state
                key = self.add(key_past, current_key)
                value = self.add(value_past, current_value)
                # Update key_present and value_present for state update
                key_present = key
                value_present = value
                attention_mask = F.reshape(self.attention_mask, (self.seq_length, self.seq_length, 1, 1))

        layer_present = (key_present, value_present)
        # Self-attention considering attention mask
        attention = self._attn(query, key, value, attention_mask)
        # [bs, seq_length, embedding_size]
        attention_merge = self.merge_heads(attention)
        # Output
        output = self.projection(attention_merge)
        output = self.dropout(output)
        return output, layer_present

    def split_heads(self, x, transpose):
        """
        split 3d tensor to 4d and switch certain axes
        Inputs:
            x: input tensor
            transpose: tuple, the transpose sequence
        Returns:
            x_transpose: the 4d output
        """
        x_size = P.Shape()(x)
        new_x_shape = x_size[:-1] + (self.n_head, self.size_per_head)
        x = self.reshape(x, new_x_shape)
        x_transpose = self.transpose(x, transpose)
        return x_transpose

    def merge_heads(self, x):
        """
        convert a 4d input to a 3d output

        Inputs:
            x: input tensor

        Returns:
            x_merge: the 3d output
        """
        x = self.merger_head_transpose(
            x, (0, 2, 1, 3))  # bs, seq_length, head, size_per_head
        x_shape = P.Shape()(x)
        new_shape = x_shape[:-2] + (x_shape[-2] * x_shape[-1],)
        x_merge = self.reshape(x, new_shape)
        return x_merge

    def _attn(self, query, key, value, attention_mask):
        """
        Get the weighted score along the seq_length

        Inputs:
            query: the query matrix
            key: the key matrix
            value: the value matrix
            attention_mask: the attention mask matrix with shape (batch_size,
            1, seq_length, seq_length)
        Returns:
            weighted_values: Tensor, the weighted sum scores
        """
        # Normalize query and key before MatMul, default off
        if not self.scale:
            query = query / F.cast(self.coeff, F.dtype(query))
            key = key / F.cast(self.coeff, F.dtype(key))

        # Attention score [bs, num_heads, seq_length_q, seq_length_k]
        score = self.batch_matmul(query, key)
        # Normalize after query and key MatMul, default on
        if self.scale:
            score = self.real_div(
                score,
                P.Cast()(self.scale_factor, P.DType()(score)))

        ori_dtype = P.DType()(score)
        score = P.Cast()(score, mstype.float32)

        # for input size of (bs, 1) namely the second graph, the shape of attention_mask matrix should be
        # (bs, 1, 1, seq_length)
        if self.use_past and not self.is_first_iteration:
            # Calculate the current total token
            current_index = self.reducesum(F.cast(self.not_equal(self.slice(key, (0, 0, 0, 0),
                                                                            (F.shape(query)[0], 1, 1, self.seq_length),
                                                                            (1, 1, 1, 1)),
                                                                 0), mstype.float32), (1, 2, 3))
            # Get the precise position index
            index = self.sub1(F.cast(current_index, mstype.int32), 1)
            index = F.reshape(index, (-1, 1, 1))
            # Calculate the attention_mask matrix via the position index
            attention_mask = F.cast(self.tensor_le(self.range, index), mstype.int32)
            attention_mask = self.expand_dims(attention_mask, 2)

        # Minus 10000 for the position where masked to exclude them from softmax
        multiplu_out = self.sub(
            P.Cast()(F.tuple_to_array((1.0,)), P.DType()(score)),
            P.Cast()(attention_mask, P.DType()(score)))

        adder = self.mul(multiplu_out, self.multiply_data)
        attention_scores = self.add(adder, score)

        shape = F.shape(attention_scores)
        # attention probs
        attention_probs = self.softmax(
            F.reshape(attention_scores,
                      (shape[0], -1, shape[-1])))
        attention_probs = P.Cast()(attention_probs, ori_dtype)
        attention_probs = F.reshape(attention_probs, shape)

        attention_probs = self.prob_dropout(attention_probs)
        # Weighted sum output [bs, num_heads, seq_length_q, size_per_head]
        weighted_values = self.batch_matmul(attention_probs, value)
        return weighted_values


class Decoder(nn.Cell):
    """
    The basic decoder structure of PanguAlpha network
    Args:
        config(PanguAlphaConfig): the config of network
        layer_idx: current layer index
    Inputs:
        x: the output of previous layer(input_ids for the first layer)
        attention_mask: the attention mask matrix with shape (batch_size, 1, seq_length, seq_length)
        init_reset: whether reset the previous state
        batch_valid_length: the valid input seq_length without padding
    Returns:
        output: Tensor, the output logit of this layer
        layer_present: Tensor, the feature map of current layer
    """

    def __init__(self, config, layer_idx):
        super(Decoder, self).__init__()
        scale = 1 / math.sqrt(2.0 * config.num_layers)
        self.layernorm1 = LayerNorm((config.embedding_size,), config.dp).to_float(mstype.float32)
        self.layernorm2 = LayerNorm((config.embedding_size,), config.dp).to_float(mstype.float32)

        self.attention = Attention(config, scale, layer_idx)
        # Feed Forward Network, FFN
        self.output = FeedForwardLayer(config, scale)
        self.post_layernorm_residual = config.post_layernorm_residual
        self.add = P.TensorAdd().shard(((config.dp, 1, 1), (config.dp, 1, 1)))
        # Last activation of this layer will be saved for recompute in backward process
        self.dtype = config.compute_dtype
        self.use_past = config.use_past
        if self.use_past:
            # operator used for state reuse
            self.reducesum = P.ReduceSum().shard(((1, 1, 1, 1),))
            self.not_equal = P.NotEqual().shard(((1, 1, 1, 1), ()))
            self.slice = P.StridedSlice().shard(((1, 1, 1, 1),))
            size_per_head = int(config.embedding_size / config.num_heads)
            self.key_shape = (config.batch_size, config.num_heads, size_per_head, config.seq_length)
            self.value_shape = (config.batch_size, config.num_heads, config.seq_length, size_per_head)
            # parameters saving key and value states
            self.key_past = Parameter(Tensor(np.zeros(shape=self.key_shape), self.dtype), name="key_past")
            self.value_past = Parameter(Tensor(np.zeros(shape=self.value_shape), self.dtype), name="value_past")
            self.tile = P.Tile().shard(((1, 1),))
            self.mul = P.Mul().shard(((1, 1, 1, 1), (1,)))
            self.assign = P.Assign().shard(((1, 1, 1, 1), (1, 1, 1, 1)))

    def construct(self, x, input_mask, init_reset=True, batch_valid_length=None):
        r"""
        The forward process of the block.
        """
        # [bs, seq_length, embedding_size]
        input_x = self.layernorm1(x)
        input_x = F.cast(input_x, self.dtype)

        # indicate whether reset saved states
        key_reset = None
        value_reset = None

        if self.use_past:
            # reset states, init_reset True for reuse and False for reset
            key_reset = self.assign(self.key_past, self.mul(self.key_past, F.cast(init_reset, self.dtype)))
            value_reset = self.assign(self.value_past, self.mul(self.value_past, F.cast(init_reset, self.dtype)))
            # add dependency for desired execution order
            input_x = F.depend(input_x, key_reset)
            input_x = F.depend(input_x, value_reset)
        attention, layer_present = self.attention(input_x, input_mask,
                                                  self.key_past, self.value_past, batch_valid_length)
        # For post-layernorm the inputs for residual path are output of self-attention and output of layernorm
        if self.post_layernorm_residual:
            x = self.add(input_x, attention)
        # For pre-layernorm the inputs for residual path are output of self-attention and input of this layer
        else:
            x = self.add(x, attention)

        output_x = self.layernorm2(x)
        output_x = F.cast(output_x, self.dtype)
        mlp_logit = self.output(output_x)

        value_update = None
        key_update = None
        if self.use_past:
            # current key and value
            key_present, value_present = layer_present
            # update key and value calculated this step
            key_update = self.assign(self.key_past, key_present)
            value_update = self.assign(self.value_past, value_present)
            # add dependency for desired execution order
            key_update = F.depend(key_update, key_reset)
            value_update = F.depend(value_update, value_reset)

        # add dependency for desired execution order
        mlp_logit = F.depend(mlp_logit, value_update)
        mlp_logit = F.depend(mlp_logit, key_update)
        if self.post_layernorm_residual:
            output = self.add(output_x, mlp_logit)
        else:
            output = self.add(x, mlp_logit)
        return output, layer_present


class Embedding(nn.Cell):
    """
    Embedding
    """

    def __init__(self, config):
        super(Embedding, self).__init__()
        self.word_embedding = EmbeddingLookup().set_comm_fusion(1)
        if config.word_emb_dp:
            self.word_embedding.gather.shard(((1, 1), (config.dp, 1)))
        else:
            self.word_embedding.gather.shard(((config.mp, 1), (1, 1)))
        if config.stage_num > 1:
            self.position_embedding = nn.Embedding(config.seq_length,
                                                   config.embedding_size,
                                                   embedding_table=Normal(0.02)).set_comm_fusion(1)
        else:
            # Position embedding
            if config.load_ckpt_path:
                # Loading the embedding table from the ckpt path:
                embedding_path = os.path.join(config.load_ckpt_path, 'position_embedding.npy')
                if os.path.exists(embedding_path):
                    p_table = np.load(embedding_path)
                    position_table_param = Tensor(p_table, mstype.float32)
                    print("###### loading Position embedding Table successed! ######")
                else:
                    raise ValueError(f"{embedding_path} file not exits, "
                                     f"please check whether position_embedding file exit.")
            else:
                position_table_param = TruncatedNormal(0.02)
            # Position embedding
            self.position_embedding = nn.Embedding(
                config.seq_length,
                config.embedding_size,
                embedding_table=position_table_param).set_comm_fusion(1)
        self.position_embedding.embedding_table.parallel_optimizer = False
        self.position_embedding.gather.shard(((1, 1), (config.dp,)))
        self.position_embedding.expand.shard(((config.dp, 1),))
        self.add = P.TensorAdd().shard(((config.dp, 1, 1), (config.dp, 1, 1)))
        self.dropout = Dropout(1 - config.dropout_rate)
        self.dropout.shard(((config.dp, 1, 1),))
        self.use_past = config.use_past
        self.is_first_iteration = True

    def construct(self, input_ids, table, input_position, valid_index=None):
        input_embedding = self.word_embedding(input_ids, table)
        if self.use_past and not self.is_first_iteration:
            _, seq_length = F.shape(input_ids)
            input_position = valid_index.view(1, seq_length)
        position_embedding = self.position_embedding(input_position)
        hidden_states = self.add(input_embedding, position_embedding)
        hidden_states = self.dropout(hidden_states)
        hidden_states = P.Cast()(hidden_states, mstype.float16)
        return hidden_states


class Mask(nn.Cell):
    """
    Mask
    """

    def __init__(self, config):
        super(Mask, self).__init__()
        self.dtype = config.compute_dtype
        self.expand_dims = P.ExpandDims().shard(((config.dp, 1, 1),))

    def construct(self, attention_mask):
        attention_mask = self.expand_dims(attention_mask, 1)
        return attention_mask


class QueryLayerAttention(Attention):
    r"""
    Self-Attention module using input query vector.
    """

    def construct(self, x, query_hidden_state, attention_mask, key_past=None, value_past=None, batch_valid_length=None):
        original_shape = F.shape(x)
        x = F.reshape(x, (-1, original_shape[-1]))
        query_hidden_state = F.reshape(query_hidden_state, (-1, original_shape[-1]))
        # For query_layer_attention, query are derived from outputs of previous layer and key, value are derived from an added parameter query_embedding
        query = self.dense1(query_hidden_state)
        key = self.dense2(x)
        value = self.dense3(x)
        query = self.transpose(
            F.reshape(
                query,
                (-1, original_shape[1], self.n_head, self.size_per_head)),
            (0, 2, 1, 3))
        key = self.transpose(
            F.reshape(
                key, (-1, original_shape[1], self.n_head, self.size_per_head)),
            (0, 2, 3, 1))
        value = self.transpose(
            F.reshape(
                value,
                (-1, original_shape[1], self.n_head, self.size_per_head)),
            (0, 2, 1, 3))

        key_present = key
        value_present = value
        if self.use_past:
            # The first graph with the input size of (bs, seq_length)
            if self.is_first_iteration:
                # Get the valid input length without padding
                valid_length_vector = F.cast(self.less(self.range, batch_valid_length.view(1, 1, -1)), self.dtype)
                # Cover the key and value numbers corresponding to the padding position
                key_present = self.mul1(key, self.expand_dims(valid_length_vector, 2))
                value_present = self.mul1(value, self.expand_dims(valid_length_vector, 3))
            # The second graph with the inpus size of (bs, 1)
            # the shape of query is (bs, num_heads, 1, size_per_head)
            # the shape of key is   (bs, num_heads, size_per_head, 1)
            # the shape of value is (bs, num_heads, 1, size_per_head)
            else:
                # Get the current token position index
                valid_length = self.reducesum(F.cast(self.not_equal(self.slice(key_past, (0, 0, 0, 0),
                                                                               (F.shape(x)[0], 1, 1, self.seq_length),
                                                                               (1, 1, 1, 1)),
                                                                    0), mstype.float32), (1, 2, 3))
                valid_length = F.reshape(valid_length, (-1, 1, 1))
                valid_length_vector = F.cast(self.equal(valid_length, self.range), self.dtype)
                # Pad the key and value to seq_length with only the position index not zero
                current_key = self.mul1(self.tile(key, (1, 1, 1, self.seq_length)),
                                        self.expand_dims(valid_length_vector, 2))
                current_value = self.mul1(self.tile(value, (1, 1, self.seq_length, 1)),
                                          self.expand_dims(valid_length_vector, 3))
                # Concat the previous saved state and current state
                key = self.add(key_past, current_key)
                value = self.add(value_past, current_value)
                # Update key_present and value_present for state update
                key_present = key
                value_present = value
                attention_mask = F.reshape(self.attention_mask, (self.seq_length, self.seq_length, 1, 1))
        layer_present = (key_present, value_present)
        attention = self._attn(query, key, value, attention_mask)
        attention_merge = self.merge_heads(attention)
        output = self.projection(attention_merge)
        output = self.dropout(output)
        return output, layer_present


class QueryLayer(nn.Cell):
    r"""
    A block usingooked out position embedding as query vector.
    This is used as the final block.
    """

    def __init__(self, config):
        super(QueryLayer, self).__init__()
        scale = 1 / math.sqrt(2.0 * config.num_layers)
        self.layernorm1 = LayerNorm((config.embedding_size,), config.dp).to_float(mstype.float32)
        self.layernorm2 = LayerNorm((config.embedding_size,), config.dp).to_float(mstype.float32)
        self.attention = QueryLayerAttention(config, scale)
        self.output = FeedForwardLayer(config, scale)
        self.post_layernorm_residual = config.post_layernorm_residual
        self.add = P.TensorAdd().shard(((config.dp, 1, 1), (config.dp, 1, 1)))
        self.dtype = config.compute_dtype
        self.use_past = config.use_past
        if self.use_past:
            # operator used for state reuse
            self.reducesum = P.ReduceSum().shard(((1, 1, 1, 1),))
            self.not_equal = P.NotEqual().shard(((1, 1, 1, 1), ()))
            self.slice = P.StridedSlice().shard(((1, 1, 1, 1),))
            size_per_head = int(config.embedding_size / config.num_heads)
            self.key_shape = (config.batch_size, config.num_heads, size_per_head, config.seq_length)
            self.value_shape = (config.batch_size, config.num_heads, config.seq_length, size_per_head)
            # parameters saving key and value states
            self.key_past = Parameter(Tensor(np.zeros(shape=self.key_shape), self.dtype), name="key_past")
            self.value_past = Parameter(Tensor(np.zeros(shape=self.value_shape), self.dtype), name="value_past")
            self.tile = P.Tile().shard(((1, 1),))
            self.mul = P.Mul().shard(((1, 1, 1, 1), (1,)))
            self.assign = P.Assign().shard(((1, 1, 1, 1), (1, 1, 1, 1)))

    def construct(self, x, query_hidden_state, input_mask, init_reset=True, batch_valid_length=None):
        r"""
        Query Layer shares a similar structure with normal layer block
        except that it is not a traditional self-attention.
        """
        input_x = self.layernorm1(x)
        input_x = F.cast(input_x, self.dtype)

        # indicate whether reset saved states
        key_reset = None
        value_reset = None

        if self.use_past:
            # reset states, init_reset True for reuse and False for reset
            key_reset = self.assign(self.key_past, self.mul(self.key_past, F.cast(init_reset, self.dtype)))
            value_reset = self.assign(self.value_past, self.mul(self.value_past, F.cast(init_reset, self.dtype)))
            # add dependency for desired execution order
            input_x = F.depend(input_x, key_reset)
            input_x = F.depend(input_x, value_reset)

        attention, layer_present = self.attention(input_x,
                                                  query_hidden_state,
                                                  input_mask,
                                                  self.key_past,
                                                  self.value_past,
                                                  batch_valid_length)
        if self.post_layernorm_residual:
            x = self.add(input_x, attention)
        else:
            x = self.add(x, attention)

        output_x = self.layernorm2(x)
        output_x = F.cast(output_x, self.dtype)
        mlp_logit = self.output(output_x)
        value_update = None
        key_update = None
        if self.use_past:
            # current key and value
            key_present, value_present = layer_present
            # update key and value calculated this step
            key_update = self.assign(self.key_past, key_present)
            value_update = self.assign(self.value_past, value_present)
            # add dependency for desired execution order
            key_update = F.depend(key_update, key_reset)
            value_update = F.depend(value_update, value_reset)

        # add dependency for desired execution order
        mlp_logit = F.depend(mlp_logit, value_update)
        mlp_logit = F.depend(mlp_logit, key_update)

        if self.post_layernorm_residual:
            output = self.add(output_x, mlp_logit)
        else:
            output = self.add(x, mlp_logit)
        return output, layer_present


class PanguAlphaEmbedding(nn.Cell):
    """
    Input embedding, i.e., word embedding and position embedding
    Args:
        config(PanguAlphaConfig): the config of network
    Inputs:
        input_ids: the tokenized inputs with datatype int32
        input_mask: the mask indicating whether each position is a valid input
        input_position: the position index of each token
        attention_mask: the attention_mask attention for self-attention module
        valid_index: only used in incremental inference, the position index of current token
    outputs:
        hidden_states: Tensor, input embeddings
        attention_mask: Tensor, attention_mask matrix
        embedding_table: Tensor, embedding_table with shape of (vocab_size, embedding_size)
    """

    def __init__(self, config):
        super(PanguAlphaEmbedding, self).__init__()
        self.embedding = Embedding(config)
        if config.stage_num > 1:
            self.embedding.pipeline_stage = 0
        self.mask = Mask(config)

    def construct(self, input_ids, input_mask, table, input_position, attention_mask, valid_index=None):
        """
        Calculate input embeddings via input token ids and input position
        """
        hidden_states = self.embedding(input_ids, table, input_position, valid_index)
        attention_mask = self.mask(attention_mask)
        return hidden_states, attention_mask


class PanguAlpha_Model(nn.Cell):
    """
    The backbone of PanguAlpha network
    Args:
        config(PanguAlphaConfig): the config of network
    Inputs:
        input_ids: the tokenized inputs with datatype int32
        input_mask: the mask indicating whether each position is a valid input
        layer_past: the previous feature map
    Returns:
        output_state: Tensor, the output logit of backbone
        present_layer: Tensor, the current feature map
        embedding_table: Tensor, the embedding table for the vocabulary
    """

    def __init__(self, config):
        super(PanguAlpha_Model, self).__init__()
        self.embedding = PanguAlphaEmbedding(config)
        self.blocks = nn.CellList()
        self.use_past = config.use_past
        self.dtype = config.compute_dtype
        self.num_layers = config.num_layers
        self.is_pipeline = (config.stage_num > 1)
        if self.is_pipeline:
            self.top_query_embedding_table = Parameter(initializer(TruncatedNormal(0.02),
                                                                   [config.seq_length, config.embedding_size]),
                                                       name='embedding_table', parallel_optimizer=False)
            self.top_query_embedding = EmbeddingLookup()
            for i in range(config.num_layers):
                if i == config.num_layers - 1:
                    self.top_query_embedding_table.comm_fusion = 2
                    self.top_query_embedding_table.add_pipeline_stage(i * config.stage_num // config.num_layers)
                    per_block = QueryLayer(config).set_comm_fusion(2)
                else:
                    per_block = Decoder(config, i + 1).set_comm_fusion(2)
                per_block.pipeline_stage = i * config.stage_num // config.num_layers
                per_block.recompute()
                self.blocks.append(per_block)

            self.layernorm = LayerNorm((config.embedding_size,), config.dp).to_float(mstype.float32)
            self.layernorm.set_comm_fusion(2)
            self.layernorm.pipeline_stage = config.stage_num - 1
        else:
            # The input_position representing the position ids will be used as the index
            # for a query embedding table to obtain top query hidden states, together with the previous outputs of normal
            # self-attention layers, a new attention layer will be attached to the output of the model
            if config.load_ckpt_path:
                # Loading the embedding table from the ckpt path:
                embedding_path = os.path.join(config.load_ckpt_path, 'top_query_embedding.npy')
                if os.path.exists(embedding_path):
                    top_query_table = np.load(embedding_path)
                    top_query_table_param = Tensor(top_query_table, mstype.float32)
                    print("###### loading Toq_query embedding Table successed! ######")
                else:
                    raise ValueError(
                        f"{embedding_path} file not exits, please check whether top_query_embedding file exist.")
            else:
                top_query_table_param = TruncatedNormal(0.02)
            self.top_query_embedding_table = Parameter(initializer(top_query_table_param,
                                                                   [config.seq_length, config.embedding_size]),
                                                       name='embedding_table', parallel_optimizer=False)
            self.top_query_embedding = EmbeddingLookup()
            # Total fusion groups for HCCL operators. Specifically, the same tyep HCCL operators in same group will be fused.
            fusion_group_num = 4
            fusion_group_size = config.num_layers // fusion_group_num
            fusion_group_size = max(fusion_group_size, 1)
            for i in range(config.num_layers-1):
                per_block = Decoder(config, i + 1).set_comm_fusion(int(i / fusion_group_size) + 2)
                # Each layer will be remoputed in the backward process. The output activation of each layer will be saved,
                # in other words, in backward process each block will be almosttotally recomputed.
                per_block.recompute()
                self.blocks.append(per_block)
            self.layernorm = LayerNorm((config.embedding_size,), config.dp).to_float(mstype.float32)
            self.layernorm.set_comm_fusion(int((config.num_layers - 1) / fusion_group_size) + 2)

            self.top_query_layer = QueryLayer(config).set_comm_fusion(
                int((config.num_layers - 1) / fusion_group_size) + 2)
            self.top_query_layer.recompute()
            self.top_query_embedding_table.comm_fusion = int((config.num_layers - 1) / fusion_group_size) + 2
        self.top_query_embedding.gather.shard(((1, 1), (config.dp,)))

    def construct(self, input_ids, input_mask, table, input_position, attention_mask,
                  init_reset=True, batch_valid_length=None):
        """PanguAlpha model"""
        # embedding for input_ids and the lower triangle like attention_mask matrix
        hidden_states, attention_mask = self.embedding(input_ids, input_mask, table,
                                                       input_position, attention_mask,
                                                       batch_valid_length)
        for i in range(self.num_layers - 1):
            hidden_states, _ = self.blocks[i](hidden_states,
                                              attention_mask, init_reset, batch_valid_length)
        if self.is_pipeline:
            top_query_hidden_states = self.top_query_embedding(input_position.view(-1,),
                                                               self.top_query_embedding_table)
            hidden_states, _ = self.blocks[self.num_layers - 1](hidden_states, top_query_hidden_states,
                                                                attention_mask, init_reset, batch_valid_length)
            output_state = self.layernorm(hidden_states)
            output_state = F.cast(output_state, self.dtype)
        else:
            output_state = self.layernorm(hidden_states)
            output_state = F.cast(output_state, self.dtype)
            top_query_hidden_states = self.top_query_embedding(input_position.view(-1,), self.top_query_embedding_table)
            output_state, _ = self.top_query_layer(output_state, top_query_hidden_states,
                                                   attention_mask, init_reset, batch_valid_length)
        return output_state


class PanguAlpha_Head(nn.Cell):
    """
    Head for PanguAlpha to get the logits of each token in the vocab
    Args:
        config(PanguAlphaConfig): the config of network
    Inputs:
        state: the output of the backbone
        embedding_table: the embedding table of the vocabulary
    Returns:
        logits: Tensor, the logits of the corresponding inputs
    """

    def __init__(self, config):
        super(PanguAlpha_Head, self).__init__()
        if config.word_emb_dp:
            self.matmul = P.MatMul(transpose_b=True).shard(((config.dp, 1), (1, 1)))
        else:
            self.matmul = P.MatMul(transpose_b=True).shard(((config.dp, 1), (config.mp, 1)))
        self.embedding_size = config.embedding_size
        self.log_softmax = P.LogSoftmax(axis=-1)
        self.dtype = config.compute_dtype
        self.cast = P.Cast()

    def construct(self, state, embedding_table):
        state = P.Reshape()(state, (-1, self.embedding_size))
        # output logits over vocabulary [bs*seq_length, vocab_size]
        logits = self.matmul(state, self.cast(embedding_table, self.dtype))
        return logits


class PanguAlpha(nn.Cell):
    """
    The PanguAlpha network consisting of two parts the backbone and the head
    Args:
        config(PanguAlphaConfig): the config of network
    Inputs:
        input_ids: the tokenized inputs
        input_mask: the mask indicating whether each position is a valid input
        past: the previous feature map
    Returns:
        logits: Tensor: the logits of the corresponding inputs with shape (batch_size, seq_length, vocab_size)
    """

    def __init__(self, config):
        super(PanguAlpha, self).__init__()
        # Network head to get logits over vocabulary
        self.head = PanguAlpha_Head(config)
        self.vocab_size = config.vocab_size
        self.embedding_size = config.embedding_size
        self.backbone = PanguAlpha_Model(config)
        if config.stage_num > 1:

            self.head.pipeline_stage = config.stage_num - 1
            self.embedding_table = Parameter(initializer(Normal(0.02), [self.vocab_size, self.embedding_size]),
                                             name="embedding_table", parallel_optimizer=False)
            self.embedding_table.add_pipeline_stage(self.backbone.blocks[0].pipeline_stage)
            self.embedding_table.add_pipeline_stage(self.head.pipeline_stage)
        else:
            if config.load_ckpt_path:
                # Loading the embedding table from the ckpt path:
                embedding_path = os.path.join(config.load_ckpt_path, 'word_embedding.npy')
                if os.path.exists(embedding_path):
                    e_table = np.load(embedding_path)
                    e_table = Tensor(e_table, mstype.float32)
                    self.embedding_table = Parameter(e_table, name="embedding_table", parallel_optimizer=False)
                    print("###### loading Word embedding Table successed! ######")
                else:
                    raise ValueError(f"{embedding_path} file not exits, "
                                     f"please check whether word_embedding file exist.")
            else:
                self.embedding_table = Parameter(initializer(Normal(0.02), [self.vocab_size, self.embedding_size]),
                                                 name="embedding_table", parallel_optimizer=False)

    def construct(self, input_ids, input_mask, input_position, attention_mask,
                  init_reset=True, batch_valid_length=None):
        output_states = self.backbone(input_ids, input_mask, self.embedding_table,
                                      input_position, attention_mask, init_reset, batch_valid_length)
        logits = self.head(output_states, self.embedding_table)
        return logits


class CrossEntropyLoss(nn.Cell):
    """
    Calculate the cross entropy loss
    Args:
        config(PanguAlphaConfig): the config of the network
    Inputs:
        logits: the output logits of the backbone
        label: the ground truth label of the sample
        input_mask: the mask indicating whether each position is a valid input
    Returns:
        loss: Tensor, the corrsponding cross entropy loss
    """

    def __init__(self, config):
        super(CrossEntropyLoss, self).__init__()
        self.mean = P.ReduceMean()
        self.sum = P.ReduceSum().shard(((config.dp, config.mp),))
        self.onehot = P.OneHot().shard(((config.dp, config.mp), (), ()))
        # on/off value for onehot, for smooth labeling, modify the off_value
        self.on_value = Tensor(1.0, mstype.float32)
        self.off_value = Tensor(0.0, mstype.float32)
        self.vocab_size = config.vocab_size
        self.max = P.ArgMaxWithValue(axis=-1, keep_dims=True).shard(
            ((config.dp, config.mp),))
        self.eps_const = Tensor(1e-24, mstype.float32)
        self.sub = P.Sub().shard(((config.dp, config.mp), (config.dp, 1)))
        self.exp = P.Exp().shard(((config.dp, config.mp),))
        self.div = P.RealDiv().shard(((config.dp, config.mp), (config.dp, 1)))
        self.log = P.Log().shard(((config.dp, config.mp),))
        self.add = P.TensorAdd().shard(((config.dp, config.mp), ()))
        self.mul = P.Mul().shard(
            ((config.dp, config.mp), (config.dp, config.mp)))
        self.neg = P.Neg().shard(((config.dp, config.mp),))
        self.sum2 = P.ReduceSum().shard(((1,),))

        self.mul2 = P.Mul().shard(((1,), (1,)))
        self.add2 = P.TensorAdd()
        self.div2 = P.RealDiv()

    def construct(self, logits, label, input_mask):
        r"""
        Compute loss using logits, label and input mask
        """
        # [bs*seq_length, vocab_size]
        logits = F.cast(logits, mstype.float32)
        # LogSoftmax for logits over last dimension
        _, logit_max = self.max(logits)
        logit_sub = self.sub(logits, logit_max)
        logit_exp = self.exp(logit_sub)
        exp_sum = self.sum(logit_exp, -1)
        exp_sum = P.Reshape()(exp_sum, (F.shape(exp_sum)[0], 1))
        softmax_result = self.div(logit_exp, exp_sum)
        log_softmax_result = self.log(self.add(softmax_result, self.eps_const))

        # Flatten label to [bs*seq_length]
        label = P.Reshape()(label, (-1,))
        # Get onehot label [bs*seq_length, vocab_size]
        one_hot_label = self.onehot(label, self.vocab_size, self.on_value,
                                    self.off_value)
        # Cross-Entropy loss
        loss = self.mul(log_softmax_result, one_hot_label)
        loss_unsum = self.neg(loss)
        loss_reduce = self.sum(loss_unsum, -1)
        # input_mask indicates whether there is padded inputs and for padded inputs it will not be counted into loss
        input_mask = P.Reshape()(input_mask, (-1,))
        numerator = self.sum2(self.mul2(loss_reduce, input_mask))

        denominator = self.add2(
            self.sum2(input_mask),
            P.Cast()(F.tuple_to_array((1e-5,)), mstype.float32))
        loss = self.div2(numerator, denominator)
        return loss


class PanguAlphaWithLoss(nn.Cell):
    """
    PanguAlpha training loss
    Args:
        network: backbone network of PanguAlpha
        loss: loss function, e.g., crossentropy
        eos_token: the end_of_sentence token
    Inputs:
        input_ids: the tokenized inputs
        past: the previous feature map
    Returns:
        output: Tensor, the loss of the network
    """

    def __init__(self, config, network, loss, eos_token=6):
        super(PanguAlphaWithLoss, self).__init__(auto_prefix=False)
        self.network = network
        self.loss = loss
        self.eos_token = eos_token
        self.slice = P.StridedSlice().shard(((config.dp, 1),))
        self.not_equal = P.NotEqual().shard(((config.dp, 1), ()))
        self.batch_size = config.batch_size
        self.len = config.seq_length
        self.micro_batch_step = 1
        if config.stage_num > 1:
            self.micro_batch_step = config.micro_size

    def construct(self, input_ids, input_position, attention_mask):
        tokens = self.slice(input_ids, (0, 0), (self.batch_size // self.micro_batch_step, -1), (1, 1))
        input_mask = F.cast(self.not_equal(tokens, self.eos_token), mstype.float32)
        
        # input_position = self.slice(input_position, (0, 0), (self.batch_size, self.len), (1, 1))
        # attention_mask = self.slice_mask(attention_mask, (0, 0, 0),
        #                                  (self.batch_size, self.len, self.len),
        #                                  (1, 1, 1))
        
        logits = self.network(tokens, input_mask, input_position, attention_mask)
        labels = self.slice(input_ids, (0, 1), (self.batch_size // self.micro_batch_step,
                                                self.len + 1), (1, 1))
        output = self.loss(logits, labels, input_mask)
        return output

class PanguAlphaWithLoss_eval_nli(nn.Cell):
    """
    PanguAlpha training loss
    Args:
        network: backbone network of PanguAlpha
        loss: loss function, e.g., crossentropy
        eos_token: the end_of_sentence token
    Inputs:
        input_ids: the tokenized inputs
        past: the previous feature map
    Returns:
        output: Tensor, the loss of the network
    """

    def __init__(self, config, network, loss, eos_token=6):
        super(PanguAlphaWithLoss_eval_nli, self).__init__(auto_prefix=False)
        self.network = network
        self.loss = loss
        self.eos_token = eos_token
        self.slice = P.StridedSlice().shard(((config.dp, 1),))
        self.not_equal = P.NotEqual().shard(((config.dp, 1), ()))
        self.batch_size = config.batch_size
        self.len = config.seq_length
        self.micro_batch_step = 1
        if config.stage_num > 1:
            self.micro_batch_step = config.micro_size

    def construct(self, input_ids, input_position=None, attention_mask=None):
        tokens = self.slice(input_ids, (0, 0), (self.batch_size // self.micro_batch_step, -1), (1, 1))
        input_mask = F.cast(self.not_equal(tokens, self.eos_token), mstype.float32)
        
        input_position = self.slice(input_position, (0, 0), (self.batch_size, self.len), (1, 1))
        attention_mask = self.slice_mask(attention_mask, (0, 0, 0),
                                         (self.batch_size, self.len, self.len),
                                         (1, 1, 1))
        
        logits = self.network(tokens, input_mask, input_position, attention_mask)
        labels = self.slice(input_ids, (0, 1), (self.batch_size // self.micro_batch_step,
                                                self.len + 1), (1, 1))
        output = self.loss(logits, labels, input_mask)
        return output
    
class PanguAlphaWithLoss_eval(nn.Cell):
    """
    GPT training loss
    Args:
        network: backbone network of GPT2/3
        loss: loss function, e.g., crossentropy
        eos_token: the end_of_sentence token
    Inputs:
        input_ids: the tokenized inputs
        past: the previous feature map
    Returns:
        output: Tensor, the loss of the network
    """
    def __init__(self, config, network, loss, eos_token=128297):
        super(PanguAlphaWithLoss_eval, self).__init__(auto_prefix=False)
        self.network = network
        self.loss = loss
        self.eos_token = eos_token
        self.slice = P.StridedSlice().shard(((config.dp, 1),))
        self.not_equal = P.NotEqual().shard(((config.dp, 1), ()))
        self.batch_size = config.batch_size
        self.len = config.seq_length
        self.slice_mask = P.StridedSlice().shard(((config.dp, 1, 1),))
            
    def construct(self, input_ids, mask_ids_input, input_position=None, attention_mask=None):
        #tokens = input_ids[:, :-1]
        tokens = self.slice(input_ids, (0, 0), (self.batch_size, -1), (1, 1))
        
        input_position = self.slice(input_position, (0, 0), (self.batch_size, self.len), (1, 1))
        attention_mask = self.slice_mask(attention_mask, (0, 0, 0),
                                         (self.batch_size, self.len, self.len),
                                         (1, 1, 1))
            
        input_mask = F.cast(self.not_equal(tokens, self.eos_token), mstype.float32)
        logits = self.network(tokens, input_mask, input_position, attention_mask)
        #labels = input_ids[:, 1:]
        labels = self.slice(input_ids, (0, 1), (self.batch_size, self.len + 1), (1, 1))
        output = self.loss(logits, labels, mask_ids_input)
        return output

class AttentionMask(nn.Cell):
    """
    Get the attention matrix for self-attention module
    Args:
        seq_length: the pre-defined sequence length
    Inputs:
        input_mask: the mask indicating whether each position is a valid input
    Returns:
        attention_mask: the attention mask matrix with shape (batch_size, seq_length, seq_length)
    """

    def __init__(self, seq_length):
        super(AttentionMask, self).__init__()
        self.reshape = P.Reshape()
        self.mul = P.BatchMatMul().shard(
            ((1, 1, 1), (1, 1, 1)))
        self.expand_dim = P.ExpandDims().shard(((1, 1),))
        ones = np.ones(shape=(seq_length, seq_length))
        self.lower_triangle_mask = Tensor(np.tril(ones), mstype.float32)
        self.multiply = P.Mul().shard(((1, 1, 1), (1, 1, 1)))

    def construct(self, input_mask):
        input_shape = P.Shape()(input_mask)
        shape_right = (input_shape[0], 1, input_shape[1])
        shape_left = input_shape + (1,)
        mask_left = self.reshape(input_mask, shape_left)
        mask_right = self.reshape(input_mask, shape_right)
        attention_mask = self.mul(mask_left, mask_right)
        lower_triangle = self.expand_dim(self.lower_triangle_mask, 0)
        attention_mask = self.multiply(attention_mask, lower_triangle)
        return attention_mask


class EvalNet(nn.Cell):
    """
    PanguAlpha evaluation net
    Args:
        backbone: backbone network of PanguAlpha
        generate: enable generate mode
    Inputs:
        input_ids: the tokenized inpus
        current_index: the index of current token
        init_reset: whether reset saved states
    Returns:
        outputs: Tensor, corresponding output for different tasks
    """

    def __init__(self, backbone, generate=False, pad_token=6, seq_length=1024):
        super(EvalNet, self).__init__(auto_prefix=False)
        self.backbone = backbone
        self.pad_token = pad_token
        self.argmax = P.Argmax()
        self.generate = generate
        self.topk = P.TopK(sorted=True).shard(((1, 1),))
        self.gather = P.GatherV2().shard(((1, 1), (1,)))
        self.log_softmax = P.LogSoftmax().shard(((1, 1, 1),))
        self.get_attention_mask = AttentionMask(seq_length)

    def construct(self, input_ids, current_index, init_reset=True, batch_valid_length=None):
        """evaluation net"""
        input_mask = F.cast(F.not_equal(input_ids, self.pad_token), mstype.float32)
        bs, seq_length = F.shape(input_ids)
        attention_mask = self.get_attention_mask(input_mask)
        input_position = F.tuple_to_array(F.make_range(seq_length))
        input_position = P.Tile()(input_position, (bs, 1))
        logits = self.backbone(input_ids, input_mask, input_position, attention_mask,
                               init_reset, batch_valid_length)
        index = current_index.view(1,)
        logits = self.gather(logits, index, 0)
        logits = logits.view(bs, 1, -1)
        log_probs = self.log_softmax(logits)
        return log_probs
