# -*- coding: utf-8 -*-

# This code from https://github.com/phohenecker/pytorch-transformer #


import typing

import numpy as np
import torch

from torch import nn
from torch.nn import init
from torch.nn import functional


import numbers



class EncDecBase(object):
    """A base class that implements common functionality of the encoder and decoder parts of the Transformer model."""

    def __init__(
            self,
            num_layers: int,
            num_heads: int,
            dim_model: int,
            dim_keys: int,
            dim_values: int,
            residual_dropout: numbers.Real,
            attention_dropout: numbers.Real,
            pad_index: int
    ):
        """Creates a new instance of ``EncDecBase``.

        Args:
            num_layers (int): The number of to use.
            num_heads (int): The number of attention heads to use.
            dim_model (int): The dimension to use for all layers. This is called d_model, in the paper.
            dim_keys (int): The size of the keys provided to the attention mechanism. This is called d_k, in the paper.
            dim_values (int): The size of the values provided to the attention mechanism. This is called d_v, in the
                paper.
            residual_dropout (numbers.Real): The dropout probability for residual connections (before they are added to
                the the sublayer output).
            attention_dropout (numbers.Real): The dropout probability for values provided by the attention mechanism.
            pad_index (int): The index that indicates a padding token in the input sequence.
        """
        super().__init__()

        # define attributes
        self._attention_dropout = None
        self._dim_keys = None
        self._dim_model = None
        self._dim_values = None
        self._num_heads = None
        self._num_layers = None
        self._pad_index = None
        self._residual_dropout = None

        # specify properties
        self.attention_dropout = attention_dropout
        self.dim_keys = dim_keys
        self.dim_model = dim_model
        self.dim_values = dim_values
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.pad_index = pad_index
        self.residual_dropout = residual_dropout

    #  PROPERTIES  #####################################################################################################

    @property
    def attention_dropout(self) -> float:
        """float: The dropout probability for residual connections (before they are added to the the sublayer output).
        """
        return self._attention_dropout

    @attention_dropout.setter
    def attention_dropout(self, attention_dropout: numbers.Real):
        self._sanitize_probability("attention_dropout", attention_dropout)
        self._attention_dropout = float(attention_dropout)

    @property
    def dim_keys(self) -> int:
        """int: The size of the keys provided to the attention mechanism.

        This value is called d_k, in "Attention Is All You Need".
        """
        return self._dim_keys

    @dim_keys.setter
    def dim_keys(self, dim_keys: int) -> None:
        self._sanitize_pos_int("dim_keys", dim_keys)
        self._dim_keys = dim_keys

    @property
    def dim_model(self) -> int:
        """int: The dimension to use for all layers.

        This value is called d_model, in "Attention Is All You Need".
        """
        return self._dim_model

    @dim_model.setter
    def dim_model(self, dim_model: int) -> None:
        self._sanitize_pos_int("dim_model", dim_model)
        self._dim_model = dim_model

    @property
    def dim_values(self) -> int:
        """int: The size of the values provided to the attention mechanism.

        This value is called d_v, in "Attention Is All You Need".
        """
        return self._dim_values

    @dim_values.setter
    def dim_values(self, dim_values: int) -> None:
        self._sanitize_pos_int("dim_values", dim_values)
        self._dim_values = dim_values

    @property
    def num_heads(self) -> int:
        """int: The number of attention heads used by the implemented module."""
        return self._num_heads

    @num_heads.setter
    def num_heads(self, num_heads: int) -> None:
        self._sanitize_pos_int("num_heads", num_heads)
        self._num_heads = num_heads

    @property
    def num_layers(self) -> int:
        """int: The number of layers used by the implemented module."""
        return self._num_layers

    @num_layers.setter
    def num_layers(self, num_layers: int) -> None:
        self._sanitize_pos_int("num_layers", num_layers)
        self._num_layers = num_layers

    @property
    def pad_index(self) -> int:
        """int: The index that indicates a padding token in the input sequence."""
        return self._pad_index

    @pad_index.setter
    def pad_index(self, pad_index: int) -> None:
        if not isinstance(pad_index, int):
            raise TypeError("<pad_index> has to be an integer!")
        if pad_index < 0:
            raise ValueError("<pad_index> has to be non-negative!")
        self._pad_index = pad_index

    @property
    def residual_dropout(self) -> float:
        """float: The dropout probability for values provided by the attention mechanism."""
        return self._residual_dropout

    @residual_dropout.setter
    def residual_dropout(self, residual_dropout: numbers.Real):
        self._sanitize_probability("residual_dropout", residual_dropout)
        self._residual_dropout = float(residual_dropout)

    #  METHODS  ########################################################################################################

    @staticmethod
    def _sanitize_pos_int(arg_name: str, arg_value) -> None:
        """Ensures that the provided arg is a positive integer.

        Args:
            arg_name (str): The name of the arg being sanitized.
            arg_value: The value being sanitized.

        Raises:
            TypeError: If ``arg_value`` is not an ``int``.
            ValueError: If ``arg_value`` is not a positive number.
        """
        if not isinstance(arg_value, int):
            raise TypeError("<{}> has to be an integer!".format(arg_name))
        if arg_value < 1:
            raise ValueError("<{}> has to be > 0!".format(arg_name))

    @staticmethod
    def _sanitize_probability(arg_name: str, arg_value):
        """Ensures that the provided arg is a probability.

        Args:
            arg_name (str): The name of the arg being sanitized.
            arg_value: The value being sanitized.

        Raises:
            TypeError: If ``arg_value`` is not a ``numbers.Real``.
            ValueError: If ``arg_value`` is not in [0, 1].
        """
        if not isinstance(arg_value, numbers.Real):
            raise TypeError("<{}> has to be a real number!".format(arg_name))
        if arg_value < 0 or float(arg_value) > 1:
            raise ValueError("<{}> has to be in [0, 1]!".format(arg_name))


class Normalization(nn.Module):
    """A normalization layer."""

    def __init__(self, eps: numbers.Real=1e-15):
        """Creates a new instance of ``Normalization``.

        Args:
            eps (numbers.Real, optional): A tiny number to be added to the standard deviation before re-scaling the
                centered values. This prevents divide-by-0 errors. By default, this is set to ``1e-15``.
        """
        super().__init__()

        self._eps = None
        self.eps = float(eps)

    #  PROPERTIES  #####################################################################################################

    @property
    def eps(self) -> float:
        """float: A tiny number that is added to the standard deviation before re-scaling the centered values.

        This prevents divide-by-0 errors. By default, this is set to ``1e-15``.
        """
        return self._eps

    @eps.setter
    def eps(self, eps: numbers.Real) -> None:
        if not isinstance(eps, numbers.Real):
            raise TypeError("<eps> has to be a real number!")
        self._eps = float(eps)

    #  METHODS  ########################################################################################################

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        """Runs the normalization layer.

        Args:
            x (torch.FloatTensor): A tensor to be normalized. To that end, ``x`` is interpreted as a batch of values
                where normalization is applied over the last of its dimensions.

        Returns:
            torch.FloatTensor: The normalized tensor.
        """
        mean = torch.mean(x, dim=-1, keepdim=True)
        std = torch.std(x, dim=-1, keepdim=True)

        return (x - mean) / (std + self._eps)


class FeedForwardLayer(nn.Module):
    """A sublayer that computes a 1-hidden-layer multi-layer perceptron for each token in a sequences."""

    def __init__(self, dim_model: int):
        """Creates a new instance of ``FeedForwardLayer``.

        Args:
             dim_model (int): The dimension of all tokens in the input sequence. This is called d_model, in the paper.
        """
        super().__init__()

        # sanitize args
        if not isinstance(dim_model, int):
            raise TypeError("<dim_model> has to be an integer!")
        if dim_model < 1:
            raise ValueError("<dim_model> has to be a positive number!")

        # store arg
        self._dim_model = dim_model

        # create layers
        self._layer_1 = nn.Conv1d(self._dim_model, self._dim_model, 1)
        self._layer_2 = nn.Conv1d(self._dim_model, self._dim_model, 1)

    #  PROPERTIES  #####################################################################################################

    @property
    def dim_model(self) -> int:
        """int: The dimension of all tokens in the input sequence.

        This is called d_model, in the paper.
        """
        return self._dim_model

    @property
    def layer_1(self) -> nn.Conv1d:
        """nn.Conv1d: The first linear layer (before the ReLU non-linearity is applied)."""
        return self._layer_1

    @property
    def layer_2(self) -> nn.Conv1d:
        """nn.Conv1d: The second linear layer."""
        return self._layer_2

    #  METHODS  ########################################################################################################

    def forward(self, sequence: torch.FloatTensor) -> torch.FloatTensor:
        """Runs the feed-forward layer.

        Args:
            sequence (torch.FloatTensor): The input sequence given as (batch_size x seq_len x dim_model)-tensor.

        Returns:
            torch.FloatTensor: The computed values as (batch_size x seq_len x dim_model)-tensor.
        """
        assert sequence.dim() == 3
        assert sequence.size(2) == self._dim_model

        sequence = functional.relu(self._layer_1(sequence.transpose(1, 2)))
        sequence = self._layer_2(sequence).transpose(1, 2)

        return sequence

    def reset_parameters(self):
        """Resets all trainable parameters of the module."""
        self._layer_1.reset_parameters()
        self._layer_2.reset_parameters()


class MultiHeadAttention(nn.Module):
    """A multi-head scaled dot-product attention mechanism as it is used in *Attention Is All You Need*."""

    def __init__(self, num_heads: int, dim_model: int, dim_keys: int, dim_values: int, dropout_rate: float):
        """Creates a new instance of ``MultiHeadAttention``.

        Notice:
            This constructor does not sanitize any parameters, which means that this has to be taken care of beforehand.

        Args:
            num_heads (int): The number of attention heads to use.
            dim_model (int): The dimension used for all layers in the model that the ``MultiHeadAttention`` belongs to.
            dim_keys (int): The target size to project keys to.
            dim_values (int): The target size to project values to.
            dropout_rate (float): The dropout probability to use.
        """
        super().__init__()

        # store all of the provided args
        self.dim_keys = dim_keys
        self.dim_model = dim_model
        self.dim_values = dim_values
        self.dropout_rate = dropout_rate
        self.num_heads = num_heads

        # create projections for inputs
        self.query_projection = nn.Parameter(torch.empty(self.num_heads, self.dim_model, self.dim_keys))
        self.key_projection = nn.Parameter(torch.empty(self.num_heads, self.dim_model, self.dim_keys))
        self.value_projection = nn.Parameter(torch.empty(self.num_heads, self.dim_model, self.dim_values))

        # create output projection
        self.output_projection = nn.Parameter(torch.empty(self.num_heads * self.dim_values, self.dim_model))

        # create softmax and dropout layers
        self.dropout = nn.Dropout(self.dropout_rate)
        self.softmax = nn.Softmax(dim=3)

        # initialize all parameters
        self.reset_parameters()

    #  METHODS  ########################################################################################################

    def _apply_attention(
            self,
            queries: torch.FloatTensor,
            keys: torch.FloatTensor,
            values: torch.FloatTensor,
            mask: typing.Optional[torch.ByteTensor]
    ) -> torch.Tensor:
        """The actual attention mechanism.

        Args:
            queries (torch.FloatTensor): The queries as (batch_size x num_heads x Q x dim_keys)-tensor.
            keys (torch.FloatTensor): The keys as (batch_size x num_heads x KV x dim_keys)-tensor.
            values (torch.FloatTensor): The values as (batch_size x num_heads x KV x dim_values)-tensor.
            mask (torch.ByteTensor): An optional binary mask that indicates which key-value pairs to consider for each
                of the queries. If provided, then this has to be a (batch_size x Q x KV)-tensor.

        Returns:
            torch.FloatTensor: The computed "attended" values as (batch_size x num_heads x Q x dim_values)-tensor. If
                the ``mask`` specifies that none of the key-value pairs shall be used for any of the queries, then the
                according attended value is set to ``0``.
        """
        # compute inputs to the softmax
        attn = queries.matmul(keys.transpose(2, 3)) / np.sqrt(self.dim_keys)  # compute (Q * K^T) / sqrt(d_k)
        # -> (batch_size x num_heads x Q x KV)

        # apply the mask (if provided)
        if mask is not None:

            # check whether the mask excludes all of the entries
            if mask.sum().item() == 0:
                return torch.zeros(queries.size())

            # expand mask to cover all heads
            mask = mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)

            # determine which token masks are all-0
            non_zero_parts = (mask.sum(dim=-1) != 0).unsqueeze(-1).expand(*mask.size())

            # remove the all-0 parts from the original mask
            mask = 1 - (1 - mask) * non_zero_parts

            # apply mask
            attn.masked_fill_(1 - mask, -np.inf)

            # compute attention scores
            attn = self.softmax(attn)

            # apply all-0 parts of the masks
            attn = attn * non_zero_parts.float()
        else:
            # compute attention scores
            attn = self.softmax(attn)

        # apply dropout
        attn = self.dropout(attn)

        # compute attended value
        return attn.matmul(values)  # -> (batch_size x num_heads x Q x dim_values)

    def _project_inputs(
            self,
            queries: torch.FloatTensor,
            keys: torch.FloatTensor,
            values: torch.FloatTensor
    ) -> typing.Tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor
    ]:
        """Projects all inputs provided to the attention mechanism to the needed sizes.

        This means that queries and keys are projected from ``dim_model`` to ``dim_keys``, and values from ``dim_model``
        to ``dim_values``.

        Args:
            queries (torch.FloatTensor): The queries as (batch_size x Q x dim_model)-tensor.
            keys (torch.FloatTensor): The keys as (batch_size x KV x dim_model)-tensor.
            values (torch.FloatTensor): The values as (batch_size x KV x dim_model)-tensor.

        Returns:
            tuple: A triple of ``FloatTensor``s, consisting of the projected queries, keys, and values.
        """
        # for each of the attention heads, project inputs to the needed dimensions
        queries = queries.unsqueeze(1).matmul(self.query_projection)  # -> (batch_size x num_heads x Q  x dim_keys)
        keys = keys.unsqueeze(1).matmul(self.key_projection)          # -> (batch_size x num_heads x KV x dim_keys)
        values = values.unsqueeze(1).matmul(self.value_projection)    # -> (batch_size x num_heads x KV x dim_values)

        return queries, keys, values

    def _project_output(self, attn_values: torch.FloatTensor) -> torch.FloatTensor:
        """Projects the "attended" values of all heads to the required output size.

        Args:
            attn_values (torch.FloatTensor): The attended values as (batch_size x num_heads x Q x dim_values)-tensor.

        Returns:
            torch.FloatTensor: The computed output as (batch_size x Q x dim_model)-tensor.
        """
        # concatenate the values retrieved from all heads
        batch_size = attn_values.size(0)
        num_queries = attn_values.size(2)
        attn_values = attn_values.transpose(1, 2).reshape(batch_size, num_queries, -1)
        # -> (batch_size x Q x (num_heads * dim_values))

        return attn_values.matmul(self.output_projection)  # -> (batch-size x Q x dim_model)

    def forward(
            self,
            queries: torch.FloatTensor,
            keys: torch.FloatTensor,
            values: torch.FloatTensor,
            mask: torch.ByteTensor=None
    ) -> torch.Tensor:
        """Runs the attention mechanism.

        Args:
            queries (torch.FloatTensor): The queries as (batch_size x Q x dim_model)-tensor.
            keys (torch.FloatTensor): The keys as (batch_size x KV x dim_model)-tensor.
            values (torch.FloatTensor): The values as (batch_size x KV x dim_model)-tensor.
            mask (torch.ByteTensor, optional): An optional binary mask that indicates which key-value pairs to consider
                for each of the queries. If provided, then this has to be a (batch_size x Q x KV)-tensor.

        Returns:
            torch.FloatTensor: The values computed by the attention mechanism as (batch_size x Q x dim_model)-tensor.
        """
        assert isinstance(queries, torch.FloatTensor) or isinstance(queries, torch.cuda.FloatTensor)
        assert isinstance(keys, torch.FloatTensor) or isinstance(keys, torch.cuda.FloatTensor)
        assert isinstance(values, torch.FloatTensor) or isinstance(values, torch.cuda.FloatTensor)
        assert queries.dim() == 3
        assert keys.dim() == 3
        assert values.dim() == 3
        assert queries.size(0) == keys.size(0)
        assert queries.size(0) == values.size(0)
        assert queries.size(2) == keys.size(2)
        assert queries.size(2) == values.size(2)
        assert keys.size(1) == values.size(1)
        if mask is not None:
            assert isinstance(mask, torch.ByteTensor) or isinstance(mask, torch.cuda.ByteTensor)
            assert mask.dim() == 3
            assert queries.size(0) == mask.size(0)
            assert queries.size(1) == mask.size(1)
            assert keys.size(1) == mask.size(2)

        # for each of the attention heads, project inputs to the needed dimensions
        queries, keys, values = self._project_inputs(queries, keys, values)

        # compute attention value
        attn_values = self._apply_attention(queries, keys, values, mask)

        # project retrieved values to needed dimensions
        return self._project_output(attn_values)

    def reset_parameters(self):
        """Resets all trainable parameters of the module."""
        init.xavier_normal_(self.query_projection)
        init.xavier_normal_(self.key_projection)
        init.xavier_normal_(self.value_projection)
        init.xavier_normal_(self.output_projection)



class Encoder(nn.Module, EncDecBase):
    """The encoder that is used in the Transformer model."""

    def __init__(self, *args, **kwargs):
        nn.Module.__init__(self)
        EncDecBase.__init__(self, *args, **kwargs)

        self._layers = nn.ModuleList([_EncoderLayer(self) for _ in range(self._num_layers)])

    #  METHODS  ########################################################################################################

    def forward(self, sequence: torch.FloatTensor, padding_mask: torch.ByteTensor=None) -> torch.FloatTensor:
        """Runs the encoder.

        Args:
            sequence (torch.FloatTensor): The input sequence as (batch-size x seq-len x dim-model)-tensor.
            padding_mask (torch.ByteTensor, optional):  Optionally, a padding mask as
                (batch-size x in-seq-len x in-seq-len)-tensor. To that end, ``1``s indicate those positions that are
                part of the according sequence, and ``0``s mark padding tokens.

        Returns:
            FloatTensor: The encoded sequence as (batch_size x seq_len x dim_model)-tensor.
        """
        assert sequence.dim() == 3
        assert sequence.size(2) == self._dim_model

        # apply all layers to the input
        for layer in self._layers:
            sequence = layer(sequence, padding_mask)

        # provide the final sequence
        return sequence

    def reset_parameters(self) -> None:
        for l in self._layers:
            l.reset_parameters()


class _EncoderLayer(nn.Module):
    """One layer of the encoder.

    Attributes:
        attn: (:class:`mha.MultiHeadAttention`): The attention mechanism that is used to read the input sequence.
        feed_forward (:class:`ffl.FeedForwardLayer`): The feed-forward layer on top of the attention mechanism.
    """

    def __init__(self, parent: Encoder):
        """Creates a new instance of ``_EncoderLayer``.

        Args:
            parent (Encoder): The encoder that the layers is created for.
        """
        super().__init__()
        self.attn = MultiHeadAttention(
                parent.num_heads,
                parent.dim_model,
                parent.dim_keys,
                parent.dim_values,
                parent.attention_dropout
        )
        self.feed_forward = FeedForwardLayer(parent.dim_model)
        self.norm = Normalization()
        self.dropout = nn.Dropout(parent.residual_dropout)

    #  METHODS  ########################################################################################################

    def forward(self, sequence: torch.FloatTensor, padding_mask: torch.ByteTensor) -> torch.FloatTensor:
        """Runs the layer.

        Args:
            sequence (torch.FloatTensor): The input sequence as (batch_size x seq_len x dim_model)-tensor.
            padding_mask (torch.ByteTensor): The padding mask as (batch_size x seq_len x seq_len)-tensor or ``None`` if
                no mask is used.

        Returns:
            torch.FloatTensor: The encoded sequence as (batch_size x seq_len x dim_model)-tensor.
        """
        # compute attention sub-layer
        sequence = self.norm(self.dropout(self.attn(sequence, sequence, sequence, mask=padding_mask)) + sequence)

        # compute feed-forward sub-layer
        sequence = self.norm(self.dropout(self.feed_forward(sequence)) + sequence)

        return sequence

    def reset_parameters(self) -> None:
        """Resets all trainable parameters of the module."""
        self.attn.reset_parameters()
        self.feed_forward.reset_parameters()
