import math
import torch
from torch import nn
import torch.nn.functional as F


class AttentionLayer(nn.Module):

    def __init__(self, feature_size=512, head_num=8):
        super(AttentionLayer, self).__init__()
        self.self_attention = MultiHeadAttention(feature_size=feature_size, head_num=head_num, dropout_rate=0.1)
        self.feed_forward = AttentionFeedForward(feature_size=feature_size, dropout_rate=0.1)
        self.layer_norm = nn.LayerNorm(feature_size)

    def forward(self, inputs, mask=None):
        q = k = v = inputs
        attention = self.self_attention(q, k, v, mask=mask)
        attention = self.feed_forward(attention)
        out = self.layer_norm(attention + inputs)

        return out


class AttentionFeedForward(nn.Module):
    def __init__(self, feature_size=512, dropout_rate=0.1):
        super(AttentionFeedForward, self).__init__()

        self.linear = nn.Linear(feature_size, feature_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, inputs):
        out = self.linear(inputs)
        out = self.dropout(out)
        return out


class MultiHeadAttention(nn.Module):
    """multi-head attention module

    Performs multi-headed attention processing for q, k, and v specified in the argument.

    Attributes:
        q_linear (torch.nn.Linear): Full connected layer for argument q
        v_linear (torch.nn.Linear): Full connected layer for argument v
        k_linear (torch.nn.Linear): Full connected layer for argument k
        dropout (torch.nn.Dropout): Dropout layer for weights of attention processing
        out_linear (torch.nn.Linear): Full connected layer for output
        head_num (int): Number of multi-head
        feature_size (int): Number of dimension of　q, v, and k features
        head_dim (int): Number of dimension of a single head

    """

    def __init__(self, feature_size=512, head_num=8, dropout_rate=0.1):
        super(MultiHeadAttention, self).__init__()
        self.q_linear = nn.Linear(feature_size, feature_size).float()
        self.v_linear = nn.Linear(feature_size, feature_size).float()
        self.k_linear = nn.Linear(feature_size, feature_size).float()
        self.dropout = nn.Dropout(dropout_rate)

        self.out_linear = nn.Linear(feature_size, feature_size).float()
        self.head_num = head_num
        self.attention_dim = feature_size
        self.head_dim = feature_size // head_num

    def forward(self, q, k, v, mask=None):
        """multi-head attention processing

        Calculate the similarity of the arguments q and k, and multiply them by argument v.
        In normal-attention, the values of k and v will be the same, and in self-attention, q, k, and v will all be the same.
        Args:
            q (torch.FloatTensor(batch_num, data_len, feature_num)): Query to be compared for similarity.
            k (torch.FloatTensor(batch_num, data_len, feature_num)): Key for calculating the similarity.
            v (torch.FloatTensor(batch_num, data_len, feature_num)): Value to be multiplied by similarity.
            mask (torch.LongTensor(batch_num, data_len)): Mask for features not considered. Classify by 0 and 1.

        Returns:
            torch.FloatTensor(batch_num, data_len, feature_num): Value through the attention process.
        """

        # (batch_num, data_len, feature_num) -> (batch_num, head_num, data_len, feature_num/head_num)
        q = self._split_head(self.q_linear(q))
        k = self._split_head(self.k_linear(k))
        v = self._split_head(self.v_linear(v))

        # Scaling with multi-heads in mind
        q *= self.head_dim ** -0.5

        weights = torch.matmul(q, torch.transpose(k, 2, 3)) / math.sqrt(self.head_dim)

        if mask is not None:
            weights = weights.masked_fill(mask == 0, -1e9)
        weights = self.dropout(weights)

        normalized_weights = F.softmax(weights, dim=-1)

        # (batch_num, head_num, data_len, feature_num/head_num) -> (batch_num, data_len, feature_num)
        output = self._combine_head(torch.matmul(normalized_weights, v))
        return self.out_linear(output)

    def _split_head(self, inputs):
        batch_size, length, _ = inputs.shape
        split_inputs = torch.reshape(inputs, (batch_size, length, self.head_num, self.head_dim))
        return torch.transpose(split_inputs, 1, 2)

    def _combine_head(self, inputs):
        batch_size, _, length, _ = inputs.shape
        combine_inputs = torch.reshape(torch.transpose(inputs, 1, 2), (batch_size, length, self.attention_dim))
        return combine_inputs
