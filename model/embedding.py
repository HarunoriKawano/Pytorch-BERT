import torch
from torch import nn


class BERTEmbedding(nn.Module):
    """Embedding module for BERT

    Convert the word ID and location information of the sentence into an embedding vector.
    Create vectors from word content, sentence position, and word position.

    Attributes:
        device (torch.device): using device(cuda:0 or cpu)
        word_embedding (nn.Embedding): Convert word ID to word vector.
        position_embedding (nn.Embedding): Convert location tensor to vector.
        token_type_embedding (nn.Embedding): Converting text information into vectors.
        layer_norm (nn.LayerNorm): layer normalization
        dropout (nn.Dropout): dropout
    """

    def __init__(self, vocab_size=30522, position_embedding_size=512, type_vocab_size=2,
                 feature_size=768, dropout_rate=0.1, device=torch.device('cpu')):
        super(BERTEmbedding, self).__init__()
        self.device = device

        self.word_embedding = nn.Embedding(
            vocab_size, feature_size, padding_idx=0
        )

        self.position_embedding = nn.Embedding(
            position_embedding_size, feature_size
        )

        self.token_type_embedding = nn.Embedding(
            type_vocab_size, feature_size
        )

        self.layer_norm = nn.LayerNorm(feature_size)

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, input_ids, token_type_ids=None):
        """
        Args:
            input_ids (torch.LongTensor(batch_size, seq_len)): A list of word IDs in a sentence.
            token_type_ids (torch.LongTensor(batch_size, seq_len)): An ID indicating how many sentences each word in input_ids is.

        Returns:
            torch.FloatTensor(batch_size, seq_len, feature_size): feature values after word embedding
        """
        seq_len = input_ids.size(1)

        # Convert word ID to word vector.
        # torch.Size([batch_size, max_sentence_len, 768])
        word_embeddings = self.word_embedding(input_ids)

        # When token_type_ids is None, all words are used as the first sentence.
        # torch.Size([batch_size, max_sentence_len, 768])
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        token_type_embeddings = self.token_type_embedding(token_type_ids)

        position_ids = torch.arange(seq_len, dtype=torch.long, device=self.device)
        # if seq_length == (2, 3): position_embeddings == torch.tensor([0, 1, 2], [0, 1, 2])
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        # Create a 768-dimensional tensor from position_ids.
        # torch.Size([batch_size, max_sentence_len, 768])
        position_embeddings = self.position_embedding(position_ids)

        # torch.Size([batch_size, max_sentence_len, 768])
        embeddings = word_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings
