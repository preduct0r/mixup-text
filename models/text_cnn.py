import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np

embed_size = 300
kernel_size = [3, 4, 5]
num_channels = 100


class TextCNN(nn.Module):
    def __init__(self, vocab_size, sequence_len, num_class, word_embeddings=None, fine_tune=True, dropout=0.5):
        super(TextCNN, self).__init__()

        # Embedding Layer
        self.embeddings = nn.Embedding(vocab_size, embed_size)
        self.sequence_len = sequence_len
        if word_embeddings is not None:
            self.embeddings.weight = nn.Parameter(word_embeddings, requires_grad=fine_tune)

        # Conv layers
        self.convs = nn.ModuleList([nn.Conv2d(1, num_channels, [k, embed_size]) for k in kernel_size])

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_channels * len(kernel_size), num_class)

    def forward(self, x):
        # (batch, seq_len, embed)
        x = self.embeddings(x).permute(1, 0, 2)
        # (batch, channel, seq_len, embed)
        x = torch.unsqueeze(x, 1)

        # (batch, channel, seq_len-k+1)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]

        # (batch, channel)
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]

        # (batch, #filters * channel)
        x = torch.cat(x, 1)

        x = self.dropout(x)

        # (batch, #class)
        x = self.fc(x)
        return x

    def _forward_dense(self, x):
        x = self.embeddings(x).permute(1, 0, 2)
        x = torch.unsqueeze(x, 1)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x = torch.cat(x, 1)
        return x

    @staticmethod
    def mix_embed_nonlinear(x1, x2, lam):
        # x.shape: (batch, seq_len, embed)
        embed = x1.shape[2]
        stride = int(round(embed * (1 - lam)))
        mixed_x = x1
        aug_type = np.random.randint(2)
        if aug_type == 0:
            mixed_x[:, :, :stride] = x2[:, :, :stride]
        else:
            mixed_x[:, :, embed-stride:] = x2[:, :, embed-stride:]
        return mixed_x

    def forward_mix_embed(self, x1, x2, lam):
        # (seq_len, batch) -> (batch, seq_len, embed)
        x1 = self.embeddings(x1).permute(1, 0, 2)
        x2 = self.embeddings(x2).permute(1, 0, 2)
        x = lam * x1 + (1.0-lam) * x2
        # x = self.mix_embed_nonlinear(x1, x2, lam)

        x = torch.unsqueeze(x, 1)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x = torch.cat(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

    def forward_mix_sent(self, x1, x2, lam):
        y1 = self.forward(x1)
        y2 = self.forward(x2)
        y = lam * y1 + (1.0-lam) * y2
        return y

    def forward_mix_encoder(self, x1, x2, lam):
        y1 = self._forward_dense(x1)
        y2 = self._forward_dense(x2)
        y = lam * y1 + (1.0-lam) * y2
        y = self.fc(y)
        return y

    def _init_embeddings_and_vocab(self, embeddings_file: str, vocab_file: str,
                                   embeddings_dim: int) -> Tuple[np.array, dict]:
        if embeddings_file is None or vocab_file is None:
            logger.info("Embeddings files are not specified. Perform random initialization of embeddings...")
            embeddings = np.random.uniform(low=-0.2, high=0.2, size=(2, embeddings_dim))
            vocab = {'<PAD>': 0, '<UNK>': 1}
        else:
            embeddings = self._load_embeddings(embeddings_file)
            vocab = self._load_word2vec_vocabulary(vocab_file)
        return embeddings, vocab


    def _load_embeddings(self, embeddings_file: str) -> np.array:
        try:
            embeddings = np.load(embeddings_file).T
        except FileNotFoundError as e:
            message = "Embeddings file doesn't exist: {}".format(e)
            logger.error(message)
            raise FileNotFoundError(message)
        except ValueError as e:
            message = "Embeddings file couldn't load: {}".format(e)
            logger.error(message)
            raise ValueError(message)
        return embeddings

