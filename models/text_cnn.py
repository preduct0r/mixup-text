import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import os

embed_size = 300
kernel_size = [3, 4, 5]
num_channels = 100

model__embeddings = '/home/den/Documents/word_embeddings/best_gensim_300_20/gensim_cbow_300d_20mc_30e_12t_0.025a_0.015ma_tf1_lk4_bln.npy'
model__vocabulary = '/home/den/Documents/word_embeddings/best_gensim_300_20/gensim_cbow_300d_20mc_30e_12t_0.025a_0.015ma_tf1_lk4_bln.dic'
model__embeddings_dim = 300
num_filters = 128
dropout_keep_prob = 0.8
l2_reg_lambda = 0.01
learning_rate = 2e-3
batch_size = 64
evaluate_every = -1
max_len = 64
max_epoch_num = 10
label_smoothing = True
filter_sizes = [1,1,1,2,2,2]


class TextCNN(nn.Module):
    def __init__(self, vocab_size, sequence_len, num_class, word_embeddings=None, fine_tune=True, dropout=0.5):
        super(TextCNN, self).__init__()

        # Embedding Layer
        self.sequence_len = sequence_len

        # if word_embeddings is not None:
        #     self.embeddings.weight = nn.Parameter(word_embeddings, requires_grad=fine_tune)

        embeddings_weight, self.vocabulary = self._init_embeddings_and_vocab(model__embeddings,
                                        model__vocabulary,
                                        model__embeddings_dim)

        self.embeddings = nn.Embedding(len(self.vocabulary), embed_size)
        self.embeddings.weight = nn.Parameter(torch.FloatTensor(embeddings_weight), requires_grad=fine_tune)



        # Conv layers
        # self.convs = nn.ModuleList([nn.Conv2d(1, num_channels, [k, embed_size]) for k in kernel_size])
        self.convs = nn.ModuleList([nn.Conv2d(in_channels=1, out_channels=num_filters,
                                              kernel_size=(k, model__embeddings_dim), bias=True) for k
                                    in filter_sizes])
        for conv in self.convs:
            torch.nn.init.trunc_normal_(conv.weight, std=0.1)

        self.dropout = nn.Dropout(p=dropout_keep_prob)
        self.fc = nn.Linear(num_channels * len(kernel_size), num_class)


    def forward(self, x):
        emb = self.embeddings(x)

        emb = torch.unsqueeze(emb, -1).permute(0, 3, 1, 2)

        convs = [F.relu(conv(emb)) for conv in self.convs]

        pools = [F.max_pool2d(input=conv, kernel_size=(max_len - k + 1, 1)) \
                 for k, conv in zip(filter_sizes, convs)]

        cat = torch.squeeze(self.dropout(torch.cat(pools, dim=1)))

        out = self.linear(cat)

        return out


    def _forward_dense(self, x):
        self.forward(x)



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
        x1 = self.embeddings(x1)
        x2 = self.embeddings(x2)
        emb = lam * x1 + (1.0-lam) * x2
        # x = self.mix_embed_nonlinear(x1, x2, lam)

        # emb = self.embeddings(x)

        emb = torch.unsqueeze(emb, -1).permute(0, 3, 1, 2)

        convs = [F.relu(conv(emb)) for conv in self.convs]

        pools = [F.max_pool2d(input=conv, kernel_size=(max_len - k + 1, 1)) \
                 for k, conv in zip(filter_sizes, convs)]

        cat = torch.squeeze(self.dropout(torch.cat(pools, dim=1)))

        out = self.linear(cat)

        return out

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
                                   embeddings_dim: int):
        if embeddings_file is None or vocab_file is None:
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
            raise FileNotFoundError(message)
        except ValueError as e:
            message = "Embeddings file couldn't load: {}".format(e)
            raise ValueError(message)
        return embeddings

    def _load_word2vec_vocabulary(self, vocab_file: str) -> dict:
        vocab = {}
        # Added a check to not wrap the whole "with ... as" block with try/except
        if not os.path.exists(vocab_file):
            message = "Vocabulary file doesn't exist"
            raise FileNotFoundError(message)
        with open(vocab_file, encoding="utf-8") as vfile:
            unk_num = 0
            for line_num, line in enumerate(vfile):
                line = line.strip()
                try:
                    cur_word, cur_num_of_occur = line.split()
                except ValueError:
                    cur_word = "unkword_%i" % unk_num
                    unk_num += 1
                if cur_word in vocab.keys():
                    cur_word = "unkword_%i" % unk_num
                    unk_num += 1
                vocab[cur_word] = line_num
        return vocab


    def _add_unknown_words_to_model(self, train_data_texts):
        # logger.info("Adding unknown words to model...")
        word_list = [w for text in train_data_texts for w in text]
        word_set = set(word_list)
        initial_vocab_len = len(self.vocabulary)
        vocab_set = set(self.vocabulary.keys())
        new_words_set = set.difference(word_set, vocab_set)
        number_added_words = len(new_words_set)
        new_words_vocab = dict(zip(new_words_set, range(initial_vocab_len, initial_vocab_len + number_added_words)))
        self.vocabulary.update(new_words_vocab)
        self.embeddings = np.concatenate(
            (self.embeddings,
             np.random.uniform(
                 low=-0.2, high=0.2, size=(number_added_words, self.properties.model__embeddings_dim)
             )),
            axis=0
        )



    def _digitize_sents(self, texts) -> np.array:
        if not isinstance(texts, list):
            message = "texts must be a list"
            raise TypeError(message)
        digitized_texts = []
        unk_word_number = self.vocabulary['<UNK>']
        pad_token_number = self.vocabulary['<PAD>']
        for sample in texts:
            if not sample:

                continue
            digitized_sample = []
            for word in sample[-self.properties.model__max_len:]:
                try:
                    idx = self.vocabulary[word]
                except KeyError:
                    idx = unk_word_number
                digitized_sample.append(idx)
            if len(digitized_sample) < self.properties.model__max_len:
                digitized_sample = [pad_token_number] * (
                        self.properties.model__max_len - len(digitized_sample)
                ) + digitized_sample
            digitized_texts.append(digitized_sample)
        if not digitized_texts:
            message = "No texts found after converting words to integers"
            raise ValueError(message)
        return np.array(digitized_texts, dtype=np.int32)


    def _digitize_labels(self, labels) -> np.array:
        """
        :param labels: list of target labels
        :return: 2d array where rows are one-hot target labels
        :raises:
            ValueError: if a label from labels isn't in self.labels
        """
        inverse_labels = {l: i for i, l in enumerate(self.labels)}

        if self.properties.model__label_smoothing:
            alpha = 0.1  # label smoothing factor
            labels_dig = np.ones((len(labels), len(inverse_labels))) * (alpha / len(self.labels))
        else:
            labels_dig = np.zeros((len(labels), len(inverse_labels)), dtype=np.int16)

        for i, label in enumerate(labels):
            try:
                if self.properties.model__label_smoothing:
                    labels_dig[i, inverse_labels[label]] += 1 - alpha
                else:
                    labels_dig[i, inverse_labels[label]] += 1
            except KeyError:
                message = "Unknown label found while one-hot encoding labels: {}" \
                    .format(label)
                raise KeyError(message)
        return labels_dig



