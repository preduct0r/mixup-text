import logging
import math
import os
from operator import itemgetter
import sys
from typing import List, Tuple, Iterator, Optional

# from . import optimize
from .classifier import Classifier, AlgorithmProps
from .utils import calculate_confidence_threshold
sys.path.append('..')
from classifier_trainer.app.exceptions import NotIntegerException, ThresholdCouldNotCountException
from classifier_trainer.app.settings import BASE_DIR
from common import custom_types
from common.utils import flatten, reverse_flatten

import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torch.nn import Sequential
from torch.nn import functional as F
from torch import nn
import torch.cuda as cuda

import gensim
import numpy as np
import pandas as pd
from copy import deepcopy
import random
import time
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix, recall_score
# from memory_profiler import profile


def set_seed(seed):
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    # torch.set_deterministic(True)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

set_seed(43)

logger = logging.getLogger(__name__)

# from sampler import BalancedBatchSampler


class CNNProps(AlgorithmProps):
    model__embeddings = custom_types.String()
    model__vocabulary = custom_types.String()
    model__embeddings_dim = custom_types.Integer(default=100)
    model__filter_sizes = custom_types.Collection(list, default=(1, 1, 1, 2, 2, 2))
    model__num_filters = custom_types.Integer(default=128)
    model__dropout_keep_prob = custom_types.Float(default=0.8)
    model__l2_reg_lambda = custom_types.Float(default=0.0)
    model__learning_rate = custom_types.Float(default=2e-3)
    model__batch_size = custom_types.Integer(default=64)
    model__evaluate_every = custom_types.Integer(default=300)
    model__max_len = custom_types.Integer(default=64)
    model__max_epoch_num = custom_types.Integer(default=10)
    model__label_smoothing = custom_types.Boolean(default=True)
    optimization__save_ckpt = custom_types.Boolean(default=True)
    optimization__freeze_graph = custom_types.Boolean(default=True)
    optimization__quantize_weights = custom_types.Boolean(default=False)

    def to_internal_value(self, config):
        return flatten(config)

    def to_external_value(self, data):
        return reverse_flatten({
            "algorithm": "CNN",
            "model__embeddings_dim": data.get("model__embeddings_dim"),
            "model__filter_sizes": data.get("model__filter_sizes"),
            "model__num_filters": data.get("model__num_filters"),
            "model__l2_reg_lambda": data.get("model__l2_reg_lambda"),
            "model__max_len": data.get("model__max_len"),
            "threshold__confidence_threshold": data.get("threshold__confidence_threshold"),
        })

    class Meta:
        fields = [
            "model__embeddings",
            "model__vocabulary",
            "model__embeddings_dim",
            "model__filter_sizes",
            "model__num_filters",
            "model__dropout_keep_prob",
            "model__l2_reg_lambda",
            "model__learning_rate",
            "model__batch_size",
            "model__evaluate_every",
            "model__max_len",
            "model__max_epoch_num",
            "threshold__confidence_threshold",
            "optimization__save_ckpt",
            "optimization__freeze_graph",
            "optimization__quantize_weights"
        ]

def create_emb_layer(embeddings, non_trainable=False):
    # emb_layer = nn.Embedding(embeddings.shape[0], embeddings.shape[1])
    # emb_layer.weight.data.copy_(torch.from_numpy(embeddings))
    # if non_trainable:
    #     emb_layer.weight.requires_grad = False

    emb_layer = nn.Embedding.from_pretrained(torch.FloatTensor(embeddings))
    emb_layer.weight.requires_grad = not non_trainable
    return emb_layer




class My_Dataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return torch.LongTensor(self.x[idx, :]), torch.FloatTensor(self.y[idx, :])



class CNN(nn.Module):
    def __init__(self, conf_file=None):
        super(CNN, self).__init__()
        if conf_file is None:
            conf_file = os.path.join(BASE_DIR, "engine", "setting", "cnn.yaml")
        self.properties = CNNProps(
            path=conf_file
        )
        self.labels = []
        self.vocabulary = []
        self.embeddings = None

        self.sess = None
        self.train_optimizer = None
        self.global_step = None
        self.best_step = None
        self.loss = None
        self.accuracy = None
        self.epoch = None

        self.input_x = None
        self.input_y = None
        self.dropout_keep_prob = None
        self.weight_dacay = 1e-4

        self.embeddings, self.vocabulary = self._init_embeddings_and_vocab(self.properties.model__embeddings,
                                                                           self.properties.model__vocabulary,
                                                                           self.properties.model__embeddings_dim)
        self.properties.model__embeddings_dim = np.shape(self.embeddings)[1]

        if not (self.properties.optimization__save_ckpt
                or self.properties.optimization__freeze_graph):
            message = "'save_ckpt', 'freeze' and 'quantize_weights' parameters are False. Please check {}"\
                .format(conf_file)
            logger.error(message)
            raise ValueError(message)


        #Tensorflow and PyTorch typically order their batches of images
        # (TF: NHWC, PyTorch: NCHW) and weights (TF: HWCiCo, PyTorch CoCiHW)
        self.convs = nn.ModuleList([nn.Conv2d(in_channels=1, out_channels=self.properties.model__num_filters,
                        kernel_size=(k, self.properties.model__embeddings_dim), bias=True) for k in self.properties.model__filter_sizes])

        # проверка влияния инициализации на результат
        for conv in self.convs:
            torch.nn.init.trunc_normal_(conv.weight, std=0.1)

        self.dropout = nn.Dropout(p=self.properties.model__dropout_keep_prob)




    def forward(self, x, hidden=None):

        emb = self.emb_layer(x)

        emb = torch.unsqueeze(emb, -1).permute(0,3,1,2)

        convs = [F.relu(conv(emb)) for conv in self.convs]

        pools = [F.max_pool2d(input=conv, kernel_size=(self.properties.model__max_len - k + 1, 1)) \
                 for k,conv in zip(self.properties.model__filter_sizes, convs)]

        cat = torch.squeeze(self.dropout(torch.cat(pools, dim=1)))

        out = self.linear(cat)

        return out

    # @profile
    def validate(self, data) -> Tuple[float, int]:
        self.labels = data.return_unique_labels()
        # создаем linear_layer когда знаем сколько уникальных классов в выборке
        self.linear = nn.Linear(in_features=768,out_features=len(self.labels))

        # проверка влияния инициализации на результат
        torch.nn.init.trunc_normal_(self.linear.weight, std=0.1)

        training_samples, target_labels = data.get_train_part()

        test_samples, target_test_labels = data.get_test_part()

        if self.properties.model__evaluate_every == -1:
            self.properties.model__evaluate_every = max(1, int(len(target_labels) / 50))
        self._add_unknown_words_to_model(training_samples)

        # создаем emb_layer только после добавления всех слов в vocabulary
        self.emb_layer = create_emb_layer(self.embeddings, False)
        training_samples = self._digitize_sents(training_samples)
        target_labels = self._digitize_labels(target_labels)

        test_samples = self._digitize_sents(test_samples)
        target_test_labels = self._digitize_labels(target_test_labels)

        accuracy, self.best_step = self._train_classifier_with_test(
            training_samples, test_samples, target_labels, target_test_labels
        )

        return accuracy, self.best_step

    def fit(self, data):
        self.labels = data.return_unique_labels()
        # создаем linear_layer когда знаем сколько уникальных классов в выборке
        self.linear = nn.Linear(in_features=768, out_features=len(self.labels))

        # проверка влияния инициализации на результат
        torch.nn.init.trunc_normal_(self.linear.weight, std=0.1)

        training_samples, target_labels = data.get_data()

        if self.properties.model__evaluate_every == -1:
            self.properties.model__evaluate_every = max(1, int(len(target_labels) / 50))
        self._add_unknown_words_to_model(training_samples)

        # создаем emb_layer только после добавления всех слов в vocabulary
        self.emb_layer = create_emb_layer(self.embeddings, False)
        training_samples = self._digitize_sents(training_samples)
        target_labels = self._digitize_labels(target_labels)

        self._train_classifier(training_samples, target_labels)

        try:
            self.properties.threshold__confidence_threshold = \
                calculate_confidence_threshold(self.properties, data.get_count_avg_examples(),
                                               data.get_count_classes())
        except NotIntegerException as e:
            logger.error(str(e))
            raise ThresholdCouldNotCountException(str(e))


    def save(self, model_path: Optional[str] = None):
        """
        folder logic, check configs
        """
        logger.info("Saving Pytorch CNN started")

        if model_path is None:
            model_path = os.path.abspath("torch_cnn_model")
        try:
            os.makedirs(model_path)
        except FileExistsError:
            logger.warning("SaveModel directory {} already exists"
                           .format(model_path))

        checkpoint_dir = os.path.join(model_path, "torch_folder")
        ckpt_filepath = os.path.join(checkpoint_dir, "torch_cnn.pt")
        try:
            os.makedirs(checkpoint_dir)
        except FileExistsError:
            logger.warning("Checkpoint directory {} already exists"
                           .format(checkpoint_dir))
        #Первый способ
        # torch.save({
        #     'epoch': self.epoch,
        #     'model_state_dict': self.state_dict(),
        #     'optimizer_state_dict': self.optimizer.state_dict(),
        #     'loss': self.loss,
        # }, ckpt_filepath)

        #Второй способ
        torch.save(self.state_dict(), os.path.join(checkpoint_dir, 'only_state_dict.pt'))

        #TODO проверка не лучше ли этот способ
        #Третий способ
        torch.save(self, os.path.join(checkpoint_dir, 'second_option.pt'))

        logger.info("Saved Pytorch CNN model checkpoint to {}".format(ckpt_filepath))
        with open(
                os.path.join(checkpoint_dir, "cnn_labels.txt"), "w", encoding="utf-8"
        ) as file_out:
            file_out.write("\n".join(self.labels))

        with open(
                os.path.join(checkpoint_dir, "cnn_vocab.txt"), "w", encoding="utf-8"
        ) as file_out:
            logger.info("Sorting {} words...".format(len(self.vocabulary)))
            sorted_vocabulary = [
                kv[0] for kv in sorted(self.vocabulary.items(), key=itemgetter(1))
            ]
            logger.info("Saving sorted vocabulary...")
            file_out.write("\n".join(sorted_vocabulary))

        logger.info("Saving Pytorch CNN ended")



    def _train_classifier(self, training_samples: np.array, target_labels: np.array):
        logger.info("Training PyTorch_CNN......")

        if cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
        device = 'cpu'
        self.to(device)

        batcher_train = DataLoader(My_Dataset(training_samples, target_labels),
                                   batch_size=self.properties.model__batch_size, shuffle=False)

        criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(),
                                     lr=self.properties.model__learning_rate, weight_decay=self.weight_dacay)


        iteration = 0

        for epoch in range(self.properties.model__max_epoch_num):
            self.epoch = epoch
            for i, (samples, labels) in enumerate(batcher_train):
                self.train()
                if labels.shape[0] != self.properties.model__batch_size:
                    break

                samples = samples.to(device)
                labels = labels.to(device)
                self.optimizer.zero_grad()

                outputs = self(samples)

                self.loss = criterion(outputs, torch.argmax(labels, 1, keepdim=False))
                self.loss.backward()
                self.optimizer.step()

                _, predicted = torch.max(outputs.data, 1)

                torch.cuda.empty_cache()
                iteration += 1

            if iteration == self.best_step:
                break
        logger.info("Finished training CNN")


    @profile
    def _train_classifier_with_test(
            self,
            training_samples: np.array,
            test_samples: np.array,
            target_labels: np.array,
            target_test_labels: np.array,
    ) -> Tuple[float, int]:
        logger.info('num_classes {}'.format(len(self.labels)))
        print('num_classes {}'.format(len(self.labels)))
        logger.info("Training PyTorch_CNN...")
        max_acc_dev = 0.0
        best_step_dev = -1

        if torch.cuda.is_available():

            # Tell PyTorch to use the GPU.
            device = torch.device("cuda")

            print('There are %d GPU(s) available.' % torch.cuda.device_count())

            # If not...
        else:
            print('No GPU available, using the CPU instead.')
            device = torch.device("cpu")

        # device = 'cpu'

        if device==torch.device("cuda"):
            print('We will use the GPU:', torch.cuda.get_device_name(0))
        else:
            print('We will use CPU')
        self.to(device)

        # shuffle поставил false для воспроизводимости результатов
        batcher_train = DataLoader(My_Dataset(training_samples, target_labels),
                                   batch_size=self.properties.model__batch_size, shuffle=False)
        batcher_val = DataLoader(My_Dataset(test_samples, target_test_labels),
                                 batch_size=self.properties.model__batch_size, shuffle=False)


        criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(),
                                     lr=self.properties.model__learning_rate,
                                     weight_decay=self.weight_dacay)

        iteration = 0
        # self.properties.model__max_epoch_num = 1
        for epoch in range(self.properties.model__max_epoch_num):

            for i, (samples, labels) in enumerate(batcher_train):
                self.train()
                if labels.shape[0] != self.properties.model__batch_size:
                    break

                samples = samples.to(device)
                labels = labels.to(device)
                self.optimizer.zero_grad()

                outputs = self(samples)

                self.loss = criterion(outputs, torch.argmax(labels, 1, keepdim=False))
                self.loss.backward()
                self.optimizer.step()

                _, predicted = torch.max(outputs.data, 1)

                torch.cuda.empty_cache()
                iteration += 1

                if iteration % self.properties.model__evaluate_every == 0:
                    loss = 0.0
                    accuracy = 0.0
                    self.eval()

                    for j, (samples, labels) in enumerate(batcher_val):
                        if labels.shape[0] != self.properties.model__batch_size:
                            break

                        samples = samples.to(device)
                        labels = labels.to(device)

                        outputs = self(samples)

                        loss += criterion(outputs, torch.argmax(labels, 1, keepdim=False)) * \
                                samples.shape[0] / float(test_samples.shape[0])

                        _, predicted = torch.max(outputs.data, 1)

                        accuracy += accuracy_score(predicted.cpu().numpy(),
                                                   np.argmax(labels.data.cpu().numpy(), axis=1)) * \
                                    samples.shape[0] / float(test_samples.shape[0])

                    if accuracy > max_acc_dev:
                        max_acc_dev = accuracy
                        best_step_dev = iteration

                    print('iter: {}, loss: {}, accuracy: {}'.format(iteration, loss, accuracy))

        logger.info("Finished training PyTorch_CNN, test accuracy={:.2f}"
                    .format(max_acc_dev))
        return max_acc_dev, best_step_dev


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



    def _load_word2vec_vocabulary(self, vocab_file: str) -> dict:
        vocab = {}
        # Added a check to not wrap the whole "with ... as" block with try/except
        if not os.path.exists(vocab_file):
            message = "Vocabulary file doesn't exist"
            logger.error(message)
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


    def _add_unknown_words_to_model(self, train_data_texts: List[List[str]]):
        logger.info("Adding unknown words to model...")
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
        logger.debug("Number of added words: {}".format(number_added_words))
        logger.debug("New vocabulary size: {}".format(len(self.vocabulary)))


    def _digitize_sents(self, texts: List[List[str]]) -> np.array:
        if not isinstance(texts, list):
            message = "texts must be a list"
            logger.warning(message)
            raise TypeError(message)
        digitized_texts = []
        unk_word_number = self.vocabulary['<UNK>']
        pad_token_number = self.vocabulary['<PAD>']
        for sample in texts:
            if not sample:
                logger.debug(
                    "Found an empty text while converting words to integers. Skipping..."
                )
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
            logger.warning(message)
            raise ValueError(message)
        return np.array(digitized_texts, dtype=np.int32)


    def _digitize_labels(self, labels: List[str]) -> np.array:
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
                message = "Unknown label found while one-hot encoding labels: {}"\
                    .format(label)
                logger.warning(message)
                raise KeyError(message)
        return labels_dig




    def _batch_iter(
            self, data: List[Tuple[np.array, np.array]]
    ) -> Iterator[Tuple[np.array, np.array]]:
        data = np.array(data)
        data_size = len(data)
        num_batches_per_epoch = math.ceil(data_size / self.properties.model__batch_size)
        for epoch in range(self.properties.model__max_epoch_num):
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
            for batch_num in range(num_batches_per_epoch):
                start_index = batch_num * self.properties.model__batch_size
                end_index = min((batch_num + 1) * self.properties.model__batch_size, data_size)
                yield shuffled_data[start_index:end_index]
