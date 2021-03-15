from torchtext import data,legacy
# from torchtext.vocab import Vectors
import spacy
import pandas as pd
import pickle
from torch.utils.data import Dataset
import torch
from transformers import BertTokenizer


class WordDataset(object):
    def __init__(self, sequence_len, batch_size):
        self.sequence_len = sequence_len
        self.batch_size = batch_size
        self.train_iterator = None
        self.test_iterator = None
        self.val_iterator = None
        self.vocab = []
        self.word_embeddings = {}

    @staticmethod
    def get_pandas_df(filename, text_col, label_col):
        """
        Load the data into Pandas.DataFrame object
        This will be used to convert data to torchtext object
        """
        df = pd.read_csv(filename, sep='\t')
        texts = []
        labels = []
        for index, row in df.iterrows():
            texts.append(row[text_col])
            labels.append(int(row[label_col]))
        df = pd.DataFrame({"text": texts, "label": labels})
        return df

    def load_data(self, train_file, test_file, val_file=None, w2v_file=None, text_col='text', label_col='label'):
        """
        Loads the data from files
        Sets up iterators for training, validation and test data
        Also create vocabulary and word embeddings based on the data

        Inputs:
            w2v_file (String): absolute path to file containing word embeddings (GloVe/Word2Vec)
            train_file (String): absolute path to training file
            test_file (String): absolute path to test file
            val_file (String): absolute path to validation file
        """

        nlp = spacy.load("en_core_web_sm")
        tokenizer = lambda sent: [x.text for x in nlp.tokenizer(sent) if x.text != " "]

        # Creating Field for data
        text = legacy.data.Field(sequential=True, tokenize=tokenizer, lower=True, fix_length=self.sequence_len)
        label = legacy.data.Field(sequential=False, use_vocab=False)
        datafields = [("text", text), ("label", label)]

        # Load data from pd.DataFrame into torchtext.data.Dataset
        train_df = self.get_pandas_df(train_file, text_col, label_col)
        train_examples = [legacy.data.Example.fromlist(i, datafields) for i in train_df.values.tolist()]
        train_data = legacy.data.Dataset(train_examples, datafields)

        test_df = self.get_pandas_df(test_file, text_col, label_col)
        test_examples = [legacy.data.Example.fromlist(i, datafields) for i in test_df.values.tolist()]
        test_data = legacy.data.Dataset(test_examples, datafields)

        # If validation file exists, load it. Otherwise get validation data from training data
        if val_file:
            val_df = self.get_pandas_df(val_file, text_col, label_col)
            val_examples = [legacy.data.Example.fromlist(i, datafields) for i in val_df.values.tolist()]
            val_data = legacy.data.Dataset(val_examples, datafields)
        else:
            train_data, val_data = train_data.split(split_ratio=0.9)

        vectors = None
        if w2v_file is not None:
            with open(w2v_file, 'rb') as handle:
                vectors = pickle.load(handle)
        # vectors = Vectors(w2v_file)
        # with open('glove.pickle', 'wb') as handle:
        #     pickle.dump(vectors, handle)
        text.build_vocab(train_data, vectors=vectors)
        self.word_embeddings = text.vocab.vectors
        self.vocab = text.vocab

        self.train_iterator = legacy.data.BucketIterator(
            train_data,
            batch_size=self.batch_size,
            sort_key=lambda x: len(x.text),
            repeat=False,
            shuffle=True)

        self.val_iterator, self.test_iterator = legacy.data.BucketIterator.splits(
            (val_data, test_data),
            batch_size=self.batch_size,
            sort_key=lambda x: len(x.text),
            repeat=False,
            shuffle=False)

        print("Loaded {} training examples".format(len(train_data)))
        print("Loaded {} test examples".format(len(test_data)))
        print("Loaded {} validation examples".format(len(val_data)))


class BertDataset(Dataset):
    def __init__(self, filename, sequence_len):
        self.df = pd.read_csv(filename, delimiter='\t')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.sequence_len = sequence_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        sentence = self.df.loc[index, 'text']
        label = self.df.loc[index, 'label']
        tokens = self.tokenizer.tokenize(sentence)
        tokens = ['[CLS]'] + tokens + ['[SEP]']
        if len(tokens) < self.sequence_len:
            tokens = tokens + ['[PAD]' for _ in range(self.sequence_len - len(tokens))]
        else:
            tokens = tokens[:self.sequence_len - 1] + ['[SEP]']

        tokens_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        tokens_ids_tensor = torch.tensor(tokens_ids)
        attn_mask = (tokens_ids_tensor != 0).long()
        return tokens_ids_tensor, attn_mask, label
