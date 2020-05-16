import torch
import torch.optim as optim

from torchtext.datasets import TranslationDataset, Multi30k
from torchtext.data import Field, BucketIterator

import spacy
import numpy as np

import random
#import matplotlib.pyplot as plt
#import matplotlib.ticker as ticker
import Conv2Seq
from utils import Utils

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Load Languages
class Trainer:
    def __init__(self, num_epochs= 10, batch_size = 16, lr = 0.1, num_layers = 2):
        self.number_of_epochs = num_epochs
        self.clip = 1
        self.german = spacy.load("de")
        self.english = spacy.load('en')

        self.util = Utils(self.german, self.english)

        self.src = Field(tokenize = self.util.tokenize_german,
                    init_token = '<sos>',
                    eos_token = '<eos>',
                    lower = True, batch_first=True)

        self.trg = Field(tokenize = self.util.tokenize_english,
                    init_token = '<sos>',
                    eos_token = '<eos>',
                    lower = True, batch_first=True)

        self.train_data, self.valid_data, self.test_data = Multi30k.splits(exts = ('.de', '.en'),
                                                            fields = (self.src, self.trg))


        self.src.build_vocab(self.train_data,min_freq=2)

        self.trg.build_vocab(self.train_data,min_freq=2)

        self.batch_size = batch_size
        self.lr = lr
        self.train_iterator, self.valid_iterator, self.test_iterator = BucketIterator.splits(
            (self.train_data, self.valid_data, self.test_data),
            batch_size = self.batch_size,
            device = device)

        self.input_dimension = len(self.src.vocab)
        self.output_dimension = len(self.trg.vocab)

        self.encoder_embedding_dimension = 256
        self.decoder_embedding_dimension = 256
        self.hidden_dimension = 512
        self.number_of_layers = num_layers
        self.encoder_dropout = 0.5
        self.decoder_dropout = 0.5



        self.model = Conv2Seq.Conv2Seq(device,self.trg.vocab.stoi[self.trg.pad_token], input_dimension= len(self.src.vocab), out_dim = len(self.trg.vocab)).to(device)

        self.optimizer = optim.SGD(self.model.parameters(), lr = self.lr, momentum= 0.99)


    def train(self, criterion):
        self.model.train()

        epoch_loss = 0

        for i, batch in enumerate(self.train_iterator):
            src = batch.src
            trg = batch.trg
            self.optimizer.zero_grad()

            output, _ = self.model(src, trg[:, :-1])

            # output = [batch size, trg len - 1, output dim]
            # trg = [batch size, trg len]

            output_dim = output.shape[-1]

            output = output.contiguous().view(-1, output_dim)
            trg = trg[:, 1:].contiguous().view(-1)

            # output = [batch size * trg len - 1, output dim]
            # trg = [batch size * trg len - 1]

            loss = criterion(output, trg)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)

            self.optimizer.step()

            epoch_loss += loss.item()

        return epoch_loss / len(self.train_iterator)


    def evaluate(self, iterator, criterion):
        self.model.eval()

        epoch_loss = 0

        with torch.no_grad():
            for i, batch in enumerate(iterator):
                src = batch.src
                trg = batch.trg
                output, _ = self.model(src, trg[:, :-1])

                # output = [batch size, trg len - 1, output dim]
                # trg = [batch size, trg len]

                output_dim = output.shape[-1]

                output = output.contiguous().view(-1, output_dim)
                trg = trg[:,1:].contiguous().view(-1)

                # output = [batch size * trg len - 1, output dim]
                # trg = [batch size * trg len - 1]

                loss = criterion(output, trg)

                epoch_loss += loss.item()

        return epoch_loss / len(iterator)

    def epoch_time(self,start_time, end_time):
        elapsed_time = end_time - start_time
        elapsed_mins = int(elapsed_time / 60)
        elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
        return elapsed_mins, elapsed_secs
