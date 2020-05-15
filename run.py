import argparse

from Trainer import Trainer
import math
import time
import torch.nn as nn
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=3, help= 'Batch size. In case of rnn this would be automatically be set to 1')
parser.add_argument('--mode', type=str, default='train', help= 'Options : train: Start training, test: Test trained model, translate : Give a sentence to be translated by a trained model. ')
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--continue', type=str, default= None)
parser.add_argument('--num_epochs', type=int, default=30)
parser.add_argument('--num_layers', type=int, default=2)

config = parser.parse_args()

trainer = Trainer(num_epochs= config.num_epochs, batch_size= config.batch_size, lr = config.lr, num_layers = config.num_layers)


best_valid_loss = float('inf')

target_pad_idx = trainer.trg.vocab.stoi[trainer.trg.pad_token]

criterion = nn.CrossEntropyLoss(ignore_index = target_pad_idx)
if config.mode == 'train':
    for epoch in range(trainer.number_of_epochs):

        start_time = time.time()

        train_loss = trainer.train(criterion)
        valid_loss = trainer.evaluate(trainer.valid_iterator, criterion)

        end_time = time.time()

        epoch_mins, epoch_secs = trainer.epoch_time(start_time, end_time)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(trainer.model.state_dict(), 'convseq.pt')

        print('Epoch: {} | Time: {}m {}s'.format(epoch, epoch_mins, epoch_secs))
        print('Train Loss: {:3f} | Train PPL: {:7.3f}'.format(train_loss, math.exp(train_loss)))
        print('Val. Loss: {:3f} |  Val. PPL: {:7.3f}'.format(valid_loss, math.exp(valid_loss)))

elif config.mode == 'test':

    trainer.model.load_state_dict(torch.load('convseq.pt'))

    test_loss = trainer.evaluate(trainer.test_iterator, criterion)

    print('| Test Loss: {:3f} | Test PPL: {:7.3f} |'.format(test_loss, math.exp(test_loss)))

else:
    pass