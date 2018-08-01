import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from q3_run import *
from cs224d.data_utils import *

CONTEXT_SIZE = 5
EMBEDDING_DIM = 10
HIDDEN = 30
BATCH_SIZE = 1
NUM_LAYER =2
SEQ = 5
# file = open('original_rt_snippets.txt')
# total_sentence = file.read().split()

total=[]
dataset = StanfordSentiment()
for sentence in dataset.sentences():
    for w in sentence:
        total.append(w)
print(len(total))
train_sentence = total[:20000]
# print(total)
# we should tokenize the input, but we will ignore that for now
# build a list of tuples.  Each tuple is ([ word_i-2, word_i-1 ], target word)
trigrams = [([train_sentence[i], train_sentence[i + 1], train_sentence[i + 2], train_sentence[i + 3],train_sentence[i + 4],
              train_sentence[i+1], train_sentence[i+2], train_sentence[i+3], train_sentence[i+4], train_sentence[i+ 5],
              train_sentence[i+2], train_sentence[i+3], train_sentence[i+4], train_sentence[i+5],train_sentence[i+6],
              train_sentence[i+3], train_sentence[i+4], train_sentence[i+5], train_sentence[i+6], train_sentence[i+7],
              train_sentence[i+4], train_sentence[i+5], train_sentence[i+6], train_sentence[i+7], train_sentence[i+8]],
             [train_sentence[i + 5], train_sentence[i+6], train_sentence[i+7], train_sentence[i+8], train_sentence[i+9]])
                for i in range(0, len(train_sentence)-5, 5)]
# print the first 3, just so you can see what they look like
print(trigrams[:3])
# print(train_sentence)
vocab = set(train_sentence)
word_to_ix = {word: i for i, word in enumerate(vocab)}
dic_list, tok = create_vector()
print(dic_list)

class LSTMTagger(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, tagset_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers= NUM_LAYER)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (torch.zeros(NUM_LAYER, BATCH_SIZE, self.hidden_dim),
                torch.zeros(NUM_LAYER, BATCH_SIZE, self.hidden_dim))

    def forward(self, sentence):
        lstm_out, self.hidden = self.lstm(
            sentence.view(SEQ, BATCH_SIZE, -1), self.hidden)
        tag_space = self.hidden2tag(lstm_out.view(BATCH_SIZE, -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores

losses = []
loss_function = nn.NLLLoss()
model = LSTMTagger( EMBEDDING_DIM* CONTEXT_SIZE , HIDDEN, len(vocab))

optimizer = optim.SGD(model.parameters(), lr=0.001)

def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr']=lr

curr_lr = 0.001
for epoch in range(10):
    total_loss = 0
    for context, target in trigrams:

        vec=[]
        for i in context:
            vec=np.append(vec, dic_list[tok[i.lower()]])

        model.zero_grad()
        model.hidden = model.init_hidden()

        log_probs = model(torch.tensor(vec, dtype = torch.float))

        loss = sum(loss_function(log_probs[j].view(1,-1), torch.tensor([word_to_ix[target[j]]], dtype=torch.long))
                   for j in range(BATCH_SIZE))

        # Step 5. Do the backward pass and update the gradient
        loss.backward()
        optimizer.step()

        # Get the Python number from a 1-element Tensor by calling tensor.item()
        total_loss += loss.item()
    losses.append(total_loss)
    if (epoch+1)%20 == 0:
        curr_lr/=3
        update_lr(optimizer, curr_lr)
    print(epoch)
    print(total_loss)
print(losses)  # The loss decreased every iteration over the training data!
torch.save(model.state_dict(),'epoch_10_LSTM_50_layer3_seq1_batch5')
