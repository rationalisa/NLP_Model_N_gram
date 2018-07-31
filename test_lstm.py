import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from q3_run import *
from cs224d.data_utils import *



CONTEXT_SIZE = 5
EMBEDDING_DIM = 10
HIDDEN = 10

class LSTMTagger(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, tagset_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (torch.zeros(1, 1, self.hidden_dim),
                torch.zeros(1, 1, self.hidden_dim))

    def forward(self, sentence):
        lstm_out, self.hidden = self.lstm(
            sentence.view(1, 1, -1), self.hidden)
        tag_space = self.hidden2tag(lstm_out.view(1, -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores



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
trigrams = [([train_sentence[i], train_sentence[i + 1], train_sentence[i + 2], train_sentence[i + 3],
              train_sentence[i + 4]], train_sentence[i + 5])
            for i in range(len(train_sentence) - 5)]
# print the first 3, just so you can see what they look like
print(trigrams[:3])
# print(train_sentence)
vocab = set(train_sentence)
word_to_ix = {word: i for i, word in enumerate(vocab)}
dic_list, tok = create_vector()
print(dic_list)



model = LSTMTagger( EMBEDDING_DIM * CONTEXT_SIZE, HIDDEN, len(vocab))
model.load_state_dict(torch.load('epoch_80_LSTM'))

product = 1
count = 0
for context, target in trigrams:
    vec = []
    for i in context:
        vec = np.append(vec, dic_list[tok[i.lower()]])

    log_probs = model(torch.tensor(vec, dtype=torch.float))
    index = word_to_ix[target]
    prob = np.exp(log_probs[:,index].detach().numpy()[0])
    product *= prob
    count += 1
    if count == 50:
        print(count)
        PP=product **(-1/count)
        print(PP)
        break