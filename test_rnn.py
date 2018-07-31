import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from q3_run import *
from cs224d.data_utils import *



CONTEXT_SIZE = 5
EMBEDDING_DIM = 10

class NGramLanguageModeler(nn.Module):

    def __init__(self, vocab_size, embedding_dim, context_size):
        super(NGramLanguageModeler, self).__init__()
        self.linear1 = nn.Linear(context_size * embedding_dim, 128)
        self.linear2 = nn.Linear(128, vocab_size)

    def forward(self, inputs):
        out = F.relu(self.linear1(inputs.view((1, -1))))
        out = self.linear2(out)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs

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

model = NGramLanguageModeler(len(vocab), EMBEDDING_DIM, CONTEXT_SIZE)
model.load_state_dict(torch.load('epoch_80'))

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