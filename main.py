
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import defaultdict
import spacy
import tqdm
import torch
import datasets
import pickle
from myModel import Encoder, Decoder, Seq2Seq
import time

seed = 1234

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True

en_nlp = spacy.load("en_core_web_sm")
de_nlp = spacy.load("de_core_news_sm")

    
input_dim = 7853
output_dim = 5893
encoder_embedding_dim = 256
decoder_embedding_dim = 256
hidden_dim = 512
encoder_dropout = 0.5
decoder_dropout = 0.5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

encoder = Encoder(
    input_dim,
    encoder_embedding_dim,
    hidden_dim,
    encoder_dropout,
)

decoder = Decoder(
    output_dim,
    decoder_embedding_dim,
    hidden_dim,
    decoder_dropout,
)

model = Seq2Seq(encoder, decoder, device).to(device)

model.load_state_dict(torch.load("tut2-model.pt"))

def lookup_indices(vocab:dict, tokens):
    return [vocab[token] for token in tokens]
def lookup_tokens(vocab_index2word:dict, indices):
    return [vocab_index2word[index] for index in indices]

def translate_sentence(
    sentence,
    model,
    en_vocab,
    de_nlp,
    en_index2word,
    de_vocab,
    lower,
    sos_token,
    eos_token,
    device,
    max_output_length=25,
):
    model.eval()
    with torch.no_grad():
        if isinstance(sentence, str):
            tokens = [token.text for token in de_nlp.tokenizer(sentence)]
        else:
            tokens = [token for token in sentence]
        if lower:
            tokens = [token.lower() for token in tokens]
        tokens = [sos_token] + tokens + [eos_token]
        ids = lookup_indices(de_vocab,tokens)
        tensor = torch.LongTensor(ids).unsqueeze(-1).to(device)
        
        encode_start = time.time()
        context = model.encoder(tensor)
        encode_end = time.time()
        
        
        hidden = context
        inputs = [en_vocab[sos_token]]
        decode_total = 0
        for _ in range(max_output_length):
            inputs_tensor = torch.LongTensor([inputs[-1]]).to(device)
            decode_start = time.time()
            output, hidden = model.decoder(inputs_tensor, hidden, context)
            decode_end = time.time()
            predicted_token = output.argmax(-1).item()
            inputs.append(predicted_token)
            if predicted_token == en_vocab[eos_token]:
                break
            decode_total += decode_end - decode_start
        tokens = lookup_tokens(en_index2word,inputs)
    return tokens, encode_end - encode_start, decode_total

unk = "<unk>"
en_token2index = pickle.load(open("en_vocab_dict.pkl", "rb")) # token to index
en_index2token = {v: k for k, v in en_token2index.items()} # index to token
de_token2index = pickle.load(open("de_vocab_dict.pkl", "rb"))
de_token2index = defaultdict(lambda: de_token2index[unk], de_token2index) # token to index
sos_token = '<sos>'
eos_token = "<eos>"


with open("test_de.txt",'r') as f:
    sentences = f.read().split('\n')


translations = [
    translate_sentence(
        sentence,
        model,
        en_token2index,
        de_nlp,
        en_index2token,
        de_token2index,
        True,
        sos_token,
        eos_token,
        device,
    ) for sentence in tqdm.tqdm(sentences)
]
predictions = [t[0] for t in translations]
with open("new_kernel_predictions.txt", "w") as f:
    for prediction in predictions:
        f.write(" ".join(prediction[1:-1]) + "\n")
print("Total Encode Time: s", sum([t[1] for t in translations]))
print("Total Decode Time: s", sum([t[2] for t in translations]))