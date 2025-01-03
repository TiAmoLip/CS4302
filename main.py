
import torch.backends
import torch.backends.cudnn
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import defaultdict
import argparse
import spacy
import tqdm
import torch
from torch.profiler import profile, record_function, ProfilerActivity
import pickle
parser = argparse.ArgumentParser()
parser.add_argument("--new_profiler_log", type=str, default="False")
args = parser.parse_args()
# print(torch.backends.cudnn.enabled)
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
if args.new_profiler_log == "False" or args.new_profiler_log == "false" or args.new_profiler_log == "None":
    new_profile_log = "original_kernel_profile"
else:
    new_profile_log = args.new_profiler_log

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
class Encoder(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, dropout):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.rnn = nn.GRU(embedding_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # src = [src length, batch size]
        embedded = self.dropout(self.embedding(src))
        # embedded = [src length, batch size, embedding dim]
        outputs, hidden = self.rnn(embedded)  # no cell state in GRU!
        # outputs = [src length, batch size, hidden dim * n directions]
        # hidden = [n layers * n directions, batch size, hidden dim]
        # outputs are always from the top hidden layer
        return hidden
    
    
class Decoder(nn.Module):
    def __init__(self, output_dim, embedding_dim, hidden_dim, dropout):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.embedding = nn.Embedding(output_dim, embedding_dim)
        self.rnn = nn.GRU(embedding_dim + hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(embedding_dim + hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, context):
        # input = [batch size]
        # hidden = [n layers * n directions, batch size, hidden dim]
        # context = [n layers * n directions, batch size, hidden dim]
        # n layers and n directions in the decoder will both always be 1, therefore:
        # hidden = [1, batch size, hidden dim]
        # context = [1, batch size, hidden dim]
        input = input.unsqueeze(0)
        # input = [1, batch size]
        embedded = self.dropout(self.embedding(input))
        # embedded = [1, batch size, embedding dim]
        emb_con = torch.cat((embedded, context), dim=2)
        # emb_con = [1, batch size, embedding dim + hidden dim]
        output, hidden = self.rnn(emb_con, hidden)
        # output = [seq len, batch size, hidden dim * n directions]
        # hidden = [n layers * n directions, batch size, hidden dim]
        # seq len, n layers and n directions will always be 1 in this decoder, therefore:
        # output = [1, batch size, hidden dim]
        # hidden = [1, batch size, hidden dim]
        output = torch.cat(
            (embedded.squeeze(0), hidden.squeeze(0), context.squeeze(0)), dim=1
        )
        # output = [batch size, embedding dim + hidden dim * 2]
        
        prediction = torch.matmul(output, self.fc_out.weight.T) + self.fc_out.bias
        
        # print("custom kernel output: ",prediction)
        # prediction = self.fc_out(output)
        # print("true kernel output",prediction)

        # print(output.shape, self.fc_out.weight.T.shape) # torch.Size([1, 1280]) torch.Size([1280, 5893])
        # 因为这里只有这一个地方有matmul，所以我打算直接写一个特殊算子，他接受的第一行正好是0, 而对于这个sgemm，他接受的第一个矩阵的行数为1.
        # prediction = [batch size, output dim]
        return prediction, hidden
    
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        assert (
            encoder.hidden_dim == decoder.hidden_dim
        ), "Hidden dimensions of encoder and decoder must be equal!"

    def forward(self, src, trg, teacher_forcing_ratio):
        # src = [src length, batch size]
        # trg = [trg length, batch size]
        # teacher_forcing_ratio is probability to use teacher forcing
        # e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time
        batch_size = trg.shape[1]
        trg_length = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        # tensor to store decoder outputs
        outputs = torch.zeros(trg_length, batch_size, trg_vocab_size).to(self.device)
        # last hidden state of the encoder is the context
        context = self.encoder(src)
        # context = [n layers * n directions, batch size, hidden dim]
        # context also used as the initial hidden state of the decoder
        hidden = context
        # hidden = [n layers * n directions, batch size, hidden dim]
        # first input to the decoder is the <sos> tokens
        input = trg[0, :]
        for t in range(1, trg_length):
            # insert input token embedding, previous hidden state and the context state
            # receive output tensor (predictions) and new hidden state
            output, hidden = self.decoder(input, hidden, context)
            # output = [batch size, output dim]
            # hidden = [1, batch size, hidden dim]
            # place predictions in a tensor holding predictions for each token
            outputs[t] = output
            # decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio
            # get the highest predicted token from our predictions
            top1 = output.argmax(1)
            # if teacher forcing, use actual next token as next input
            # if not, use predicted token
            input = trg[t] if teacher_force else top1
            # input = [batch size]
        return outputs
    

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
        
        context = model.encoder(tensor)
        hidden = context
        inputs = [en_vocab[sos_token]]
        decode_total = 0
        for _ in range(max_output_length):
            inputs_tensor = torch.LongTensor([inputs[-1]]).to(device)
            output, hidden = model.decoder(inputs_tensor, hidden, context)
            predicted_token = output.argmax(-1).item()
            inputs.append(predicted_token)
            if predicted_token == en_vocab[eos_token]:
                break
        tokens = lookup_tokens(en_index2word,inputs)
    return (tokens,)

unk = "<unk>"
en_token2index = pickle.load(open("vocabs/en_vocab_dict.pkl", "rb")) # token to index
en_index2token = {v: k for k, v in en_token2index.items()} # index to token
de_token2index = pickle.load(open("vocabs/de_vocab_dict.pkl", "rb"))
de_token2index = defaultdict(lambda: de_token2index[unk], de_token2index) # token to index
sos_token = '<sos>'
eos_token = "<eos>"


with open("vocabs/test_de.txt",'r') as f:
    sentences = f.read().split('\n')

# translate_sentence(
#             sentences[0],
#             model,
#             en_token2index,
#             de_nlp,
#             en_index2token,
#             de_token2index,
#             True,
#             sos_token,
#             eos_token,
#             device,)
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True,with_stack=True) as prof:
    translations = [translate_sentence(
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
        ) for sentence in tqdm.tqdm(sentences)]
with open(f"output/{new_profile_log}.txt", "w") as f:
    f.write(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20, max_src_column_width=100))

    
# print(translations)

def check_correctness(translations, correct_translations):
    """
    Check whether using custom kernels destory the correctness of the model.
    Args:
        translations: List[str],
        correct_translations: List[long_str]
    """
    correct_translations = [t.split(" ") for t in correct_translations]
    translations = [t[0][1:-1] for t in translations]
    char_level_correctness = 0
    assert len(translations) <= len(correct_translations)
    
    for i in range(len(translations)):
        if translations[i] == correct_translations[i]:
            char_level_correctness += 1
    print(f"Character level correctness: {char_level_correctness/len(translations)}")
with open("vocabs/original_predictions.txt",'r') as f:
    correct_translations = f.read().split('\n')

check_correctness(translations, correct_translations)
