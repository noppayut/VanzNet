import pickle as pk

import torch
import torch.nn as nn

filepath = 'vanznames.txt'
test_mode = False
use_saved_model = False
save_model = True
letter_path = 'letters.pkl'
model_save_path = 'vanznet_lstm.pth'
model_load_path = 'vanznet_lstm.pth'
test_start_letters = ['ก', 'ค', 'ม', 'จ', 'ว', 'ร', 'บ']

def encode_name(name):
    return [all_letters.index(s) for s in name]

# define network
class VanzNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dor=0.15, bid=True):
        super(VanzNet, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=3
                            , dropout=dor, bidirectional=bid)
        if bid:
            input_mult = 2
        else:
            input_mult = 1
        self.lstm2o = nn.Linear(hidden_size*input_mult, output_size)
        self.softmax = nn.LogSoftmax(dim=2)
    
    def forward(self, input, h, c):
        if h is None:            
            output_lstm, (h_out, c_out) = self.lstm(input)
        else:
            output_lstm, (h_out, c_out)  = self.lstm(input, (h, c))
        output_fc = self.lstm2o(output_lstm)
        output = self.softmax(output_fc)
        return output, h_out, c_out
    
# one-hot matrix for characters
def name2input(name_enc):
    """    
    dim: [len_name * 1 * n_letters]
    e.g. KASPAROV -> [OneHot(K), OneHot(A), ..., OneHot(V)]
    """
    tensor = torch.zeros(len(name_enc), 1, n_letters)
    for i, n in enumerate(name_enc):
        tensor[i][0][n] = 1
    return tensor

def name2target(name_enc):
    """
    dim: [len_name]
    e.g. target(KASPAROV) -> ASPAROV<EOS> -> [Idx(A), Idx(S), ..., Idx(EOS)]
    """
    return torch.LongTensor(name_enc[1:] + [n_letters - 1])
       
def reconstruct_char(output):
    topv, topi = output.topk(1)
    idx = topi[0][0]
    if idx == n_letters - 1:
        return 'EOS'
    else:
        return all_letters[idx]

def sample_name(start_char):
    name_sample = start_char
    if not (start_char in all_letters):
        return "Invalid start character!"
    else:
        start_char_enc = [all_letters.index(start_char)]
        input_tensor = name2input(start_char_enc)
        out_char = ""
        h, c = None, None
        
        while True:
            output, h, c = rnn(input_tensor[0].view(1,1,-1), h, c)
            out_char = reconstruct_char(output)
            if out_char == 'EOS':
                break
            name_sample += out_char
            input_tensor = name2input([all_letters.index(out_char)])
    
    return name_sample

if __name__ == "__main__":
    # prepare data
    with open(filepath, 'r') as f:
        names_raw = list(map(lambda x: x.replace('\n', ''), f.readlines()))
    
    if use_saved_model:
        with open(letter_path, 'rb') as f:
            all_letters = pk.load(f)
    else:
        all_letters = list(set("".join(names_raw)))
        if save_model:
            with open(letter_path, 'wb') as f:
                pk.dump(all_letters, f)
    
    n_letters = len(all_letters) + 1 # including EOS

    names_enc = [encode_name(name) for name in names_raw]

    # setup a network, prepare parameters    
    hidden_size = 128
    rnn = VanzNet(n_letters, hidden_size, n_letters)
    
    if use_saved_model:
        rnn.load_state_dict(torch.load(model_load_path))
        print("*** Loaded saved model ***")

    lr = 0.005
    epochs = 50
    print_every = 5
    max_training_size = -1
    lr_decay_step = 5
    lr_decay_rate = 0.95

    if max_training_size > 0:
        names_train = names_enc[:max_training_size]
    else:
        names_train = names_enc

    criterion = nn.NLLLoss()
    optim = torch.optim.Adam(rnn.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optim
                                                , lr_decay_step
                                                , lr_decay_rate)
    losses = []
    print("Start training")
    for epoch in range(epochs):
        if test_mode:
            break
        loss_epoch = 0
        for name in names_train:
            input_tensor = name2input(name)        
            target_tensor = name2target(name)
            target_tensor.unsqueeze_(-1)
            optim.zero_grad()
            loss = 0
            h, c = None, None
            for i in range(input_tensor.size(0)):
                output, h, c = rnn(input_tensor[i].view(1,1, -1), h, c)
                # print(output.size(), target_tensor[i].size())
                loss += criterion(output.view(1, -1), target_tensor[i])
                loss_epoch += loss
            loss_epoch /= input_tensor.size(0)
            loss.backward()
            optim.step()

        scheduler.step()
        losses.append(loss_epoch / len(names_train))

        if (epoch + 1) % print_every == 0:
            print("%d/%d: Loss %f" % (epoch+1, epochs, loss_epoch))
            if save_model:
                torch.save(rnn.state_dict(), model_save_path)
                print("***** Model saved *****")
        
        # save model
        if save_model:            
            torch.save(rnn.state_dict(), model_save_path)
    
    with torch.no_grad():
        for sp in test_start_letters:
            print(sample_name(sp))
