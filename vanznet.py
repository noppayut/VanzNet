import torch
import torch.nn as nn

filepath = 'vanznames.txt'
use_saved_model = True
save_model = True
model_save_path = 'vanznet.pth'
model_load_path = 'vanznet.pth'


def encode_name(name):
    return [all_letters.index(s) for s in name]

# define network
class VanzNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dor=0.15):
        super(VanzNet, self).__init__()
        self.hidden_size = hidden_size
        total_input_size = input_size + hidden_size
        self.i2h = nn.Linear(total_input_size, hidden_size)
        self.i2o = nn.Linear(total_input_size, output_size)
        self.o2o = nn.Linear(hidden_size + output_size, output_size)
        self.dropout = nn.Dropout(dor)
        self.softmax = nn.LogSoftmax(dim=1)
    
    def forward(self, input, hidden):
        input_combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(input_combined)
        output = self.i2o(input_combined)
        output_combined = torch.cat((hidden, output), 1)
        output = self.o2o(output_combined)
        output = self.dropout(output)
        output = self.softmax(output)
        return output, hidden
    
    def initHidden(self):
        return torch.zeros(1, self.hidden_size)
    
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
        hidden = rnn.initHidden()
        out_char = ""
        
        while True:
            output, hidden = rnn(input_tensor[0], hidden)
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

    all_letters = list(set("".join(names_raw)))
    n_letters = len(all_letters) + 1 # including EOS

    names_enc = [encode_name(name) for name in names_raw]

    # setup a network, prepare parameters    
    hidden_size = 128            
    rnn = VanzNet(n_letters, hidden_size, n_letters)

    lr = 0.0005
    epochs = 1000
    print_every = 25
    max_training_size = -1

    if max_training_size > 0:
        names_train = names_enc[:max_training_size]
    else:
        names_train = names_enc

    criterion = nn.NLLLoss()
    optim = torch.optim.Adam(rnn.parameters(), lr=lr)
    losses = []
    
    # training loop
    if use_saved_model:        
        rnn.load_state_dict(torch.load(model_load_path))
    else:        
        print("Start training...")
        print("No. of epochs: %d  ,  Print every: %d eps" % (epochs, print_every))
        print("===================================================")
        for epoch in range(epochs):
            loss_epoch = 0
            for name in names_train:
                input_tensor = name2input(name)
                target_tensor = name2target(name)
                target_tensor.unsqueeze_(-1)
                # print(input_tensor)
                # print(target_tensor)
                hidden = rnn.initHidden()        
                optim.zero_grad()
                loss = 0
                for i in range(input_tensor.size(0)):
                    output, hidden = rnn(input_tensor[i], hidden)
                    loss += criterion(output, target_tensor[i])
                    loss_epoch += loss
                loss /= input_tensor.size(0)
                loss_epoch /= input_tensor.size(0)
                loss.backward()
                optim.step()

            losses.append(loss_epoch / len(names_train))

            if (epoch + 1) % print_every == 0:
                print("%d/%d: Loss %f" % (epoch+1, epochs, loss_epoch))
        
        # save model
        if save_model:            
            torch.save(rnn.state_dict(), model_save_path)
    
    # sampling test
    start_letters = ['ก', 'ค', 'ม', 'จ', 'ว', 'ร', 'บ']

    with torch.no_grad():
        for sp in start_letters:
            print(sample_name(sp))
