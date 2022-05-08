import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super().__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first = True)
        #self.dropout = nn.Dropout(0.5)
        
        self.fc = nn.Linear(hidden_size, vocab_size)
        
        
    def init_hidden(self):
        return torch.zeros(1,1,self.hidden_size),torch.zeros(1,1,self.hidden_size)     
    
    def forward(self, features, captions):
        # embedding captions
        captions = self.embed(captions[:, :-1])
        lstm_input = torch.cat((features.unsqueeze(1), captions), dim=1)
        
        # LSTM
        x, self.hidden = self.lstm(lstm_input)

               
        # Linear
        x = self.fc(x)
        
        # return x 
        return x

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        words = []
                
        # initialize the hidden states as inputs
        hidden_state = (torch.randn(self.num_layers, 1, self.hidden_size).to(inputs.device),
                 torch.randn(self.num_layers, 1, self.hidden_size).to(inputs.device))
        
        # now to get the caption feed the lstm output and hidden states back to itself
        for i in range(max_len):
            out, hidden_state = self.lstm(inputs, hidden_state)
            output  = self.fc(out) 
            output  = output.squeeze(1)
            word_id = output.argmax(dim = 1)
            words.append(word_id.item())
            
            # input for next iteratons
            inputs = self.embed(word_id.unsqueeze(0))
            
        return words
            
