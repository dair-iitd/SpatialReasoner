import torch
import torch.nn as nn
from transformers import BertModel

# Best Epoch: 4
class SPNetNoDRL(nn.Module):
    def __init__(self):
        super(SPNetNoDRL, self).__init__()
        self.v = 131
        self.h = 32
        self.l = 3
        self.gru = nn.GRU(self.v, self.h, bidirectional = True, batch_first = True, num_layers = self.l)
        self.fc1 = nn.Linear(2 * self.h, 2 * self.h)
        self.fc2 = nn.Linear(2 * self.h, 2 * self.h)
        self.fc3 = nn.Linear(2 * self.h, 2 * self.h)
        self.fc4 = nn.Linear(2 * self.h  , 1)

    def forward(self, word_embeddings, bid_encoding, **kwargs):
        x = torch.cat([word_embeddings, bid_encoding], dim = 2)
        _, x = self.gru(x)
        x = torch.flatten(x.view(self.l, 2, x.size()[1], self.h)[-1].permute(1, 0, 2), start_dim = 1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

# Best Epoch: 13
class SPNet(nn.Module):
    def __init__(self):
        super(SPNet, self).__init__()
        self.v = 131
        self.h = 32
        self.l = 3
        self.gru = nn.GRU(self.v, self.h, bidirectional = True, batch_first = True, num_layers = self.l)
        self.fc1 = nn.Linear(2 * self.h, 2 * self.h)
        self.fc2 = nn.Linear(2 * self.h, 2 * self.h)
        self.fc3 = nn.Linear(2 * self.h, 2 * self.h)
        self.fc4 = nn.Linear(2 * self.h , 1)

    def forward(self, word_embeddings, bid_encoding, **kwargs):
        d = (bid_encoding[:, :, 0] * bid_encoding[:, :, 2]).unsqueeze(2).permute(0, 2, 1)
        x = torch.cat([word_embeddings, bid_encoding], dim = 2)
        x, _ = self.gru(x)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.tanh(self.fc4(x))
        x = torch.bmm(d, x)
        return x

# Best Epoch: 10
class BertSPNetNoDRL(nn.Module):
    def __init__(self, mode):
        super(BertSPNetNoDRL, self).__init__()
        assert mode in ["train", "test"]
        self.mode = mode
        self.v = 771
        self.h = 32
        self.l = 3
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.gru = nn.GRU(self.v, self.h, bidirectional = True, batch_first = True, num_layers = self.l)
        self.fc1 = nn.Linear(2 * self.h, 2 * self.h)
        self.fc2 = nn.Linear(2 * self.h, 2 * self.h)
        self.fc3 = nn.Linear(2 * self.h, 2 * self.h)
        self.fc4 = nn.Linear(2 * self.h , 1)

    def forward(self, bert_token_ids, bid_encoding, **kwargs):
        with torch.no_grad():
            if(self.mode == "test"):
                x, _ = self.bert(bert_token_ids[:1, :])
                x = x.expand(bert_token_ids.size(0), bert_token_ids.size(1), self.v - 3)
            else:
                x, _ = self.bert(bert_token_ids)
        x = torch.cat([x, bid_encoding], dim = 2)
        _, x = self.gru(x)
        x = torch.flatten(x.view(self.l, 2, x.size()[1], self.h)[-1].permute(1, 0, 2), start_dim = 1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

# Best Epoch: 14
class BertSPNet(nn.Module):
    def __init__(self, mode):
        super(BertSPNet, self).__init__()
        assert mode in ["train", "test"]
        self.mode = mode
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.v = 771
        self.h = 32
        self.l = 3
        self.gru = nn.GRU(self.v, self.h, bidirectional = True, batch_first = True, num_layers = self.l)
        self.fc1 = nn.Linear(2 * self.h, 2 * self.h)
        self.fc2 = nn.Linear(2 * self.h, 2 * self.h)
        self.fc3 = nn.Linear(2 * self.h, 2 * self.h)
        self.fc4 = nn.Linear(2 * self.h , 1)

    def forward(self, bert_token_ids, bid_encoding, **kwargs):
        d = (bid_encoding[:, :, 0] * bid_encoding[:, :, 2]).unsqueeze(2).permute(0, 2, 1)
        with torch.no_grad():
            if(self.mode == "test"):
                x, _ = self.bert(bert_token_ids[:1, :])
                x = x.expand(bert_token_ids.size(0), bert_token_ids.size(1), self.v - 3)
            else:
                x, _ = self.bert(bert_token_ids)
        x = torch.cat([x, bid_encoding], dim = 2)
        x, _ = self.gru(x)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.tanh(self.fc4(x))
        x = torch.bmm(d, x)
        return x
