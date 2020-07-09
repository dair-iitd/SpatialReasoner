import os
import sys
import json
import time
import torch
import random
import argparse
import itertools
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataset import Dataset
from . import net
from utils import common

# No Bert
# random.seed(2208)
# torch.manual_seed(2208)

# Bert
# random.seed(2207)
# torch.manual_seed(2207)

class TrainDataset(Dataset):
    def __init__(self, dataset, num_negative_samples):
        inputs = []
        for item in dataset:
            num_tokens = len(item["tokens"])
            gold_distance_encoding = common.getDistanceEncodings(num_tokens, item["chosen_positions"], [item["gold_distances"]])[0]

            chosen_negative_sample_indices = random.sample(list(range(len(item["negative_samples_distances"]))), num_negative_samples)
            negative_samples_distances = torch.tensor(item["negative_samples_distances"])[chosen_negative_sample_indices]
            negative_samples_distance_encodings = common.getDistanceEncodings(num_tokens = num_tokens, chosen_positions = item["chosen_positions"], candidate_distances = negative_samples_distances.tolist())

            inputs.append((item["word_embeddings"], item["bert_token_ids"], item["bi_encoding"], gold_distance_encoding, negative_samples_distance_encodings))

        self.length = len(inputs) * num_negative_samples
        self.num_negative_samples = num_negative_samples

        self.word_embeddings_list = pad_sequence([torch.tensor(x) for x, _, _, _, _ in inputs], batch_first = True, padding_value = 0)
        self.bert_token_ids_list = pad_sequence([torch.tensor(x) for _, x, _, _, _ in inputs], batch_first = True, padding_value = 0)
        self.bi_encoding_list = pad_sequence([torch.tensor(x) for _, _, x, _, _ in inputs], batch_first = True, padding_value = 0)
        self.gd_encoding_list = pad_sequence([torch.tensor(x) for _, _, _, x, _ in inputs], batch_first = True, padding_value = 0)
        self.nd_encodings_list = pad_sequence(list(itertools.chain.from_iterable([torch.tensor(x) for _, _, _, _, x in inputs])), batch_first = True, padding_value = 0).unsqueeze(1).view(len(inputs), num_negative_samples, -1)

    def __getitem__(self, index):
        indexi = index // self.num_negative_samples
        indexj = index % self.num_negative_samples

        word_embeddings = self.word_embeddings_list[indexi]
        bert_token_ids = self.bert_token_ids_list[indexi]
        bi_encoding = self.bi_encoding_list[indexi]
        gd_encoding = self.gd_encoding_list[indexi]
        nd_encoding = self.nd_encodings_list[indexi][indexj]

        gbid_encoding = torch.cat([bi_encoding, gd_encoding.unsqueeze(1)], dim = 1)
        nbid_encoding = torch.cat([bi_encoding, nd_encoding.unsqueeze(1)], dim = 1)

        return (word_embeddings, bert_token_ids, gbid_encoding, nbid_encoding)

    def __len__(self):
        return self.length

def train(options):
    dataset = json.load(open(options.dataset_file_path, "r"))[:options.num_examples]
    print("Training on %d examples" % (len(dataset)))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Running on %s device" % device)

    model = None
    if(options.model == "SPNetNoDRL"):
        model = net.SPNetNoDRL()
    if(options.model == "SPNet"):
        model = net.SPNet()
    if(options.model == "BertSPNetNoDRL"):
        model = net.BertSPNetNoDRL(mode = "train")
    if(options.model == "BertSPNet"):
        model = net.BertSPNet(mode = "train")
    if(model is None):
        raise Exception("Model should be one of SPNetNoDRL, SPNet, BertSPNetNoDRL, BertSPNet")

    print("Using %s model" % (options.model))
    model.to(device)

    if(options.pretrained_model_path is not None):
        print("Loading saved model from %s" % (options.pretrained_model_path))
        model.load_state_dict(torch.load(options.pretrained_model_path, map_location = torch.device(device)))

    criterion = nn.MarginRankingLoss(margin = options.margin)
    optimizer = optim.Adam(model.parameters(), lr = options.learning_rate)

    print()
    print("Batch Size     : %d" % (options.batch_size))
    print("Num Epochs     : %d" % (options.num_epochs))
    print("Margin         : %f" % (options.margin))
    print("Learning Rate  : %f" % (options.learning_rate))
    print("Num Neg Samples: %d" % (options.num_negative_samples))
    print()

    print("Training . . .")
    y = -torch.ones(options.batch_size).cuda()
    for epoch in range(options.num_epochs):
        print("Epoch %d" % (epoch + 1))
        running_loss = 0.0
        trainset = TrainDataset(dataset, options.num_negative_samples)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size = options.batch_size, shuffle = False, num_workers = 0)
        start = time.time()
        for index, data in enumerate(trainloader):
            word_embeddings, bert_token_ids, gbid_encoding, nbid_encoding = data[0].to(device), data[1].to(device), data[2].to(device), data[3].to(device)
            optimizer.zero_grad()
            goutputs, noutputs = model(word_embeddings = word_embeddings, bert_token_ids = bert_token_ids, bid_encoding = gbid_encoding), model(word_embeddings = word_embeddings, bert_token_ids = bert_token_ids, bid_encoding = nbid_encoding)
            loss = criterion(goutputs, noutputs, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if(index % 1000 == 999):
                end = time.time()
                print('[%d, %5d] loss: %.3f, time: %0.2f' % (epoch + 1, index + 1, running_loss / 1000, end - start))
                running_loss = 0.0
                start = time.time()
        print()

        if(options.save_model_file_path is not None):
            print("Saving trained model to %s" % (options.save_model_file_path + ".epoch%d" % (epoch)))
            torch.save(model.state_dict(), (options.save_model_file_path + ".epoch%d" % (epoch)))

if(__name__ == "__main__"):
    project_root_path = common.getProjectRootPath()

    defaults = {}

    defaults["dataset_file_path"] = project_root_path / "dataset/train.json"
    defaults["model"] = "SPNet"
    defaults["num_examples"] = 6000
    defaults["batch_size"] = 20
    defaults["num_epochs"] = 15
    defaults["margin"] = 0.5
    defaults["learning_rate"] = 0.002
    defaults["num_negative_samples"] = 40
    defaults["pretrained_model_path"] = None
    defaults["save_model_file_path"] = project_root_path / ("models/%s.th" % defaults["model"])

    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_file_path", type = str, default = defaults["dataset_file_path"])
    parser.add_argument("--model", type = str, default = defaults["model"])
    parser.add_argument("--num_examples", type = int, default = defaults["num_examples"])
    parser.add_argument("--batch_size", type = int, default = defaults["batch_size"])
    parser.add_argument("--num_epochs", type = int, default = defaults["num_epochs"])
    parser.add_argument("--margin", type = float, default = defaults["margin"])
    parser.add_argument("--learning_rate", type = float, default = defaults["learning_rate"])
    parser.add_argument("--num_negative_samples", type = int, default = defaults["num_negative_samples"])
    parser.add_argument("--pretrained_model_path", type = str, default = defaults["pretrained_model_path"])
    parser.add_argument("--save_model_file_path", type = str, default = defaults["save_model_file_path"])

    options = parser.parse_args(sys.argv[1:])

    train(options)
