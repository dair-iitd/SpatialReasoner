import os
import sys
import json
import tqdm
import torch
import argparse
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataset import Dataset
from . import net
from utils import common

class TestExample(Dataset):
    def __init__(self, item, entity_location_map):
        candidate_ids = []
        candidate_locations = []
        for candidate_id, candidate_location in entity_location_map.items():
            if(candidate_id not in item["chosen_ids"]):
                candidate_ids.append(candidate_id)
                candidate_locations.append(candidate_location)

        num_tokens = len(item["tokens"])
        num_candidates = len(candidate_ids)

        candidate_distances = common.getCandidateDistances(candidate_locations = candidate_locations, locations = item["chosen_locations"])
        candidate_distance_encodings = common.getDistanceEncodings(num_tokens = num_tokens, chosen_positions = item["chosen_positions"], candidate_distances = candidate_distances)

        candidate_bid_encodings = torch.cat([torch.tensor(item["bi_encoding"]).unsqueeze(0).expand(num_candidates, num_tokens, 2), torch.tensor(candidate_distance_encodings).unsqueeze(2)], dim = 2)

        self.length = len(candidate_ids)
        self.candidate_ids = candidate_ids
        self.candidate_bid_encodings = pad_sequence(candidate_bid_encodings, batch_first = True, padding_value = 0)

    def __getitem__(self, index):
        return (self.candidate_ids[index], self.candidate_bid_encodings[index])

    def __len__(self):
        return self.length

def getAnalysisTable(results):
    metrics = ["N", "Acc@1", "Acc@3", "Acc@5", "Acc@10", "Acc@30", "MR", "MRR", "DistG"]
    df = pd.DataFrame(index = metrics)

    grouper1 = {
        "close1": "close",
        "iclose1": "close",
        "close2": "close",
        "iclose2": "close",
        "far1": "far",
        "ifar1": "far",
        "far2": "far",
        "ifar2": "far",
        "mixed": "mixed",
        "imixed": "mixed"
    }

    grouper2 = {
        "close1": "non-distractor",
        "iclose1": "distractor",
        "close2": "non-distractor",
        "iclose2": "distractor",
        "far1": "non-distractor",
        "ifar1": "distractor",
        "far2": "non-distractor",
        "ifar2": "distractor",
        "mixed": "non-distractor",
        "imixed": "distractor"
    }

    group_properites = defaultdict(lambda: {"ranks": [], "distances": []})
    for item in results:
        group_properites[grouper1[item["group"]]]["ranks"].append(item["rank"])
        group_properites[grouper1[item["group"]]]["distances"].append(item["distances"])
        group_properites[grouper2[item["group"]]]["ranks"].append(item["rank"])
        group_properites[grouper2[item["group"]]]["distances"].append(item["distances"])
        group_properites["aggregate"]["ranks"].append(item["rank"])
        group_properites["aggregate"]["distances"].append(item["distances"])

    for group, properties in group_properites.items():
        metrics = common.getMetrics(properties["ranks"], properties["distances"])
        df.insert(0, group, metrics)

    table = common.getTable(df.T)
    return table

def test(options):
    dataset = json.load(open(options.dataset_file_path))
    city_type_entity_location = json.load(open(options.city_type_entity_location_file_path))
    print("Testing on %d examples" % (len(dataset)))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Running on %s device" % device)

    model = None
    if(options.model == "SPNetNoDRL"):
        model = net.SPNetNoDRL()
    if(options.model == "SPNet"):
        model = net.SPNet()
    if(options.model == "BertSPNetNoDRL"):
        model = net.BertSPNetNoDRL(mode = "test")
    if(options.model == "BertSPNet"):
        model = net.BertSPNet(mode = "test")
    if(model is None):
        raise Exception("Model should be one of SPNetNoDRL, SPNet, BertSPNetNoDRL, BertSPNet")

    print("Using %s model" % (options.model))
    model.to(device)

    print("Loading saved model from %s" % (options.model_file_path))
    model.load_state_dict(torch.load(options.model_file_path, map_location = torch.device(device)))

    print()
    print("Batch Size: %d" % (options.batch_size))
    print()

    results = []

    print("Testing . . .")
    bar = tqdm.tqdm(total = len(dataset))
    with torch.no_grad():
        for item in dataset:
            entity_location_map = city_type_entity_location[item["gold_id"].split("_")[0]][item["gold_id"].split("_")[1]]

            test_example = TestExample(item, entity_location_map)
            test_example_loader = torch.utils.data.DataLoader(test_example, batch_size = options.batch_size, shuffle = False, num_workers = 0)

            word_embeddings = torch.tensor(item["word_embeddings"]).to(device)
            bert_token_ids = torch.tensor(item["bert_token_ids"]).to(device)

            predictions = {}
            for batch in test_example_loader:
                candidate_ids, candidate_bid_encodings = batch[0], batch[1].to(device)
                batched_word_embeddings = word_embeddings.unsqueeze(0).expand(candidate_bid_encodings.size(0), word_embeddings.size(0), word_embeddings.size(1))
                batched_bert_token_ids = bert_token_ids.unsqueeze(0).expand(candidate_bid_encodings.size(0), bert_token_ids.size(0))
                outputs = model(word_embeddings = batched_word_embeddings, bert_token_ids = batched_bert_token_ids, bid_encoding = candidate_bid_encodings)
                candidate_scores = outputs.data.tolist()
                predictions.update(dict(zip(candidate_ids, candidate_scores)))

            sorted_candidate_ids = [candidate_id for candidate_id, _ in sorted(predictions.items(), key = lambda i: i[1])]
            item["rank"] = sorted_candidate_ids.index(item["gold_id"]) + 1
            item["distances"] = common.getCandidateDistances(candidate_locations = [entity_location_map[entity_id] for entity_id in sorted_candidate_ids[:3]], locations = [entity_location_map[item["gold_id"]]])

            results.append(item)
            bar.update()

    bar.close()

    table = getAnalysisTable(results)
    print()
    print("Results")
    print(table)

if(__name__ == "__main__"):
    project_root_path = common.getProjectRootPath()

    defaults = {}

    defaults["dataset_file_path"] = project_root_path / "dataset/dev.json"
    defaults["batch_size"] = 1000
    defaults["model"] = "SPNet"
    defaults["model_file_path"] = "models/SPNet.th.epoch13"
    defaults["city_type_entity_location_file_path"] = project_root_path / "data/utils/city_type_entity_location.json"

    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_file_path", type = str, default = defaults["dataset_file_path"])
    parser.add_argument("--batch_size", type = int, default = defaults["batch_size"])
    parser.add_argument("--model", type = str, default = defaults["model"])
    parser.add_argument("--model_file_path", type = str, default = defaults["model_file_path"])
    parser.add_argument("--city_type_entity_location_file_path", type = str, default = defaults["city_type_entity_location_file_path"])

    options = parser.parse_args(sys.argv[1:])

    test(options)
