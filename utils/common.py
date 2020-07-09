import json
import torch
import pickle
import itertools
import numpy as np
import pandas as pd
import torch.nn as nn
from pathlib import Path
from tabulate import tabulate
from nested_lookup import nested_lookup
from transformers import BertTokenizer
from haversine import haversine, haversine_vector, Unit

# File Utils
def getProjectRootPath() -> Path:
    return Path(__file__).parent.parent

def create(path) -> None:
    path = Path(path)
    path.mkdir(parents = True, exist_ok = True)

def dumpJSON(data, path, sort_keys = False) -> None:
    create(Path(path).parent)
    json.dump(data, open(path, "w"), indent = 4, ensure_ascii = False, sort_keys = sort_keys)

# Distance Utils
def getDistance(a, b):
    return haversine(a, b, Unit.KILOMETERS)

def getCandidateDistances(candidate_locations, locations):
    num_candidates = len(candidate_locations)
    num_locations = len(locations)

    candidate_distances = np.empty([num_candidates, 0])
    for location in locations:
        distances = haversine_vector(np.repeat([location], num_candidates, axis = 0), candidate_locations, Unit.KILOMETERS)
        candidate_distances = np.hstack((candidate_distances, np.expand_dims(distances, axis = 1)))

    return candidate_distances.tolist()

def getSortedCandidatesByDistance(candidates, locations, k):
    candidate_ids = list(candidates.keys())
    candidate_locations = np.array(list(candidates.values()))

    candidate_distances = np.array(getCandidateDistances(candidates_locations, locations))
    candidate_distances = np.sum(np.sort(candidate_distances, axis = 1)[:, :k], axis = 1).tolist()

    sorted_candidates = sorted(list(zip(candidate_ids, candidate_distances)), key = lambda item: item[1])

    return sorted_candidates

# Dataset Utils
class Word2Vec():
    def __init__(self, vocab_file_path, word2vec_file_path):
        self.vocab = pickle.load(open(vocab_file_path, "rb"))
        self.pretrained_embeddings = pickle.load(open(word2vec_file_path, "rb"))

        self.vocab_size = len(self.vocab)
        self.word_embedding_dim = 128

        self.embedding_model = nn.Embedding(self.vocab_size, self.word_embedding_dim)
        self.embedding_model.weight = nn.Parameter(torch.tensor(self.pretrained_embeddings).float(), requires_grad = False)

    def __call__(self, tokens):
        indexes = torch.tensor([[self.vocab[token.lower()] if token.lower() in self.vocab else self.vocab_size for token in tokens]])
        word_embeddings = self.embedding_model(indexes)[0]
        return word_embeddings.tolist()

class TokensList2BertTokensIdsList():
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    def __call__(self, tokens):
        bert_token_ids = self.tokenizer.encode(tokens, add_special_tokens = False, is_pretokenized = True)
        return list(bert_token_ids)

def getBIencoding(num_tokens, chosen_positions):
    bi_encoding = torch.zeros((2, num_tokens))
    b_positions = [positions[0] for positions in chosen_positions]
    bi_encoding[0][b_positions] = 1
    i_positions = list(itertools.chain.from_iterable([positions[1:] for positions in chosen_positions]))
    bi_encoding[1][i_positions] = 1
    return bi_encoding.transpose(1, 0).tolist()

def getDistanceEncodings(num_tokens, chosen_positions, candidate_distances):
    num_candidates = len(candidate_distances)
    candidate_distances = torch.tensor(candidate_distances)
    candidate_distance_encodings = torch.zeros(num_candidates, num_tokens)
    for index, positions in enumerate(chosen_positions):
        candidate_distance_encodings[:, positions] = candidate_distances[:, index].unsqueeze(1)
    return candidate_distance_encodings.tolist()

# Analysis Utils
def getMetrics(ranks, distances):
    results = {"N": len(ranks)}

    ranks = np.array(ranks)

    results["Acc@1"] = np.mean(ranks == 1) * 100
    results["Acc@3"] = np.mean(ranks <= 3) * 100
    results["Acc@5"] = np.mean(ranks <= 5) * 100
    results["Acc@10"] = np.mean(ranks <= 10) * 100
    results["Acc@30"] = np.mean(ranks <= 30) * 100
    results["Acc@50"] = np.mean(ranks <= 50) * 100
    results["MR"] = np.mean(ranks)
    results["MRR"] = np.mean(1 / ranks)
    results["DistG"] = np.mean(list(map(lambda _distances: np.mean(np.min(_distances, axis = 1)), distances)))

    df = pd.DataFrame.from_dict(results, orient = "index")
    return df

def getTable(df):
    return tabulate(df, headers = "keys", tablefmt = "psql")
