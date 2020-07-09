import re
import sys
import json
import tqdm
import random
import argparse
import functools
import numpy as np
from collections import defaultdict
from nested_lookup import nested_lookup
from nltk.parse import CoreNLPParser
from . import common

random.seed(1911)

def shortenEntityNames(city_type_entity_data):
    for city in city_type_entity_data:
        for type in city_type_entity_data[city]:
            for entity_id in city_type_entity_data[city][type]:
                city_type_entity_data[city][type][entity_id]["name"] = city_type_entity_data[city][type][entity_id]["name"].split(", ", 1)[0]

def getGoldAndNegativeSamplesDistances(weights, chosen_locations, candidate_locations, num_negative_samples):
    candidate_locations = np.array(candidate_locations)

    candidate_distances = np.array(common.getCandidateDistances(candidate_locations = candidate_locations, locations = chosen_locations))
    candidate_scores = np.dot(candidate_distances, np.array(weights))

    gold_index = np.argmin(candidate_scores)

    if(np.count_nonzero(candidate_scores == candidate_scores[gold_index]) > 1):
        raise Exception()

    negative_sample_indices = np.argpartition(candidate_scores, int(0.35 * num_negative_samples))[:(num_negative_samples + 1)].tolist()
    negative_sample_indices.remove(gold_index)

    return candidate_distances[gold_index].tolist(), candidate_distances[negative_sample_indices].tolist()

def generateTrainData(options):
    data = []

    tokenizer = CoreNLPParser(url = "http://localhost:9000")

    metonyms = json.load(open(options.metonyms_file_path, "r"))
    templates = json.load(open(options.templates_file_path, "r"))
    city_type_entity_data = json.load(open(options.city_type_entity_data_file_path, "r"))
    shortenEntityNames(city_type_entity_data)

    grouper = {"close1": "close", "iclose1": "close", "close2": "close", "iclose2": "close", "far1": "far", "ifar1": "far", "far2": "far", "ifar2": "far", "mixed": "mixed", "imixed": "mixed"}

    counts = defaultdict(int)

    bar = tqdm.tqdm(total = options.num_examples)
    while(1):
        try:
            cities = list(city_type_entity_data.keys())
            city = random.choice(cities)

            types = list(city_type_entity_data[city].keys())
            type = random.choice(random.choice(types))

            template = random.choice(list(templates.keys()))
            weights = templates[template]["weights"]
            group = templates[template]["group"]

            if(counts[grouper[group]] == (options.num_examples / len(set(grouper.values())))):
                continue

            if(len(city_type_entity_data[city][type]) - len(weights) < 1 + options.num_negative_samples):
                continue

            chosen_metonym = random.choice(metonyms[type])

            chosen_items = random.sample(list(city_type_entity_data[city][type].values()), len(weights))
            chosen_ids, chosen_names, chosen_locations = nested_lookup("id", chosen_items), nested_lookup("name", chosen_items), nested_lookup("location", chosen_items)

            question = functools.reduce(lambda x, y: x.replace("LOCATION", y, 1), chosen_names, template.replace("ENTITY", chosen_metonym))
            tokens = [token.replace(" ", "") for token in list(tokenizer.tokenize(question))]

            tokens_positions = " ".join(["%s_%d" % (token, index) for index, token in enumerate(tokens)])
            chosen_tokens_list = list(map(lambda i: re.findall("_\d+ ".join(list(map(re.escape, i))) + "_\d+", tokens_positions)[0], list(map(lambda i: list(tokenizer.tokenize(i)), chosen_names))))
            chosen_positions = [list(map(lambda x: int(x.rsplit("_", 1)[1]), chosen_entity_tokens.split(" "))) for chosen_entity_tokens in chosen_tokens_list]

            candidate_locations = [entity_item["location"] for entity_id, entity_item in city_type_entity_data[city][type].items() if entity_id not in chosen_ids]
            gold_distances, negative_samples_distances = getGoldAndNegativeSamplesDistances(weights, chosen_locations, candidate_locations,  options.num_negative_samples)

            item = {
                "group": group,
                "tokens": tokens,
                "chosen_positions": chosen_positions,
                "gold_distances": gold_distances,
                "negative_samples_distances": negative_samples_distances,
            }

            data.append(item)
            counts[grouper[group]] += 1

            bar.update()
            if(bar.n == options.num_examples):
                break
        except:
            pass

    bar.close()

    random.shuffle(data)
    common.dumpJSON(data, options.data_file_path)

if(__name__ == "__main__"):
    project_root_path = common.getProjectRootPath()

    defaults = {}

    defaults["num_examples"] = 6000
    defaults["num_negative_samples"] = 500
    defaults["metonyms_file_path"] = project_root_path / "data/utils/metonyms.json"
    defaults["templates_file_path"] = project_root_path / "data/utils/templates.json"
    defaults["city_type_entity_data_file_path"] = project_root_path / "data/utils/city_type_entity_data.json"
    defaults["data_file_path"] = project_root_path / "data/train.json"

    parser = argparse.ArgumentParser()

    parser.add_argument("--num_examples", type = int, default = str(defaults["num_examples"]))
    parser.add_argument("--num_negative_samples", type = int, default = str(defaults["num_negative_samples"]))
    parser.add_argument("--metonyms_file_path", type = str, default = str(defaults["metonyms_file_path"]))
    parser.add_argument("--templates_file_path", type = str, default = str(defaults["templates_file_path"]))
    parser.add_argument("--city_type_entity_data_file_path", type = str, default = str(defaults["city_type_entity_data_file_path"]))
    parser.add_argument("--data_file_path", type = str, default = str(defaults["data_file_path"]))

    options = parser.parse_args(sys.argv[1:])

    generateTrainData(options)
