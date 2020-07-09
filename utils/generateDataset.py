import sys
import json
import tqdm
import argparse
from . import common

def generateDataset(options):
    dataset = json.load(open(options.data_file_path, "r"))

    word2vec = common.Word2Vec(options.vocab_file_path, options.word2vec_file_path)
    bert_tokenizer = common.TokensList2BertTokensIdsList()

    bar = tqdm.tqdm(total = len(dataset))
    for item in dataset:
        item["word_embeddings"] = word2vec(tokens = item["tokens"])
        item["bert_token_ids"] = bert_tokenizer(tokens = item["tokens"])
        item["bi_encoding"] = common.getBIencoding(num_tokens = len(item["tokens"]), chosen_positions = item["chosen_positions"])
        bar.update()

    bar.close()
    common.dumpJSON(dataset, options.dataset_file_path)

if(__name__ == "__main__"):
    project_root_path = common.getProjectRootPath()

    defaults = {}

    defaults["data_file_path"] = project_root_path / "data/test.json"
    defaults["dataset_file_path"] = project_root_path / "dataset/test.json"
    defaults["vocab_file_path"] = project_root_path / "data/utils/vocab.pkl"
    defaults["word2vec_file_path"] = project_root_path / "data/utils/word2vec.pkl"

    parser = argparse.ArgumentParser()

    parser.add_argument("--data_file_path", type = str, default = defaults["data_file_path"])
    parser.add_argument("--dataset_file_path", type = str, default = defaults["dataset_file_path"])
    parser.add_argument("--vocab_file_path", type = str, default = defaults["vocab_file_path"])
    parser.add_argument("--word2vec_file_path", type = str, default = defaults["word2vec_file_path"])

    options = parser.parse_args(sys.argv[1:])

    generateDataset(options)
