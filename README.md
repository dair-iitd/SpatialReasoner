# SpatialReasoner

> This repository presents a detailed study of a spatial-reasoner using a simple artificially generated toy-dataset. This allows us to probe and study different aspects of spatial-reasoning in the absence of textual reasoning.
---

## Requirements

-   Python 3.4+
-   Linux-based system
---

## Installation

### Clone

> Clone this repository to your local machine.
```bash
git clone "https://github.com/dair-iitd/SpatialReasoner.git"
```

### Environment Setup

Please follow the instructions at the following link to set up anaconda. [https://www.digitalocean.com/community/tutorials/how-to-install-anaconda-on-ubuntu-18-04-quickstart](https://www.digitalocean.com/community/tutorials/how-to-install-anaconda-on-ubuntu-18-04-quickstart)
> Set up the conda environment
```bash
$ conda env create -f environment.yml
```

> Install the required python packages

```bash
$ conda activate spatial-reasoner
$ pip install -r requirements.txt
```
---

## Set up

### Stanford Core-NLP-Server
Please install the stanford core-nlp server library using the following link:
http://nlp.stanford.edu/software/stanford-corenlp-latest.zip
The server needs to run (on port 9000) when generating the test and train data.

Use the following command to run the server.

```bash
java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -preload tokenize,ssplit,pos,lemma,ner,parse,depparse -status_port 9000 -port 9000 -timeout 15000
```
To shut down the server use the following command:
```bash
wget "localhost:9000/shutdown?key=`cat /tmp/corenlp.shutdown`" -O -
```

## Description

The repository is used to generate a toy dataset using real-world entities. The data for entity names and locations have not uploaded due to licensing issues. Please refer to the [https://github.com/dair-iitd/TourismQA](https://github.com/dair-iitd/TourismQA) for crawling the data. Four different models have been provided with the default settings and the best epochs have been mentioned in the src/net.py. The src/test.py file shows a class-wise analysis  of results.

## Generating Data

To generate the train data, use the following command (defaults provided in file for all arguments): 
```bash
python -m utils.generateTrainData --num_examples 6000 --num_negative_samples 500 --metonyms_file_path "data/utils/metonyms.json" --templates_file_path "data/utils/templates.json" --city_type_entity_data_file_path "data/utils/city_type_entity_data.json" --data_file_path "data/train.json"
```

To generate the test data, use the following command (defaults provided in file for all arguments): 
```bash
$ python -m utils.generateTrainData --num_examples 2000 --metonyms_file_path "data/utils/metonyms.json" --templates_file_path "data/utils/templates.json" --city_type_entity_data_file_path "data/utils/city_type_entity_data.json" --data_file_path "data/test.json"
```

Note: 
1) To generate the dev data, use random seed: 2208
1) To generate the dev data, use random seed: 2303

## Generating Dataset

```bash
$ python -m utils.generateDataset --data_file_path "data/train.json" --dataset_file_path "dataset/train.json" --vocab_file_path "data/utils/vocab.pkl" --word2vec_file_path "data/utils/word2vec.pkl"
```
The above command can be used for dev and test data.

## Training

```bash
$ python -m src.train --dataset_file_path "dataset/train.json" --model "SPNet" --num_examples 6000 --batch_size 20 --num_epochs 15 --margin 0.5 --learning_rate 0.002 --num_negative_samples 40 --save_model_file_path "models/SPNet.th"
```

1. The defaults have been provided in the file.
2. Pre trained model can be specified using the --pretrained_model_path argument.
3. Model can be one of SPNetNoDRL, SPNet, BertSPNetNoDRL, BertSPNet and should match the model saved in pretrained_model_path (if specified).

## Testing

```bash
$ python -m src.test --dataset_file_path "dataset/test.json" --model "SPNet" --model_file_path "models/SPNet.th.epoch13" --city_type_entity_location_file_path "data/utils/city_type_entity_location.json"
```

1. The defaults have been provided in the file.
2. Model can be one of SPNetNoDRL, SPNet, BertSPNetNoDRL, BertSPNet and should match the model saved in model_file_path (if specified).
3. Batch size can be specified using --batch_size. It should be decreased if device memory is insufficient. Default is 1000.

## Reproducibility

The default parameters have been specified in the src/train.py.
1. For the non-bert models (i.e. SPNetNoDRL, SPNet), use torch and random seed both as 2208 and learning rate is 0.001.
2. For the bert models (i.e. BertSPNetNoDRL, BertSPNet), use torch and random seed both as 2207 and learning rate is 0.0002.

The following table shows the best epochs for each model:

| Model            | Best Epoch |
| ---              | ---        |
| SPNetNoDRL       | 3          | 
| SPNet            | 13         |
| BertSPNetNoDRL   | 10         |
| BertSPNet        | 14         | 

## License

[![License](https://img.shields.io/badge/License-Apache%202.0-yellowgreen.svg)](https://opensource.org/licenses/Apache-2.0) 

- **[Apache 2.0 license](https://opensource.org/licenses/Apache-2.0)**
