""" type_inspector.py

This module implements Inspector, a system that uses a pre-trained deep learning model to
infer data types from columns. The inspector must be trained before-hand or weights shall
be provided by MindsDB in order to work.
"""
import json

import torch
import pandas
import numpy
from transformers import BertModel, BertTokenizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder


class Inspector:
    """ Implementation of Inspector.
    """
    def __init__(self):
        """
        """
        self.data_ = pandas.DataFrame
        self.classifier_ = RandomForestClassifier(n_estimators=100)
        self.embeddings_ = []
        self.emb_targets_ = []

        self.targets_ = {}
        self.enc_ = LabelEncoder()
        self.tokenizer_ = torch.nn.Module
        self.model_ = torch.nn.Module
        self.device_ = torch.device

    def load_model(self):
        """
        """
        # Initialize the tokenizer with a pretrained model
        self.tokenizer_ = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model_ = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)

        # Set the device to GPU (cuda) if available, otherwise stick with CPU
        self.device_ = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model_ = self.model_.to(self.device_)
        self.model_.eval()

    def load_data(self, data_path: str, target_path: str):
        """
        """
        self.data_ = pandas.read_csv(data_path, engine='python', nrows=1000)
        if target_path is not None:
            f = open(target_path, 'r')
            self.targets_ = json.loads(f.read())

            for t in self.data_.columns.to_list():
                if t not in self.targets_:
                    print(t)
                    raise ValueError(f"column {t} not in target list.")

    def get_column_embeddings(self, col_name: str):
        """
        """
        data = self.data_[col_name].astype(str).to_list()
        data_as_str = ' '.join(data)
        embeddings = []
        for i in range(0, len(data_as_str), 512):
            tokens = self.tokenizer_.encode(data_as_str[i:i + 512])
            ids = numpy.asarray(tokens)
            # Convert the list of IDs to a tensor of IDs
            ids = torch.LongTensor(ids)
            ids = torch.unsqueeze(ids, 0)
            ids = ids.to(self.device_)
            with torch.no_grad():
                out = self.model_(input_ids=ids)
            # we only want the hidden_states
            hidden_states = out[2]
            sentence_embedding = torch.mean(hidden_states[-1], dim=1).squeeze()
            embeddings.append(sentence_embedding.cpu().numpy())

        return embeddings

    def add_data(self):
        """
        """
        emb_data = []
        tgt_data = []
        for col_name, target in self.targets_.items():
            emb = self.get_column_embeddings(col_name)
            tgt_data += [target] * len(emb)
            emb = numpy.asarray(emb)
            print(emb.shape)
            emb_data.append(emb)
        self.embeddings_ += emb_data
        self.emb_targets_ += tgt_data

    def train(self):
        """
        """
        emb_data = numpy.concatenate(self.embeddings_, axis=0).reshape((-1, 768))
        tgt_data = self.enc_.fit_transform(self.emb_targets_)
        trainX, testX, trainY, testY = train_test_split(emb_data, tgt_data)

        self.classifier_ = self.classifier_.fit(trainX, trainY)

        ground_truth = self.enc_.inverse_transform(testY)
        prediction = self.enc_.inverse_transform(self.classifier_.predict(testX))

        print(classification_report(ground_truth, prediction))

    def run(self):
        """
        """
        col_names = self.data_.columns.to_list()
        emb_data = []
        col_list = []
        for col_name in col_names:
            emb = self.get_column_embeddings(col_name)
            emb = numpy.asarray(emb)
            col_list += [col_name] * emb.shape[0]
            emb_data.append(emb)
        emb_data = numpy.concatenate(emb_data, axis=0).reshape((-1, 768))
        print(len(col_list), emb_data.shape)
        prediction = self.enc_.inverse_transform(self.classifier_.predict(emb_data))

        results = pandas.DataFrame()
        results['column_name'] = col_list
        results['column_type'] = prediction

        return results


if __name__ == '__main__':

    ins = Inspector()
    ins.load_model()
    ins.load_data('../tests/data/airline_sentiment.csv',
                  '../tests/data/airline_sentiment.json')
    ins.add_data()
    ins.load_data('/srv/storage/ml/datasets/individual_household_power_comsumption/data.csv',
                  '/srv/storage/ml/datasets/individual_household_power_comsumption/data.json')
    ins.add_data()
    ins.load_data('/srv/storage/ml/datasets/used_car_price/data.csv',
                  '/srv/storage/ml/datasets/used_car_price/data.json')
    ins.add_data()
    print("training...")
    ins.train()
    print("")

    print("running...")
    # ins.load_data('../tests/data/stack_overflow_survey_sample.csv', None)
    # ins.load_data('/srv/storage/ml/datasets/individual_household_power_comsumption/data.csv', None)
    ins.load_data('/srv/storage/ml/datasets/airline_delays/data.csv', None)
    results = ins.run()
    print("")

    for col_name in results['column_name'].unique():
        weights = results[results['column_name'] == col_name].value_counts()
        weights = weights / weights.sum()
        inferred_types = weights.index.get_level_values('column_type').values
        probs = weights.values
        print(col_name)
        for t, pt in zip(inferred_types, probs):
            print(f"type: {t} with probability: {pt:2.1f}")
        print("")
