# -*- coding: utf-8 -*-
import click
import logging
import os
from pathlib import Path
import numpy as np
import pandas as pd
from dotenv import find_dotenv, load_dotenv
from torch.utils.data import Dataset
from transformers import AutoFeatureExtractor, AutoModel, HubertModel, Wav2Vec2FeatureExtractor
import soundfile as sf


class KSUEMotionsDataset(Dataset):
    def __init__(self, ksu_emotions_metadata_path, raw_data_dir, model_path, transform=None, target_transform=None):
        self.metadata = pd.read_csv(ksu_emotions_metadata_path)
        self.processor = AutoFeatureExtractor.from_pretrained(model_path)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        audio_path = os.path.join("/content", self.metadata.iloc[idx, 2])
        speech, samplerate = sf.read(audio_path)

        preprocessed_speech = self.processor(speech, padding="max_length",  max_length = 250000,return_tensors="pt", sampling_rate = 16000).input_values
        label = self.metadata.iloc[idx, 3]

        return preprocessed_speech, label


def get_train_test_set(args, folds_metadata_dir: str, raw_data_dir:str, leave_one_out_num: int = 5,):
    # Load the data
    train_folds = []
    for i in range(5):
        if i != leave_one_out_num:
            train = pd.read_csv(folds_metadata_dir + f'/KSUEmotions_fold{i}.csv')
            train_folds.append(train)
    
    train_folds = pd.concat(train_folds)
    test_fold = pd.read_csv(folds_metadata_dir + f'/KSUEmotions_fold{leave_one_out_num}.csv')
    train_set = KSUEMotionsDataset(train_folds, raw_data_dir, args.SSL_model)
    test_set = KSUEMotionsDataset(test_fold, raw_data_dir, args.SSL_model)

    return train_set, test_set


