import os
import collections
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist, euclidean, cosine
from glob import glob
import boto3
import utils as ut
import json
import argparse
from sagemaker.predictor import RealTimePredictor, npy_serializer, numpy_deserializer
import sagemaker as sage
from sklearn.neighbors import KNeighborsClassifier

parser = argparse.ArgumentParser(description='Compares the inference in keras and in tf.')
parser.add_argument("--train_path", default='../train/',
                    help='Path to a directory that contains wav files.')
parser.add_argument("--test_path", default='../test/phillipe_2.wav',
                    help='Path to a directory that contains wav files.')
parser.add_argument("--endpoint_feature", default='sagemaker-tensorflow-2019-05-24-12-40-42-099',
                    help='Name of sagemaker endpoint of feature extraction')

FLAGS = parser.parse_args()

endpoint_feat_extract = FLAGS.endpoint_feature
boto_session = boto3.Session(region_name="eu-west-1")
session = sage.Session(boto_session=boto_session)
predictor = RealTimePredictor(endpoint_feat_extract, sagemaker_session=session,serializer=npy_serializer, deserializer=numpy_deserializer)
ArgsParameters = collections.namedtuple('ArgsParameters',['gpu', 'batch_size','net','ghost_cluster',
                                                         'vlad_cluster','bottleneck_dim','aggregation_mode',
                                                         'resume','loss','test_type'])
args = ArgsParameters(gpu='', batch_size=16, net='resnet34s',
                      ghost_cluster=2, vlad_cluster=8,bottleneck_dim=512,
                      aggregation_mode='gvlad',
                      resume='./model/weights.h5',
                      loss='softmax',
                      test_type='normal')

params = {'dim': (257, None, 1),
              'nfft': 512,
              'spec_len': 250,
              'win_length': 400,
              'hop_length': 160,
              'n_classes': 5994,
              'sampling_rate': 16000,
              'normalize': True}

def get_embedding(wav_path):
    specs = ut.load_data(wav_path, win_length=params['win_length'], sr=params['sampling_rate'],
                             hop_length=params['hop_length'], n_fft=params['nfft'],
                             spec_len=params['spec_len'], mode='eval')
    specs = np.expand_dims(np.expand_dims(specs, 0), -1)
    embedding_vox = predictor.predict(specs.astype(np.float32))
    return embedding_vox


def main():
    train_embs = list()
    label_embs = list()
    for i in sorted(glob(os.path.join(FLAGS.train_path, '*.wav'))):
        train_embs.append(get_embedding(i).reshape(-1))
        label_embs.append(i)
    X = np.vstack(train_embs)   
    print('Training...')
    neigh = KNeighborsClassifier(n_neighbors=1)
    print(X.shape)
    neigh.fit(X, range(len(label_embs))) 
    print('Predicting: '+FLAGS.test_path)
    test_emb = get_embedding(FLAGS.test_path).reshape(-1, params['nfft'])
    print('Parole similaire: '+label_embs[neigh.predict(test_emb)[0]])


if __name__ == '__main__':
    main()
