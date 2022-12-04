import re
import sys
import os

import warnings
warnings.filterwarnings("ignore")

from gensim.models import Word2Vec
from gensim.models import FastText
import numpy
from helper import *


class Word2VecTrain:

    def __init__(self, vector_length, word2vec_vector_file):
        self.gadgets = []
        self.vector_length = vector_length
        self.vector_file = word2vec_vector_file
        print("----------")
        print("Word2VecTrain")
        print("vector_length:", vector_length)
        print("----------")

    def add_gadget(self, list_of_contexts):
        for context in list_of_contexts:
          self.gadgets.append(tokenize(context))
       
           
   
    def train_model(self):
        # Set min_count to 1 to prevent out-of-vocabulary errors
        model = Word2Vec(self.gadgets, min_count=1, size=self.vector_length, sg=1)
        print("Vocab size:", len(model.wv.vocab))
        self.embeddings = model.wv
        self.embeddings.save(self.vector_file)
        del model
        del self.gadgets
        #-----
        # print("fasttext")
        # model = FastText(size=self.vector_length, window=3, min_count=1)
        # model.build_vocab(self.gadgets)
        # model.train(self.gadgets, total_examples=len(self.gadgets), epochs=50) 
        # self.embeddings = model.wv
        # self.embeddings.save(self.vector_file)
        # del model
        # del self.gadgets