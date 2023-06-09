import sys
from functools import reduce

import numpy as np
import pandas as pd
import swifter
from gensim.models.fasttext import FastText
from matplotlib import pyplot as plt
from nltk.tokenize import sent_tokenize, word_tokenize
import nltk
import fasttext
import fasttext.util
from umap import UMAP


class EmbeddingTransformer:

    def __init__(self, df: pd.DataFrame, reduct: bool, n_components: int):
        self.ft = fasttext.load_model('cc.pl.300.bin')
        self.df = df
        self.reduct = reduct
        self.n_components = n_components

        self.final_df = pd.DataFrame()

    def tokenize(self):
        self.df['content'] = self.df['content'].swifter.apply(sent_tokenize)
        self.df['content'] = self.df['content'].swifter.apply(lambda x: [i for i in x if i != '.'])
        word_tokenizer = nltk.WordPunctTokenizer()
        self.df['content_url'] = self.df['content_url'].swifter.apply(lambda sent: word_tokenizer.tokenize(sent))
        self.df['content_url'] = self.df['content_url'].swifter.apply(
            lambda row: [token for token in row if len(token) > 2 and token != 'www']
        )

    def create_embeddings(self, col, sentence=False):
        if not sentence:
            self.df[col + '_ft'] = self.df[col].swifter.apply(
                lambda x: np.array([self.ft.get_word_vector(word) for word in x]).mean(axis=0))
        else:
            self.df[col + '_ft'] = self.df[col].swifter.apply(
                lambda x: np.array([self.ft.get_sentence_vector(sent) for sent in x]).mean(axis=0))

    def transform_df(self, col):
        if not self.reduct:
            cols_l = ['d_vec_' + str(i) for i in range(300)]
            self.final_df = pd.DataFrame(self.df[col].to_list(), columns=cols_l)
        else:
            reducer = UMAP(n_neighbors=100,
                           n_components=self.n_components,
                           metric='euclidean',
                           n_epochs=1000)
            cols_l = ['d_vec_' + str(i) for i in range(self.n_components)]
            array_reduced = reducer.fit_transform(self.df[col].to_list())
            self.final_df = pd.DataFrame(array_reduced, columns=cols_l)

        self.final_df['uid'] = self.df['uid'].to_list()

    def merge_data(self, col):
        self.df = pd.merge(self.df, self.final_df, on='uid', how='inner')
        self.df = self.df.drop(col, axis=1)


class FastextModeller:
    def __init__(self, text_array,
                 df = None,
                 vector_size=300,
                 window_size=5,
                 min_word=5,
                 down_sampling=1e-2,
                 epochs=100,
                 seed=1,
                 workers=4):
        self.fast_Text_model = None
        self.text_array = text_array
        self.df = df

        self.vector_size = vector_size
        self.window_size = window_size
        self.min_word = min_word
        self.down_sampling = down_sampling
        self.epochs = epochs
        self.seed = seed
        self.workers = workers

    def train(self):
        self.fast_Text_model = FastText(self.text_array,
                                        window=self.window_size,
                                        min_count=self.min_word,
                                        sample=self.down_sampling,
                                        workers=self.workers,
                                        seed=self.seed,
                                        epochs=self.epochs,
                                        vector_size=self.vector_size)

    def transform(self, sent):
        list_vec = [self.fast_Text_model.wv[word] for word in sent if word != '.']
        l_len = [len(self.fast_Text_model.wv[word]) for word in sent if word != '.']
        try:
            list_vec = np.array(list_vec).mean(axis=0)
        except Exception as e:
            print(l_len)
            print(e)
        return list_vec

    def col_transform(self):
        self.df['lemma'] = self.df['lemma'].swifter.apply(lambda x: self.transform(sent=x))

    def save_model(self):
        self.fast_Text_model.save("model_fastext_new")

    def load_model(self):
        self.fast_Text_model = FastText.load("model_fastext_new")


if __name__ == '__main__':
    reduct = False
    df = pd.read_csv('train_df_small.csv', index_col=0)
    emb = EmbeddingTransformer(df=df, reduct=reduct, n_components=300)
    emb.tokenize()
    emb.create_embeddings(col='content', sentence=True)
    emb.transform_df(col='content_ft')
    emb.merge_data(col='content_ft')
    df = emb.df
    df.to_csv('to_model_train.csv')

    if reduct:
        plt.scatter(
            df[df['code'] == 1]['d_vec_0'],
            df[df['code'] == 1]['d_vec_1'],
            color='red',
            label='ones'
        )
        plt.scatter(
            df[df['code'] == 0]['d_vec_0'],
            df[df['code'] == 0]['d_vec_1'],
            color='blue',
            label='zeros'
        )
        plt.legend(loc=2)

        # plt.xlim([-5, 5])
        # plt.ylim([-5, 5])

        plt.xlabel("Feature $x_1$", fontsize=12)
        plt.ylabel("Feature $x_2$", fontsize=12)

        plt.grid()
        plt.show()
