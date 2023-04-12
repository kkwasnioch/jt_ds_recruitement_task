import pandas as pd
import re
import spacy
import swifter


def cleanhtml(raw_html):
    clean = re.compile('<.*?>')
    cleantext = re.sub(clean, ' ', raw_html)
    cleantext = cleantext.replace("  ", " ")
    return cleantext


class TextPreprocesser:

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.stop = None
        self.nlp = spacy.load("pl_core_news_sm", disable=['parser', 'ner'])

    def full_text_preprocess(self, col):
        # more than spaces into separates sentences
        self.df[col] = self.df[col].str.replace(r'\s{2,}', ' . ', regex=True)
        # special char except dot
        self.df[col] = self.df[col].str.replace(r'[^\w. ]', ' ', regex=True)
        # links -> www.xyz.com
        self.df[col] = self.df[col].str.replace(r'\S+\.\S+', ' ', regex=True)
        # digits
        self.df[col] = self.df[col].str.replace(r'\d+', ' ', regex=True)
        self.df[col] = self.df[col].str.lower()
        # all words <= 2
        self.df[col] = self.df[col].str.replace(r'\b\w{1,2}\b', ' ', regex=True)
        # more than 1 whitespaces and characters except . and letters
        self.df[col] = self.df[col].str.replace(r'\s{2,}', ' ', regex=True)
        # stopwords
        self.df[col] = self.df[col].swifter.apply(
            lambda x: ' '.join([word for word in x.split() if word not in self.stop]))
        self.file.close()

    def short_text_preprocess(self, col):
        self.df[col] = self.df[col].str.lower()
        self.df[col] = self.df[col].str.replace(r'\d+', '', regex=True)
        self.df[col] = self.df[col].str.replace(r'[^\w. ]', ' ', regex=True)
        # stopwords
        self.df[col] = self.df[col].swifter.apply(
            lambda x: ' '.join([word for word in x.split() if word not in self.stop]))
        self.file.close()

    def lemma(self, col):
        lemma_text_list = []
        for doc in self.nlp.pipe(self.df[col].values, n_process=2, batch_size=1000):
            lemma_text_list.append(" ".join(token.lemma_ for token in doc))
        self.df[col] = lemma_text_list

    def stopwords_opener(self):
        with open('stop.txt', encoding='utf-8') as self.file:
            self.stop = [word.replace('\n', '') for word in self.file.readlines()]


class DatasetPreparer:
    def __init__(self):
        self.content_df = pd.read_csv('content.csv')

    def create_content(self):
        self.content_df = self.content_df.dropna(subset=['url', 'keywords'])
        self.content_df['content'] = self.content_df['content'].astype(str)
        self.content_df['keywords'] = self.content_df['keywords'].astype(str)
        self.content_df = self.content_df.drop_duplicates()
        self.content_df = self.content_df.groupby(
            ['url']).agg({'content': ' '.join, 'keywords': ', '.join}).reset_index()

    def transform_datasets(self, df, url):
        df = df.merge(url, on='uid', how='left')
        df = pd.merge(df, self.content_df, on='url', how='left')

        df = df.dropna(subset=['url', 'keywords'])
        df = df.drop_duplicates()
        df = transform_urls(df)
        df = df.drop(['url'], axis=1)
        df = df.groupby(['uid']).agg({'content': ' '.join,
                                      'keywords': ','.join,
                                      'code': 'mean',
                                      'content_url': ' '.join}).reset_index()
        return df


def transform_urls(df):
    df['content_url'] = df['url'].str.replace(r'^https://|[^\w.]+', ' ', regex=True)
    df['content_url'] = df['content_url'].str.findall(r'\S+\.\S+')
    df['content_url'] = df['content_url'].transform(lambda x: x[0])

    return df


if __name__ == '__main__':
    train_df = pd.read_csv('train.csv')
    train_urls = pd.read_csv('urls_train.csv')
    test_df = pd.read_csv('test.csv')
    test_urls = pd.read_csv('urls_test.csv')

    prep = DatasetPreparer()
    prep.create_content()
    train_df = prep.transform_datasets(df=train_df, url=train_urls)
    # test_df = prep.transform_datasets(df=test_urls, url=test_urls)

    text_prep = TextPreprocesser(df=train_df)
    text_prep.stopwords_opener()
    text_prep.full_text_preprocess(col='content')
    text_prep.short_text_preprocess(col='keywords')
    text_prep.lemma(col='content')
    text_prep.lemma(col='keywords')
    train_df = text_prep.df
    train_df.to_csv('train_df_small.csv')
