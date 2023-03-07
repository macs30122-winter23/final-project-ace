

import string
import pandas as pd
import matplotlib.pyplot as plt



import nltk
from nltk.corpus import stopwords
from gensim import corpora, models
from gensim.utils import effective_n_jobs
import pyLDAvis
import pyLDAvis.gensim_models
from wordcloud import WordCloud

import os
import gc

# functions to lemmatize news texts
STOP = set(nltk.corpus.stopwords.words('english') + list(string.punctuation) + ['``', "''", "’", "“", "”","–", "\'s"])

def get_lemmas(text):
    '''
    Gets lemmas for a string input, excluding stop words, punctuation, as well
    as a set of study-specific stop-words
    '''
    lemmas = [nltk.stem.WordNetLemmatizer().lemmatize(t)
              for t in nltk.word_tokenize((str(text).lower())) if t not in STOP
              ]
    return lemmas

def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=2):
    '''
    Computes Coherence values for LDA models with differing numbers of topics.

    Returns list of models along with their respective coherence values (pick
    models with the highest coherence)
    '''
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        model = models.ldamulticore.LdaMulticore(corpus=corpus,
                                                 id2word=dictionary,
                                                 num_topics=num_topics,
                                                 workers=effective_n_jobs(-1))
        model_list.append(model)
        coherence_model = models.coherencemodel.CoherenceModel(model=model,
                                                               corpus=corpus,
                                                               dictionary=dictionary,
                                                               coherence='u_mass')
        coherence_values.append(coherence_model.get_coherence())

    return model_list, coherence_values

def train_lda(topic_df, model_name, num_topics, workers):
    '''
    Trains LDA model on a dataframe of news articles, saves model to disk
    Inputs:
        topic_df: dataframe of news articles
        model_name: name of model to be saved
        num_topics: number of topics to be trained
        workers: number of workers to be used in training
    '''
    # Get lemmas for each article
    lemmas = topic_df['text'].apply(get_lemmas)
    #reduce memory load
    del topic_df
    gc.collect()
    # Initialize Series of lemmas as Gensim Dictionary for further processing
    dictionary = corpora.Dictionary(lemmas)
    # Convert dictionary into bag of words format: list of (token_id, token_count) tuples
    bow_corpus = [dictionary.doc2bow(text) for text in lemmas]
    model_list, coherence_values = compute_coherence_values(dictionary=dictionary,
                                                            corpus=bow_corpus,
                                                            texts=lemmas,
                                                            start=2,
                                                            limit=40,
                                                            step=6)
    # train LDA model
    ldamodel = models.ldamulticore.LdaMulticore(bow_corpus, num_topics=num_topics, id2word=dictionary, workers=workers, passes=20, iterations=400)
    ldamodel.save('{}.model'.format(model_name))

class Word_Cloud(object):
    '''
    the class to generate the word cloud of the news titles and texts
    from given keywords and medias

    Parameters
    ----------
    medias: list, the list of the medias to get the news
    keywords: list, the list of the keywords to get the news
    root: str, the root directory to save the results
    limit: int, the maximum number of the news to use
    custom_stopwords: list, the list of the custom stopwords to remove from the word cloud
    titles: list, the list of the titles of the word cloud
    WC: WordCloud, the word cloud object
    '''
    def __init__(self,
                 medias=['CNN'],
                 keywords=['gun'],
                 root='data',
                 limit=5000,
                 custom_stopwords=[None]):
        self.medias = medias
        self.keywords = keywords
        self.stopwords = stopwords.words('english')
        self.stopwords.extend(custom_stopwords)
        self.root = root
        self.limit = limit
        self.titles = [media + '_' + keyword for media in self.medias for keyword in self.keywords]
        self.WC = WordCloud(width=2000,
                            height=1000,
                            stopwords=self.stopwords,
                            background_color="white",
                            mode="RGBA",
                            max_words=50,
                            max_font_size=300)

    def load_data(self, i):
        '''
        load the data from the csv file according to the title
        and drop the duplicate and na rows
        '''
        return pd.read_csv(f'{os.path.join(self.root, self.titles[i].split("_")[0], self.titles[i])}.csv') \
            .drop_duplicates().dropna()

    def show(self):
        '''
        show and save the word cloud of the news titles and texts
        '''
        print('This may take a while has high demand on RAM', flush=True)
        for id, title in enumerate(self.titles):
            # load data
            print(f'Processing {id+1}/{len(self.titles)}', flush=True)
            df = self.load_data(id)
            if len(df) > self.limit:
                df = df[:self.limit]

            # extract the titles and texts
            texts = ' '.join(df.title).join(df.text)
            del df
            gc.collect()

            # generate the word cloud
            result = self.WC.generate(texts)
            del texts
            gc.collect()

            # show and save the word cloud
            plt.title(self.titles[id])
            plt.imshow(result, interpolation="bilinear")
            plt.axis("off")
            plt.savefig(f'{os.path.join("figures", self.titles[id])}.png')
            plt.show()
            print(f'Saved to {os.path.join("figures", self.titles[id])}.png', flush=True)
            plt.close()
            del result
            gc.collect()