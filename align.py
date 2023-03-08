#

import os
from abc import ABC
from collections import Counter

import gensim
import numpy as np
import pandas as pd
import scipy
from nltk.tokenize import sent_tokenize
from sklearn.cross_decomposition import CCA
from sklearn.metrics.pairwise import cosine_similarity


class Aligner(ABC):
    def __init__(self, method, source, target, w2id, id2w, mtxA, mtxB, trainvoc):
        self.method = method
        self.src = source
        self.tgt = target
        self.w2idA = w2id
        self.id2wB = id2w
        self.mtxA = mtxA
        self.mtxB = mtxB
        self.anchors = trainvoc

    def translate_mtx(self, mtx):
        """
        MTX -> MTX
        """
        pass

    def encode_input(self, words):
        """
        [STRING] -> MTX
        """
        embs = [self.mtxA[self.w2idA[w], :] for w in words]
        return np.vstack(embs)

    def decode_output(self, mtx, k=1):
        """
        MTX -> [[STRING]]
        """
        similarities = cosine_similarity(mtx, self.mtxB)
        most_similar = np.argsort(similarities, axis=1)[:, ::-1]
        topsims = np.sort(similarities, axis=1)[:, ::-1][:, :k]
        res = [[self.id2wB[i] for i in row[:k]] for row in most_similar]
        return res, topsims

    def translate_word(self, word, k=1):
        """
        STRING -> STRING
        """
        encoding = self.encode_input([word])
        translated = self.translate_mtx(encoding)
        decoded = self.decode_output(translated, k=k)
        return decoded[0][:k]

    def translate_words(self, words, k=1):
        """
        [STRING] -> [STRING]
        """
        encoding = self.encode_input(words)
        translated = self.translate_mtx(encoding)
        decoded, simscores = self.decode_output(translated, k=k)
        return decoded, simscores


class CCAAligner(Aligner):
    def set_params(self, cca):
        self.cca = cca

    def translate_mtx(self, mtx):
        return mtx

    def translate_word(self, word, k=1):
        tmpA = self.mtxA
        tmpB = self.mtxB
        self.mtxA, self.mtxB = self.cca.transform(tmpA, tmpB)
        res = super().translate_word(word, k=k)
        self.mtxA = tmpA
        self.mtxB = tmpB
        return res

    def translate_words(self, words, k=1):
        tmpA = self.mtxA
        tmpB = self.mtxB
        self.mtxA, self.mtxB = self.cca.transform(tmpA, tmpB)
        res, simscores = super().translate_words(words, k=k)
        self.mtxA = tmpA
        self.mtxB = tmpB
        return res, simscores


class SVDAligner(Aligner):
    def set_params(self, T):
        self.T = T

    def translate_mtx(self, mtx):
        return mtx.dot(self.T)


def align_cca(source, target):
    N_dims = source.shape[1]
    cca = CCA(n_components=N_dims, max_iter=2000)
    cca.fit(source, target)
    return cca


def align_svd(source, target):
    product = np.matmul(source.transpose(), target)
    U, s, V = np.linalg.svd(product)
    T = np.matmul(U, V)
    return T


def get_cca_aligner(model_a, model_b, anchorlist):
    # get wordmaps
    awords = list(sorted(list(model_a.wv.key_to_index)))
    bwords = list(sorted(list(model_b.wv.key_to_index)))
    w2idA = {w: i for i, w in enumerate(awords)}
    id2wA = {i: w for i, w in enumerate(awords)}
    w2idB = {w: i for i, w in enumerate(bwords)}
    id2wB = {i: w for i, w in enumerate(bwords)}

    # build the base matrices
    a_mtx = np.vstack([model_a.wv[w] for w in awords])
    b_mtx = np.vstack([model_b.wv[w] for w in bwords])

    # get the anchors
    a_anchor = np.vstack([a_mtx[w2idA[w], :] for w in anchorlist])
    b_anchor = np.vstack([b_mtx[w2idB[w], :] for w in anchorlist])

    # compute CCA
    cca = align_cca(a_anchor, b_anchor)

    # build and return the aligner
    aligner = CCAAligner('cca', model_a, model_b, w2idA, id2wB, a_mtx, b_mtx, anchorlist)
    aligner.set_params(cca)
    return aligner


def get_svd_aligner(model_a, model_b, anchorlist):
    # get wordmaps
    awords = list(sorted(list(model_a.wv.vocab)))
    bwords = list(sorted(list(model_b.wv.vocab)))
    w2idA = {w: i for i, w in enumerate(awords)}
    w2idB = {w: i for i, w in enumerate(bwords)}
    id2wB = {i: w for i, w in enumerate(bwords)}

    # build the base matrices
    a_mtx = np.vstack([model_a.wv[w] for w in awords])
    b_mtx = np.vstack([model_b.wv[w] for w in bwords])
    print(a_mtx.shape, b_mtx.shape)

    # get the anchors
    a_anchor = np.vstack([a_mtx[w2idA[w], :] for w in anchorlist])
    b_anchor = np.vstack([b_mtx[w2idB[w], :] for w in anchorlist])

    # get the translation matrix
    T = align_svd(a_anchor, b_anchor)

    # build and return the aligner
    aligner = SVDAligner('svd', model_a, model_b, w2idA, id2wB, a_mtx, b_mtx, anchorlist)
    aligner.set_params(T)
    return aligner


def JS_divergence(p, q):
    M = (p + q) / 2
    return 0.5 * scipy.stats.entropy(p, M, base=2) + 0.5 * scipy.stats.entropy(q, M, base=2)


def research_topic(keywords, t_align, forward_cnn, forward_nypost, model_general, model_cnn, model_nypost):

    # Gether the articles from topic-specified files.
    cnn_path = f"CNN_{keywords}"
    for f_name in os.listdir("./data/CNN"):
        if f_name.startswith(cnn_path):
            break
    cnn_file = open(os.path.join("./data/CNN", f_name), 'r')

    nypost_path = f"nypost_{keywords}"
    for f_name in os.listdir("./data/nypost"):
        if f_name.startswith(nypost_path):
            break
    nypost_file = open(os.path.join("./data/nypost", f_name), 'r')

    # Count the content cluster attrbution of each word, and generate the articles' coverage
    # representation vector, by proportion of each content cluster with the articles.
    word_intersec = []
    t_vec_cnn = np.zeros(300)
    df = pd.read_csv(cnn_file)
    all_cnn = 0
    for idx, row in df.iterrows():
        for st in sent_tokenize(str(row['text'])):
            sts_list = gensim.utils.simple_preprocess(st)
            hist = Counter(sts_list)
            for k in hist:
                num = t_align.get(k, -1)
                if num >= 0:
                    word_intersec.append(k)
                    all_cnn += hist[k]
                    t_vec_cnn[num] += hist[k]
    t_vec_cnn /= all_cnn

    t_vec_nypost = np.zeros(300)
    df = pd.read_csv(nypost_file)
    all_nypost = 0
    for idx, row in df.iterrows():
        for st in sent_tokenize(str(row['text'])):
            sts_list = gensim.utils.simple_preprocess(st)
            hist = Counter(sts_list)
            for k in hist:
                num = t_align.get(k, -1)
                if num >= 0:
                    word_intersec.append(k)
                    all_nypost += hist[k]
                    t_vec_nypost[num] += hist[k]
    t_vec_nypost /= all_nypost
    # Calculate their variance by two measures. It turns out that the JS works better than cosine
    # similarity, for it can distinguish words better.
    topic_js = JS_divergence(t_vec_nypost, t_vec_cnn)
    topic_cos = cosine_similarity(np.array(t_vec_nypost).reshape([1, -1]), np.array(t_vec_cnn).reshape([1, -1]))

    # Calculate the average cosine similarity of shared words by cnn embedding and nypost embedding.
    dis = []
    for wd in word_intersec:
        if wd in model_general.wv.key_to_index and wd in model_cnn.wv.key_to_index:
            vec_nypost = forward_nypost.translate_mtx(model_nypost.wv[wd])
            vec_cnn = forward_cnn.translate_mtx(model_cnn.wv[wd])
            cos_sim = cosine_similarity(np.array(vec_nypost).reshape([1, -1]), np.array(vec_cnn).reshape([1, -1]))
            dis.append(cos_sim)
    mean_c = np.mean(np.array(dis))

    return topic_js, topic_cos, mean_c, t_vec_cnn, t_vec_nypost
