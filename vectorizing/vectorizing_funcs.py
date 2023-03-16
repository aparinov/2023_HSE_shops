import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import wget
import zipfile
from gensim.test.utils import datapath
from zipfile import ZipFile
import gensim
from gensim.models import word2vec
import numpy as np
from tqdm import tqdm
import torch
import transformers
from transformers import BertTokenizer


BERT_tokenizer = BertTokenizer.from_pretrained("DeepPavlov/rubert-base-cased", do_lower_case=True)

def check_sim_quality(model):
    """
    проверяет как хорошо работает модель по построению embeddings на основе синонимов
    :param model:
    :return:
    """
    wget.download('https://rusvectores.org/static/testsets/ru_simlex999_tagged.tsv')
    res = model.evaluate_word_pairs('ru_simlex999_tagged.tsv')
    return res


def check_analog_quality(model):
    """
    проверяет как хорошо работает модель по построению embeddings на основе антонимов
    :param model:
    :return:
    """
    wget.download('https://rusvectores.org/static/testsets/ru_analogy_tagged.txt')
    res = model.evaluate_word_analogies('ru_analogy_tagged.txt')
    return res[0]

def get_bow_vector(data_samples: list, ngram_range: tuple = (1, 1), n_features: int = 1000):
    """
    реализация BoW для построения embeddings
    :param model:
    :return:
    """
    vectorizer = CountVectorizer(max_features=n_features, ngram_range=ngram_range)
    x_vector = vectorizer.fit_transform(data_samples)
    return x_vector, vectorizer


def get_tfidf_vector(data_samples: list, ngram_range: tuple = (1, 1), n_features: int = 1000):
    """
    реализация tf-idf для построения embeddings
    :param model:
    :return:
    """
    vectorizer = TfidfVectorizer(max_features=n_features, ngram_range=ngram_range)
    x_vector = vectorizer.fit_transform(data_samples)
    return x_vector, vectorizer


def load_model():
    """
    зашрузка предобученной word2vec модели
    :param model:
    :return:
    """
    model_url = 'http://vectors.nlpl.eu/repository/20/180.zip'
    m = wget.download(model_url)
    model_file = model_url.split('/')[-1]
    with ZipFile(model_file, 'r') as zObject:
        zObject.extractall(
            path="temp/")
    word2vec_path = 'temp/model.bin'
    w2v_model = gensim.models.KeyedVectors.load_word2vec_format(word2vec_path, binary=True)
    return w2v_model


def document_vector(text, w2v_model):
    """
    построение вектора для документа из модели word2vec
    :param model:
    :return:
    """
    doc = [word for word in text if word in w2v_model.wv]
    return np.mean(w2v_model.wv[doc], axis=0)



def get_pretrained_word2vec_vector(data_pos):
    """
    пполучение векторного представления из предобученной модели word2vec
    :param model:
    :return:
    """
    model = load_model()
    X = []
    for doc in tqdm(data_pos):
        X.append(document_vector(doc, model))

    # вектор для всех текстов (один текст это вектор из 300 элементов)
    X = np.array(X)
    return X, model


# data - токенезированный, лемматизированный список текстов, подается в виде токенов!
def get_retrained_word2vec_vector(data):
    """
    получение векторного представления из предобученной модели word2vec с дообучением
    :param model:
    :return:
    """
    model = load_model()
    #
    model_path = "pretrained.model"
    model.save(model_path)

    model = word2vec.Word2Vec.load(model_path)

    model.build_vocab(data, update=True)
    model.train(data, total_examples=model.corpus_count, epochs=5)

    return model


def get_trained_word2vec_vector(data, workers=4, min_count=10, window=10, sample=1e-3):
    """
    получение векторного представления из модели word2vec обученной полностью на представленных данных
    :param model:
    :return:
    """
    model_en = word2vec.Word2Vec(data, workers=workers, min_count=min_count, window=window, sample=sample)
    return model_en, [document_vector(doc, model_en) for doc in data]


def get_fasttext_vector():
    pass


# texts - просто список текстов, не предобработанных (а можно и предобработать)
def get_bert_vector(texts):
    """
    получение векторного представления с помощью предубученного BERT
    :param model:
    :return:
    """
    input_ids = []
    attention_masks = []

    for sent in texts:
        encoded_dict = BERT_tokenizer.encode_plus(
            sent,  # Sentence to encode.
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            max_length=20,  # Pad & truncate all sentences. # увеличить для описаний
            pad_to_max_length=True,
            return_attention_mask=True,  # Construct attn. masks.
            return_tensors='pt',  # Return pytorch tensors.
        )

        # Add the encoded sentence to the list.
        input_ids.append(encoded_dict['input_ids'])

        # And its attention mask (simply differentiates padding from non-padding).
        attention_masks.append(encoded_dict['attention_mask'])

    # Convert the lists into tensors.
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    return input_ids, attention_masks
