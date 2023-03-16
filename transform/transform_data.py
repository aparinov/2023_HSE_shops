# 1 - токенезация + удаление всего ненужного: пунктуация+взять от Татьяны доп символы, стоп-слова, числа.
# 2 - лемматизация (pymorphy)
import nltk
import re
import string
from natasha import Doc, MorphVocab, Segmenter, NewsEmbedding, NewsMorphTagger

nltk.download('punkt')
nltk.download('stopwords')

from pymorphy2 import MorphAnalyzer

TOK_TOK_TOKENIZER = nltk.tokenize.ToktokTokenizer()
WORD_PUNCT_TOKENIZER = nltk.tokenize.WordPunctTokenizer()

MORPH = MorphAnalyzer()

STOPWORDS = nltk.corpus.stopwords.words('russian')
PUNCTS = string.punctuation + '\n\xa0«»\t—...'


def tokenize_lemmatize_text(text, tokenizer=WORD_PUNCT_TOKENIZER, natasha=False, postagging=False):
    '''to working with df use applymap for whole column df.applymap(tokenize_lemmatize_text)'''
    if tokenizer is None and not natasha:
        print("use natasha or write tokenizer")
        return ""
    # for item in data:  # item - текст, data - список текстов
    if natasha:
        text = tokenize_lemmatize_text_natasha(text, postagging)
        return text
    tokens = tokenize_text(text, tokenizer)
    text = lemmatize_text(tokens, postagging)
    return text


def tokenize_text(text, tokenizer):
    text_lower = text.lower()  # convert words in a text to lower case
    tokens = tokenizer.tokenize(text_lower)  # splits the text into tokens (words)

    # remove punct and stop words from tokens
    return remove_noise(tokens)


def lemmatize_text(tokens, postagging):
    if postagging:
        # print(tokens)
        text_lemmatized = [MORPH.parse(x)[0].normal_form + "_" + MORPH.parse(x)[0].tag.POS for x in tokens if MORPH.parse(x)[0].tag.POS is not None]
        return ' '.join(text_lemmatized)
    text_lemmatized = [MORPH.parse(x)[0].normal_form for x in tokens]  # apply lemmatization to each word in a text
    text = ' '.join(text_lemmatized)  # unite all stemmed words into a new text
    return text


def tokenize_lemmatize_text_natasha(text, postagging):
    doc = Doc(text)
    doc.segment(Segmenter())
    doc.tag_morph(NewsMorphTagger(NewsEmbedding()))
    for token in doc.tokens:
        token.lemmatize(MorphVocab())
    if postagging:
        tokens = [token.lemma + "_" + token.pos for token in doc.tokens]
        return ' '.join(remove_noise(tokens))

    tokens = [token.lemma for token in doc.tokens]
    return ' '.join(remove_noise(tokens))


def remove_noise(tokens):
    text = [word for word in tokens if
            (word not in string.punctuation and
             word not in STOPWORDS and
             word not in string.digits and
             not any([char.isdigit() for char in word]))]
    words = re.compile(r'\b\w+\b')
    new_text = words.findall(' '.join(text))
    return new_text

# def main(df):
#     df = df['text'].applymap(tokenize_lemmatize_text)
#     # df = df['text'].applymap(tokenize_lemmatize_text,natasha = True)
#     # df = df['text'].applymap(tokenize_lemmatize_text, tokenizer=TOK_TOK_TOKENIZER)
#     return df
