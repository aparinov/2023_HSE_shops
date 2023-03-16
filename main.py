import pandas as pd
from transform.transform_data import tokenize_lemmatize_text
import topics.extract_topics as extract_topics
from vectorizing import vectorizing_funcs
import sys

def get_topics_from_own_embeddings(n_topics = 15):
    df = pd.read_csv("../data/products.csv")
    df = df.rename(columns={"Наименование": "item_name"})
    df = df[['item_name', "category"]]
    df['item_name'] = df[['item_name']].applymap(tokenize_lemmatize_text)
    input_ids, attention_masks = vectorizing_funcs.get_bert_vector(df['item_name'].tolist())
    topic_model, topics, probs = extract_topics.create_and_train_model(df['item_name'].tolist(), embeddings=input_ids, n_topics=n_topics)
    return topic_model, topics, probs

def get_topics(n_topics = 15):
    df = pd.read_csv("../data/products.csv")
    df = df.rename(columns={"Наименование": "item_name"})
    df = df[['item_name', "category"]]
    topic_model, topics, probs = extract_topics.create_and_train_model(df['item_name'].tolist(),n_topics=n_topics)
    return topic_model, topics, probs


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "true":
            topic_model, topics, probs = get_topics_from_own_embeddings()
            pred = extract_topics.predict_topic(topic_model, "Развивающие книги для детей пиши стирай", n_count=3)
            print(pred)
    else:
        topic_model, topics, probs = get_topics()
        pred = extract_topics.predict_topic(topic_model, "Развивающие книги для детей пиши стирай", n_count=3)
        print(pred)







