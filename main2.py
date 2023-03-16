from bertopic import BERTopic
import translators as ts


def get_topics_names(topic_model):
    topics = topic_model.get_topic_info()['Name'].tolist()
    topics_dict = {}
    for topic in topics:
        elems = topic.split("_")
        topics_dict[elems[0]] = "_".join(elems[:4])
    return topics_dict


def predict_topic(topic_model, text, n_count=1):
    text = ts.translate_text(text, to_language="ru", translator="google")
    similar_topics, similarity = topic_model.find_topics(text, top_n=n_count)
    topics_dict = get_topics_names(topic_model)
    topics_ = [topics_dict[str(similar_topics[i])] for i in range(n_count)]

    return topics_


if __name__ == "__main__":
    topic_model = BERTopic.load("topic_model_cpu")
    pred = predict_topic(topic_model, "Развивающие книги для детей пиши стирай", n_count=1)
    print(pred)
