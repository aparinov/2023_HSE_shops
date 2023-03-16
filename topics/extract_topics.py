from bertopic import BERTopic
from bertopic.representation import MaximalMarginalRelevance

representation_model = MaximalMarginalRelevance(diversity=0.3)


def create_and_train_model(df, n_topics=10, embeddings=None):
    """
    создание и тренировка модели bertopic
    :param model:
    :return:
    """
    if embeddings is None:
        return get_topics(df['item_name'].tolist(), n_topics)

    return get_topics_own_embeddings(df['item_name'].tolist(), embeddings, n_topics)


def get_topics(preprocessed_data, n_topics=10, n_gram_range=(2, 2)):
    """
    создание и обучение модели bertopic на основе sentence transformer
    :param model:
    :return:
    """
    topic_model = BERTopic(language="russian", calculate_probabilities=True, verbose=True, n_gram_range=n_gram_range,
                           nr_topics=n_topics, representation_model=representation_model)
    topics, probs = topic_model.fit_transform(preprocessed_data)  # docs - просто тексты
    return topic_model, topics, probs


def get_topics_own_embeddings(preprocessed_data, embeddings, n_topics=10, n_gram_range=(2, 2)):
    """
    создание и обучение модели bertopic на основе собственных векторных представлений
    :param model:
    :return:
    """
    topic_model = BERTopic(representation_model=representation_model,
                           calculate_probabilities=True, verbose=True, n_gram_range=n_gram_range, nr_topics=n_topics)
    topics, probs = topic_model.fit_transform(preprocessed_data, embeddings)
    return topic_model, topics, probs


def get_topics_names(topic_model):
    """
    получение сопоставления номера топика с его названием
    :param model:
    :return:
    """
    topics = topic_model.get_topic_info()['Name'].tolist()
    topics_dict = {}
    for topic in topics:
        elems = topic.split("_")
        topics_dict[elems[0]] = "_".join(elems[:4])
    return topics_dict


def predict_topic(topic_model, text, n_count=1):
    """
    предсказание n топиков, по убыванию вероятности принадлежности
    :param model:
    :return:
    """
    similar_topics, similarity = topic_model.find_topics(text, top_n=n_count)
    # print(f'The top {num_of_topics} similar topics are {similar_topics}, and the similarities are {np.round(similarity,2)}')
    topics_dict = get_topics_names(topic_model)
    topics_ = [topics_dict[str(similar_topics[i])] for i in range(n_count)]

    return topics_


def predict_topics(texts, n_count):
    """
    предсказание n топиков для множества текстов, по убыванию вероятности принадлежности
    :param model:
    :return:
    """
    topics = [predict_topic(text, n_count) for text in texts]
    return topics


def save_model(topic_model, filename):
    topic_model.save(filename)
