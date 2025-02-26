import os
import pandas as pd
import gensim
from gensim import corpora
from gensim.models import LdaModel
import nltk
from nltk.corpus import stopwords
import string

# Загрузим список стоп-слов (если уже не загружен)
nltk.download('stopwords')


def preprocess_text(text):
    """
    Приводит текст к нижнему регистру, удаляет знаки препинания,
    разбивает текст на отдельные слова и удаляет стоп-слова.
    """
    # Приводим текст к нижнему регистру
    text = text.lower()
    # Удаляем знаки препинания
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Разбиваем текст на отдельные слова
    words = text.split()
    # Удаляем стоп-слова
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    return words


def run_topic_modeling(num_topics=5, num_words=10):
    """
    Функция, которая выполняет тематическое моделирование:
    - Загружает данные
    - Предобрабатывает заголовки
    - Строит модель LDA с заданным количеством тем
    - Сохраняет результаты в файл
    """
    # Создаем папку results, если её нет (относительный путь: на уровень выше, т.к. мы внутри src)
    os.makedirs('../results', exist_ok=True)

    # Загружаем данные
    # Можно использовать функцию load_data из analysis.py, если она настроена правильно,
    # например: from analysis import load_data; df = load_data()
    # Здесь для наглядности читаем напрямую:
    df = pd.read_csv('../data/news_dataset.csv', sep=';', encoding='utf-8')

    # Берем только те строки, где заголовок не пустой
    df = df[df['title'].notnull()]
    titles = df['title'].tolist()

    # Предобработка текстов: для каждого заголовка создаем список слов
    processed_titles = [preprocess_text(title) for title in titles]

    # Создаем словарь: сопоставляем каждому слову уникальный идентификатор
    dictionary = corpora.Dictionary(processed_titles)

    # Создаем корпус: для каждого заголовка получаем "мешок слов" – список пар (id, количество)
    corpus = [dictionary.doc2bow(text) for text in processed_titles]

    # Обучаем LDA-модель (параметр passes отвечает за количество проходов по данным)
    lda_model = LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=10)

    # Получаем найденные темы: каждая тема — набор слов с весами
    topics = lda_model.print_topics(num_words=num_words)

    # Сохраняем результаты в файл в папке results
    topics_file = '../results/topics_modeling_results.txt'
    with open(topics_file, 'w', encoding='utf-8') as f:
        for topic in topics:
            f.write(str(topic) + "\n")

    print("Найденные темы сохранены в:", topics_file)
    for topic in topics:
        print(topic)


if __name__ == "__main__":
    run_topic_modeling()
