import os
import pandas as pd
import spacy


def run_ner():
    """
    Извлечение именованных сущностей (NER) из заголовков новостей.
    1. Загружаем данные.
    2. Обрабатываем заголовки новостей с помощью spaCy.
    3. Извлекаем именованные сущности для каждого заголовка.
    4. Сохраняем результаты в CSV-файл в папке results.
    """
    # Создаем папку results, если её нет
    os.makedirs('../results', exist_ok=True)

    # Загружаем данные (обязательно проверь, что в файле есть столбец 'title')
    df = pd.read_csv('../data/news_dataset.csv', sep=';', encoding='utf-8')
    df = df[df['title'].notnull()]  # Убираем строки с пустыми заголовками
    titles = df['title'].tolist()

    # Загружаем модель spaCy для английского языка
    nlp = spacy.load('en_core_web_sm')

    # Список для сохранения результатов
    results = []
    for title in titles:
        doc = nlp(title)
        # Для каждого заголовка сохраняем список кортежей: (текст сущности, тип сущности)
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        results.append({'title': title, 'entities': entities})

    # Преобразуем результаты в DataFrame и сохраняем в CSV
    results_df = pd.DataFrame(results)
    results_file = '../results/ner_results.csv'
    results_df.to_csv(results_file, index=False, encoding='utf-8')
    print("Результаты NER сохранены в:", results_file)


if __name__ == '__main__':
    run_ner()
