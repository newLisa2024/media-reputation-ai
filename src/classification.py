import os
import pandas as pd
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# Если вдруг используешь стоп-слова из NLTK, раскомментируй и скачай их:
# nltk.download('stopwords')
from nltk.corpus import stopwords


def run_classification():
    """
    Классификация новостей по темам на основе их заголовков.
    1) Загрузка и предобработка данных.
    2) Разделение на обучающую и тестовую выборки.
    3) Преобразование текста в числовые вектора (TfidfVectorizer).
    4) Обучение модели (LogisticRegression).
    5) Оценка результатов и сохранение отчета.
    """

    # Создаём папку results, если её нет
    os.makedirs('../results', exist_ok=True)

    # Шаг 1. Загрузка данных
    df = pd.read_csv('../data/news_dataset.csv', sep=';', encoding='utf-8')

    # Убедимся, что в датасете есть нужные столбцы
    # Предположим, что 'title' - это текст новости, а 'topic' - целевая метка
    df = df.dropna(subset=['title', 'topic'])  # Удалим строки, где нет заголовка или темы

    # Берём X - это заголовки, y - это тема
    X = df['title'].values
    y = df['topic'].values

    # Шаг 2. Разделение на обучающую и тестовую выборки
    # test_size=0.2 значит 20% данных пойдёт на тест
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Шаг 3. Преобразование текста в вектора
    # Укажем stop_words='english', если у нас новости на английском.
    # Если часть новостей на другом языке, можно либо убрать stop_words,
    # либо использовать собственный список.
    vectorizer = TfidfVectorizer(stop_words='english')

    # Преобразуем текст
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # Шаг 4. Обучение модели (LogisticRegression)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_tfidf, y_train)

    # Шаг 5. Оценка результатов
    predictions = model.predict(X_test_tfidf)
    report = classification_report(y_test, predictions)
    cm = confusion_matrix(y_test, predictions)

    print("Отчёт по классификации:\n", report)
    print("Матрица ошибок:\n", cm)

    # Сохраним отчёт в текстовый файл
    with open('../results/classification_report.txt', 'w', encoding='utf-8') as f:
        f.write("Classification Report:\n")
        f.write(report)
        f.write("\nConfusion Matrix:\n")
        f.write(str(cm))

    # При желании можно сохранить модель и vectorizer, чтобы потом не обучать заново
    joblib.dump(model, '../results/news_classifier.pkl')
    joblib.dump(vectorizer, '../results/vectorizer.pkl')
    print("Модель и векторизатор сохранены в папке results.")


if __name__ == "__main__":
    run_classification()
