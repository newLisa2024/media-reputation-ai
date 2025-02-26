import os
from src.analysis import load_data
from transformers import pipeline


def sentiment_analysis():
    """
    Анализ тональности первых 5 заголовков новостей.
    Результаты сохраняются в текстовый файл.
    """
    # Создаем папку results, если её нет
    os.makedirs('../results', exist_ok=True)

    df = load_data()
    # Берем непустые заголовки и ограничиваемся 5 строками
    texts = df['title'].dropna().head(5).tolist()

    # Инициализируем модель для анализа тональности (многоязычная)
    sentiment_pipeline = pipeline("sentiment-analysis",
                                  model="nlptown/bert-base-multilingual-uncased-sentiment")

    # Открываем файл для записи результатов
    with open('../results/sentiment_results.txt', 'w', encoding='utf-8') as f:
        for text in texts:
            result = sentiment_pipeline(text)[0]
            f.write(f"Текст: {text}\n")
            f.write(f"Тональность: {result}\n\n")

    print("Результаты анализа тональности сохранены в results/sentiment_results.txt")


if __name__ == "__main__":
    sentiment_analysis()

