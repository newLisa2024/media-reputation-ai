import os
import pandas as pd
from transformers import pipeline
import matplotlib.pyplot as plt


def run_sentiment_prototype():
    """
    Простой прототип анализа тональности:
    1. Загружает данные из CSV (берём заголовки).
    2. Анализирует тональность первых 100 заголовков.
    3. Сохраняет результат в CSV.
    4. Строит диаграмму распределения.
    """
    # Вывод текущей рабочей директории для проверки
    current_dir = os.getcwd()
    print("Current working directory:", current_dir)

    # Формируем абсолютный путь к файлу с данными.
    # Предполагается, что папка data находится в корне проекта.
    data_file = os.path.join(current_dir, "data", "news_dataset.csv")
    print("Data file path:", data_file)

    # Загружаем данные
    df = pd.read_csv(data_file, sep=';', encoding='utf-8')
    df = df[df['title'].notnull()]  # Убираем строки с пустыми заголовками

    # Берем для демонстрации первые 100 заголовков
    df_sample = df.head(100).copy()

    # Инициализируем pipeline для анализа тональности
    sentiment_pipeline = pipeline(
        "sentiment-analysis",
        model="nlptown/bert-base-multilingual-uncased-sentiment"
    )

    sentiments = []
    for text in df_sample['title']:
        # Ограничиваем длину текста, чтобы избежать проблем (если заголовок очень длинный)
        result = sentiment_pipeline(text[:512])[0]
        sentiments.append(result['label'])

    # Добавляем результаты в DataFrame
    df_sample['sentiment'] = sentiments

    # Создаем папку results в корне проекта, если её нет
    results_folder = os.path.join(current_dir, "results")
    os.makedirs(results_folder, exist_ok=True)

    # Сохраняем результаты анализа в CSV
    output_csv = os.path.join(results_folder, "sentiment_prototype_results.csv")
    df_sample.to_csv(output_csv, index=False, encoding='utf-8')
    print(f"Результаты анализа тональности сохранены в: {output_csv}")

    # Строим диаграмму распределения тональностей
    sentiment_counts = df_sample['sentiment'].value_counts()
    plt.figure(figsize=(6, 4))
    sentiment_counts.plot(kind='bar', color=['green', 'red', 'blue', 'orange', 'purple'])
    plt.title('Распределение тональностей (прототип)')
    plt.xlabel('Тональность')
    plt.ylabel('Количество')
    plt.tight_layout()

    # Сохраняем диаграмму в файл
    output_png = os.path.join(results_folder, "sentiment_prototype_distribution.png")
    plt.savefig(output_png)
    print(f"Диаграмма тональностей сохранена в: {output_png}")
    plt.show()


if __name__ == "__main__":
    run_sentiment_prototype()
