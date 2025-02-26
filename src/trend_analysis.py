import os
import pandas as pd
import matplotlib.pyplot as plt


def run_trend_analysis():
    """
    Анализ трендов во времени:
    1. Загружаем данные.
    2. Преобразуем столбец published_date в datetime.
    3. Группируем данные по месяцам и считаем количество новостей.
    4. Строим и сохраняем график трендов.
    """
    # Создаем папку results, если её нет
    os.makedirs('../results', exist_ok=True)

    # Загружаем данные
    df = pd.read_csv('../data/news_dataset.csv', sep=';', encoding='utf-8')

    # Преобразуем столбец published_date в datetime (если формат отличается, можно указать параметр format)
    df['published_date'] = pd.to_datetime(df['published_date'], errors='coerce')

    # Отбрасываем строки с нераспознанными датами
    df = df.dropna(subset=['published_date'])

    # Группируем данные по месяцу
    df['year_month'] = df['published_date'].dt.to_period('M')
    trend = df.groupby('year_month').size()

    # Строим график
    plt.figure(figsize=(10, 5))
    trend.plot()
    plt.title('Тренд новостей по месяцам')
    plt.xlabel('Месяц')
    plt.ylabel('Количество новостей')
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Сохраняем график в файл
    trend_file = '../results/trend_analysis.png'
    plt.savefig(trend_file)
    print("График трендов сохранен в:", trend_file)
    plt.show()


if __name__ == '__main__':
    run_trend_analysis()
