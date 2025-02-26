import os
import pandas as pd


def load_data():
    """
    Функция для загрузки CSV-файла в DataFrame.
    Используем sep=';' согласно структуре файла.
    """
    df = pd.read_csv('../data/news_dataset.csv', sep=';', encoding='utf-8')
    return df


def analyze_data():
    """
    Базовый анализ: вывод первых строк, информация о датасете, пропуски.
    А также сохранение промежуточных результатов в папку results.
    """
    # Создаем папку results, если её нет
    os.makedirs('../results', exist_ok=True)

    df = load_data()

    print("Первые 5 строк датасета:")
    print(df.head())

    print("\nИнформация о датасете:")
    print(df.info())

    print("\nПропуски по столбцам:")
    print(df.isnull().sum())

    # Сохраняем обработанные данные в CSV
    df.to_csv('results/processed_data.csv', index=False, encoding='utf-8')
    print("\nОбработанные данные сохранены в results/processed_data.csv")


if __name__ == "__main__":
    analyze_data()




