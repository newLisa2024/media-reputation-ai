import os
import matplotlib.pyplot as plt
import seaborn as sns
from src.analysis import load_data  # Импортируем функцию загрузки данных


def visualize_topics():
    """
    Построение графика распределения новостей по темам.
    """
    # Создаем папку results, если её нет
    os.makedirs('../results', exist_ok=True)

    df = load_data()

    print("Распределение тем:")
    print(df['topic'].value_counts())

    plt.figure(figsize=(10, 5))
    sns.countplot(x='topic', data=df, order=df['topic'].value_counts().index)
    plt.title('Распределение новостей по темам')
    plt.xlabel('Тема')
    plt.ylabel('Количество')
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Сохраняем график в папку results
    plt.savefig('results/topics_distribution.png')
    print("График распределения тем сохранен как results/topics_distribution.png")
    plt.show()


def visualize_languages():
    """
    Построение круговой диаграммы распределения новостей по языкам.
    """
    # Создаем папку results, если её нет
    os.makedirs('../results', exist_ok=True)

    df = load_data()

    print("Распределение языков:")
    print(df['lang'].value_counts())

    plt.figure(figsize=(6, 6))
    df['lang'].value_counts().plot(kind='pie', autopct='%1.1f%%')
    plt.title('Языки новостей')
    plt.ylabel('')

    # Сохраняем график языков
    plt.savefig('results/languages_distribution.png')
    print("Круговая диаграмма языков сохранена как results/languages_distribution.png")
    plt.show()


if __name__ == "__main__":
    visualize_topics()
    visualize_languages()

