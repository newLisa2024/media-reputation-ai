# Media Reputation AI

Проект **Media Reputation AI** предназначен для анализа новостных данных с использованием методов обработки естественного языка (NLP) для оценки репутации компаний в СМИ.

---

## Описание проекта

В данном проекте реализованы следующие возможности:

- **Анализ тональности (Sentiment Analysis):** Определение эмоциональной окраски заголовков новостей.
- **Тематическое моделирование (Topic Modeling):** Выявление скрытых тем в новостях с помощью методов, таких как LDA.
- **Классификация:** Разделение новостных сообщений по тематикам.
- **Извлечение именованных сущностей (NER):** Определение и выделение имен, организаций, локаций и других сущностей.
- **Анализ трендов (Trend Analysis):** Отслеживание динамики публикаций во времени.

Все результаты (CSV-файлы, отчёты, графики) сохраняются в папку `results/`. Подробное описание архитектуры и основных сложностей при переходе в продакшен доступно в [docs/system_design.md](docs/system_design.md).

---

## Установка

### Клонирование репозитория

```bash
git clone https://github.com/newLisa2024/media-reputation-ai
```

Перейдите в папку проекта:

```bash
cd media-reputation-ai
```

### Настройка виртуального окружения

- **Windows:**

  ```bash
  .venv\Scripts\activate
  ```

- **Unix/Linux/macOS:**

  ```bash
  source .venv/bin/activate
  ```

### Установка зависимостей

```bash
pip install -r requirements.txt
```

Убедитесь, что установлены все необходимые библиотеки (pandas, matplotlib, gensim, nltk, spacy, transformers и т.д.).

---

## Использование

После установки и активации окружения вы можете запускать модули из папки `src/`.

### Примеры команд:

- **Предобработка данных:**

  ```bash
  python src/analysis.py
  ```

- **Анализ тональности (прототип):**

  ```bash
  python src/sentiment_prototype.py
  ```

- **Классификация новостей:**

  ```bash
  python src/classification.py
  ```

- **Тематическое моделирование:**

  ```bash
  python src/topic_modeling.py
  ```

- **Извлечение именованных сущностей (NER):**

  ```bash
  python src/ner.py
  ```

- **Анализ трендов:**

  ```bash
  python src/trend_analysis.py
  ```

---

## Структура проекта

```
media-reputation-ai/
├── data/                    # Исходные данные (CSV-файл с новостями)
├── docs/                    # Документация (system_design.md и др.)
├── results/                 # Результаты анализа (CSV, графики, отчёты)
├── src/                     # Исходный код проекта
│   ├── analysis.py
│   ├── sentiment_prototype.py
│   ├── sentiment.py
│   ├── classification.py
│   ├── topic_modeling.py
│   ├── ner.py
│   └── trend_analysis.py
├── .venv/                   # Виртуальное окружение (необязательно хранить в репозитории)
├── requirements.txt         # Список зависимостей
└── README.md                # Описание проекта
```

---

## Документация

Подробное описание архитектуры системы, пайплайна обработки данных и основных сложностей при переходе в продакшен доступно в [docs/system_design.md](docs/system_design.md).

---

## Контакты

Если у вас возникнут вопросы или предложения по улучшению проекта, пожалуйста, свяжитесь со мной:

- **Email:** loskutovaelena50@gmail.com
- **GitHub:** https://github.com/newLisa2024
- **Telegram:** https://t.me/Elena_PromptLab

---
