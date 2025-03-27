
# Исторический ВестникЪ 📜📰

*Представленный ниже проектъ "Исторический ВестникЪ" былъ сотворёнъ усердными трудами въ рамкахъ курса учебнаго (сирѣчь образовательнаго), именуемаго:*

📜 **"Дѣлаемъ свой AI-продуктъ на базѣ ChatGPT или другихъ LLM моделей"** 📜

*Гдѣ умы пытливые наставлялись искусству обученія машинъ и созданію разума механическаго.*

*Дабы всякъ желающій могъ ознакомиться съ первоисточникомъ знаній сихъ, оставляемъ здѣсь ссылку на курсъ оный на платформѣ Stepik обрѣтающійся:*

🔗 [https://stepik.org/course/178846/](https://stepik.org/course/178846/) 🔗

*Исполнено съ прилежаніемъ и надеждой на пользу общественную.*

***
**Добро пожаловать в редакцию газеты "Хронографъ"!** Это приложение позволяет заглянуть в прошлое и получить сводку "горячих" новостей на выбранную вами историческую дату, сдобренную толикой юмора и стилизацией под старину.

***
✨ **Дисклеймеръ отъ Редакціи** ✨

*Достопочтенная публика! Редакція "Хронографа" симъ извѣщаетъ, что содержаніе настоящихъ депешъ имѣетъ своей цѣлью единственно развлеченіе и просвѣщеніе умовъ любознательныхъ. Основываясь на дѣйствительныхъ событіяхъ минувшихъ лѣтъ, авторы сихъ замѣтокъ позволяютъ себѣ нѣкоторые художественные вольности, сатирическія преувеличенія и даже вымыселъ ради пущей занимательности. Посему, просимъ не принимать все написанное за чистую монету и историческую правду въ послѣдней инстанціи. Любыя совпаденія съ реально происходившими событіями, а равно съ дѣйствительными пороками и добродѣтелями нашихъ предковъ, считать случайными и непреднамѣренными. Читайте съ удовольствіемъ и здравымъ скептицизмомъ!*
***

## 1. Основная идея, уникальность, целевая аудитория

### Идея проекта
"Исторический ВестникЪ" — это веб-приложение, которое генерирует стилизованные под старину новостные заметки на основе реальных исторических событий, произошедших около выбранной пользователем даты. Приложение использует большую языковую модель (LLM) для переосмысления исторических фактов, добавления юмора, вымышленных деталей и создания текста в духе газет прошлых веков.

### Уникальность
*   **Сочетание фактов и вымысла:** Проект не просто пересказывает историю, а креативно её интерпретирует, создавая уникальный развлекательный и образовательный контент.
*   **RAG для контекста:** Используется техника Retrieval-Augmented Generation (RAG) для поиска релевантных исторических событий в векторной базе данных, что обеспечивает фактологическую основу для генерации.
*   **Стилизация и юмор:** LLM применяется для придания тексту характерного стиля выбранной эпохи и добавления ироничных или шутливых комментариев.
*   **Структурированный вывод:** Результат генерируется в четко заданном формате (заголовок, текст, рубрика и т.д.) с помощью Pydantic, что упрощает отображение и дальнейшую обработку.
*   **Интерактивность:** Пользователь может легко выбрать дату и другие параметры через простой веб-интерфейс на Streamlit.

### Целевая аудитория
*   **Любители истории:** Для развлечения и получения необычного взгляда на известные события.
*   **Студенты и школьники:** Как интерактивный инструмент для повышения интереса к изучению истории.
*   **Преподаватели:** Для демонстрации исторических событий в неформальной и запоминающейся манере.
*   **Создатели контента:** Как источник вдохновения или готовых идей для постов, статей, видео на историческую тематику.
*   **Все любознательные:** Люди, интересующиеся возможностями LLM и ищущие необычные веб-приложения.

### Маркетинговые перспективы (Возможности)
Проект может быть интересен образовательным платформам, историческим пабликам, музеям (как интерактивный экспонат), а также как независимый развлекательный веб-сервис. Потенциально может монетизироваться через подписку на расширенные функции или генерацию больших объемов контента.

## 2. Особенности реализации

### Основные фишки
*   **Генерация псевдо-новостей:** Создание уникального контента на стыке истории и креатива LLM.
*   **Выбор даты:** Удобный календарь для выбора интересующей даты.
*   **Выбор стиля эпохи:** Возможность указать век, под стиль которого будут генерироваться новости (влияет на промпт LLM).
*   **Настройка количества новостей:** Пользователь может выбрать желаемое число заметок в сводке.
*   **Фильтрация событий по дате:** Реализован механизм фильтрации найденных RAG событий, чтобы в контекст LLM попадали только те, что близки к выбранной дате.
*   **Структурированный вывод JSON:** Использование Pydantic и возможностей LLM для получения ответа в предсказуемом формате.
*   **Пред-созданная векторная база:** Для ускорения работы и снижения нагрузки на приложение, векторная база данных (FAISS) создается заранее (офлайн) и загружается при старте.

### Текущие ограничения
*   **Хронологический охват:** На данный момент база данных событий (`historical_events.csv`) и, соответственно, генерируемые новости, **сконцентрированы преимущественно на первой трети XIX века** (1800 - 1830 гг.). Для других периодов релевантных событий может не найтись.
*   **Качество стилизации:** Степень соответствия стиля генерируемого текста реальным газетам эпохи зависит от возможностей LLM и может варьироваться.
*   **Точность дат:** Хотя реализована фильтрация, иногда в сводку могут попадать события, немного отстоящие от выбранной даты, если точных совпадений мало.
*   **Повторяемость:** При многократной генерации на одну и ту же дату могут встречаться похожие формулировки.

### Дальнейшие шаги для улучшения (очень гипотетические)
*   **Расширение базы данных:** Добавление событий из других исторических периодов и регионов.
*   **Улучшение стилизации:** Более детальная проработка промптов для лучшего соответствия стилю конкретных газет или эпох.
*   **Фильтры по региону/теме:** Добавление возможности фильтровать события не только по дате, но и по географии или тематике (политика, культура, наука).
*   **Улучшение алгоритма выбора новостей:** Внедрение более сложной логики отбора событий из найденных RAG (например, по важности, разнообразию).
*   **Сохранение/экспорт "выпусков":** Возможность сохранить понравившуюся сводку новостей.
*   **Визуальное оформление:** Улучшение дизайна вывода новостей для большего сходства с газетой.

## 3. Необходимые библиотеки и ресурсы

### Библиотеки Python
Основные зависимости перечислены в файле `requirements.txt`:
*   `streamlit`: Фреймворк для создания веб-интерфейса.
*   `langchain`, `langchain-openai`, `langchain-community`: Основной фреймворк для работы с LLM, RAG, промптами, парсерами.
*   `openai`: Клиент для взаимодействия с API OpenAI-совместимых моделей.
*   `faiss-cpu`: Библиотека для создания и работы с векторным хранилищем FAISS (загрузка пред-созданного индекса).
*   `pandas`: Для чтения и обработки исходного CSV-файла (в скрипте генерации индекса и потенциально для анализа).
*   `pydantic`: Для определения структуры данных (`NewsArticle`, `NewsReport`) и валидации вывода LLM.
*   `python-dotenv`: Для загрузки переменных окружения (API ключей) при локальной разработке.
*   `sentence-transformers`: (Используется только в офлайн-скрипте для генерации эмбеддингов, не требуется для работы самого Streamlit-приложения).

### Внешние ресурсы
*   **LLM Модель:** GPT-4o (или аналогичная), доступная через API (в данном случае, используется эндпоинт `https://forgetapi.ru/v1`, но может быть заменен на OpenAI API или другой). API-ключ и URL задаются через секреты Streamlit или переменные окружения.
*   **Embedding Модель:** `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` (используется для создания векторной базы FAISS в офлайн-скрипте).
*   **База данных событий:** Файл `data/historical_events.csv`, содержащий даты, описания, и (опционально) места и категории исторических событий.
*   **Векторный Индекс:** Предварительно сгенерированные файлы `faiss_index_historical/index.faiss` и `faiss_index_historical/index.pkl`.

## 4. Структура проекта

```
исторический-вестник/
├── .env                   # Секреты для локального запуска (не в Git!)
├── data/
│   └── historical_events.csv # Исходные данные
├── faiss_index_historical/  # Папка с готовым индексом FAISS
│   ├── index.faiss        # Файл индекса
│   └── index.pkl          # Файл метаданных индекса
├── modules/               # Модули Python с основной логикой
│   ├── __init__.py        # Маркер пакета Python (может быть пустым)
│   ├── generator.py       # Логика генерации: вызов RAG, LLM, парсинг
│   ├── models.py          # Pydantic модели для структуры данных
│   └── rag.py             # Логика RAG: загрузка индекса, создание ретривера
├── app.py                 # Главный файл Streamlit приложения (UI)
├── requirements.txt       # Зависимости Python
├── create_vector_db.ipynb # (Опционально) Jupyter/Colab ноутбук для генерации индекса
└── README.md              # Этот файл
```

## 5. Технические особенности

### 5.1. Сбор и работа с данными
*   **Источник:** Основным источником данных служит файл `data/historical_events.csv`. Он был скомпилирован с использованием открытых источников ([проект Хронос](https://hrono.ru/)) с акцентом на первую треть XIX века.
*   **Формат:** CSV файл содержит колонки: `date` (дата события в формате ГГГГ-ММ-ДД или ГГГГ), `event_description` (текстовое описание), `location` (место), `category` (категория).
*   **Офлайн-обработка (Создание Индекса):** Для подготовки данных к RAG используется отдельный скрипт (например, `create_vector_db.ipynb`):
    1.  Скрипт читает `historical_events.csv` с помощью Pandas.
    2.  Загружается модель эмбеддингов (`sentence-transformers/paraphrase-multilingual-mpnet-base-v2`).
    3.  Для каждого события из `event_description` (и опционально других полей) генерируется векторный эмбеддинг.
    4.  Создается индекс FAISS на основе этих эмбеддингов и связанных метаданных (дата, место и т.д.).
    5.  Готовый индекс сохраняется в виде двух файлов (`index.faiss`, `index.pkl`) в папку `faiss_index_historical`.

### 5.2. RAG (Retrieval-Augmented Generation)
*   **Цель:** Найти релевантные исторические события, близкие к выбранной пользователем дате, чтобы передать их LLM в качестве контекста.
*   **Реализация (во время работы приложения):**
    1.  **Загрузка Индекса:** При первом обращении (благодаря `@st.cache_resource`) функция `get_vector_store` из `rag.py` загружает пред-созданный индекс FAISS из файлов `faiss_index_historical/index.faiss` и `faiss_index_historical/index.pkl`. Для загрузки также требуется инициализировать модель эмбеддингов (`get_embeddings_loader`), но она используется только для интерпретации структуры индекса, а не для генерации новых эмбеддингов.
    2.  **Создание Ретривера:** Функция `get_retriever` создает объект-ретривер LangChain на основе загруженного индекса FAISS.
    3.  **Семантический Поиск:** Ретривер используется для поиска событий (`retriever.invoke(query)`). В качестве запроса (`query`) используется текстовое представление даты, например, "События около 7 Сентября 1812". Ретривер находит `k` документов, чьи эмбеддинги наиболее близки к эмбеддингу запроса (семантически похожи).
    4.  **Фильтрация по Дате:** Полученный от ретривера список документов дополнительно фильтруется в функции `generate_news`. Даты из метаданных документов (`doc.metadata['date']`) сравниваются с выбранной пользователем датой, и остаются только те документы, которые попадают в заданное окно (например, +/- 7 дней).
    5.  **Формирование Контекста:** Текстовые описания (`doc.page_content`) отфильтрованных по дате документов объединяются и передаются в промпт LLM.

### 5.3. Модели
*   **Языковая модель (LLM):** Используется `gpt-4o` (через совместимый API), отвечающая за генерацию текста новостей, стилизацию и форматирование вывода в JSON.
*   **Модель Эмбеддингов:** `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` (используется только офлайн для создания векторной базы).

### 5.4. Техники промптинга и структурированный вывод
*   **Промптинг:** Используется `ChatPromptTemplate` из LangChain.
    *   **Системный промпт:** Задает роль LLM ("редактор газеты 'Хронографъ'"), основную задачу, требование использовать RAG-контекст, добавлять юмор/стиль, и **строго следовать формату JSON**. Инструкции по формату JSON (`format_instructions`) передаются динамически из Pydantic парсера.
    *   **Пользовательский промпт:** Содержит конкретные параметры запроса (дата, стиль эпохи, желаемое кол-во статей).
*   **Структурированный вывод:**
    *   **Pydantic Модели:** В `models.py` определены модели `NewsArticle` (для одной новости с полями `headline`, `date_location`, `body`, `rubric`, `reporter`) и `NewsReport` (содержащая список `articles`).
    *   **PydanticOutputParser:** Парсер LangChain используется в цепочке после LLM (`prompt | llm | pydantic_parser`). Он берет JSON-ответ от LLM и автоматически валидирует его по схеме `NewsReport`, преобразуя в Python-объект. Это обеспечивает надежность и предсказуемость структуры данных.
    *   **Обработка ошибок парсинга:** В коде предусмотрены попытки повторной генерации (`MAX_RETRIES`) и ручного извлечения JSON из ответа LLM в случае сбоя автоматического парсинга.

### 5.5. Хранилище (Векторная база)
*   **Тип:** FAISS (Facebook AI Similarity Search).
*   **Назначение:** Эффективное хранение векторных эмбеддингов исторических событий и быстрый поиск ближайших соседей (семантически похожих событий).
*   **Использование:** Индекс **создается офлайн** и **загружается при старте** приложения из локальных файлов (`index.faiss`, `index.pkl`). Это решение выбрано для ускорения запуска Streamlit-приложения и снижения потребления ресурсов во время работы, так как генерация эмбеддингов является ресурсоемкой операцией. Загруженный индекс кэшируется в памяти с помощью `@st.cache_resource`.

### 5.6. Логика работы приложения (По шагам)
1.  **Запуск:** Пользователь открывает URL приложения. `app.py` выполняется, рисует UI. Кэшированные ресурсы (`get_vector_store`, `get_embeddings_loader`) пока не загружаются.
2.  **Ввод пользователя:** Пользователь выбирает дату, век, кол-во статей, окно дат.
3.  **Нажатие кнопки:** Пользователь нажимает "Сгенерировать".
4.  **Вызов `generate_news`:** `app.py` вызывает функцию `generate_news` из `generator.py`.
5.  **Загрузка/Получение Ретривера:** `generate_news` вызывает `get_retriever`. Тот вызывает `get_vector_store`.
    *   *При первом вызове:* `get_vector_store` видит, что индекса нет в кэше. Вызывает `get_embeddings_loader` (тот загружает модель эмбеддингов и кэширует ее), затем загружает файлы `index.faiss` и `index.pkl`, создает объект `FAISS` и кэширует его. Streamlit показывает "Running: get_vector_store()...".
    *   *При последующих вызовах:* `get_vector_store` мгновенно возвращает объект `FAISS` из кэша `@st.cache_resource`.
6.  **RAG Поиск:** `generate_news` использует ретривер для семантического поиска событий по текстовому запросу с датой. Получает список `all_docs`.
7.  **Фильтрация по дате:** `generate_news` фильтрует `all_docs`, оставляя только те, чья дата в метаданных попадает в заданное окно (`date_window_days`) от даты пользователя. Получает `filtered_docs`.
8.  **Подготовка контекста:** Выбирается до `num_articles` документов из `filtered_docs`, их тексты объединяются в `context`.
9.  **Создание Цепочки LLM:** `generate_news` вызывает `create_generation_chain`, которая инициализирует LLM, Pydantic парсер и создает цепочку `prompt | llm | pydantic_parser`.
10. **Вызов LLM:** Цепочка выполняется (`chain.invoke`) с подготовленным контекстом, датой, стилем и инструкциями по форматированию.
11. **Парсинг и Валидация:** `PydanticOutputParser` обрабатывает ответ LLM, проверяет соответствие JSON схеме `NewsReport` и возвращает Pydantic объект `result`. Предусмотрены повторные попытки и ручной парсинг при ошибках.
12. **Возврат результата:** `generate_news` возвращает объект `NewsReport` (или пустой) в `app.py`.
13. **Отображение:** `app.py` получает результат и отображает сгенерированные статьи или сообщение об ошибке/отсутствии данных.

## 6. Деплой

### Платформа
Приложение развернуто с использованием **Streamlit Community Cloud**.

### Причина выбора
*   **Бесплатно:** Платформа предоставляет щедрый бесплатный тир для публичных репозиториев GitHub.
*   **Простота:** Идеально подходит для развертывания приложений, написанных на Streamlit. Процесс деплоя сводится к подключению GitHub репозитория и указанию главного файла.
*   **Интеграция с GitHub:** Автоматическое обновление приложения при push'ах в связанную ветку репозитория.
*   **Управление секретами:** Удобный интерфейс для безопасного хранения API ключей (`st.secrets`).

### Взаимодействие с пользователем
Взаимодействие происходит через веб-интерфейс:
*   Пользователь выбирает дату с помощью виджета `st.date_input`.
*   Выбирает стиль эпохи (век) из выпадающего списка `st.selectbox`.
*   Задает желаемое количество новостей и окно дат с помощью слайдеров `st.slider`.
*   Нажимает кнопку `st.button` для запуска генерации.
*   Результат (сгенерированные новости или сообщения об ошибках/статусе) отображается в основной части страницы.

### Ссылки
*   **GitHub репозиторий:** [https://github.com/alexeyvkuznetsov/historical-news-generator](https://github.com/alexeyvkuznetsov/historical-news-generator)
*   **Работающее приложение:** [https://historical-news-generator.streamlit.app/](https://historical-news-generator.streamlit.app/)