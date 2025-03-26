# Файл: modules/generator.py

import os
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers.openai_functions import PydanticOutputFunctionsParser
from langchain_core.utils.function_calling import convert_to_openai_function

# Убедитесь, что пути импорта верны относительно вашей структуры папок
from .models import NewsReport # Импортируем нашу Pydantic модель
from .rag import get_retriever   # Импортируем функцию получения ретривера

# --- Конфигурация ---

# Имя секрета/переменной окружения для вашего API ключа
CUSTOM_API_KEY_NAME = "CUSTOM_LLM_API_KEY"
# Базовый URL вашего OpenAI-совместимого провайдера
CUSTOM_BASE_URL = "https://api.sree.shop/v1"
# Конкретное имя модели, которое вы хотите использовать и которое доступно у провайдера
MODEL_NAME = "gpt-4o" # Используем указанную вами модель



# --- Функции ---

def get_llm():
    """
    Инициализирует и возвращает экземпляр ChatOpenAI,
    настроенный для работы с кастомным OpenAI-совместимым API.
    """
    # Получаем ключ из секретов Streamlit или из .env для локального запуска
    api_key = st.secrets.get(CUSTOM_API_KEY_NAME, os.getenv(CUSTOM_API_KEY_NAME))
    if not api_key:
        raise ValueError(
            f"Не найден API ключ. Добавьте {CUSTOM_API_KEY_NAME} "
            "в секреты Streamlit или в .env файл."
        )

    print(f"Инициализация LLM: модель='{MODEL_NAME}', endpoint='{CUSTOM_BASE_URL}'")

    try:
        llm = ChatOpenAI(
            model=MODEL_NAME,           # Используем указанную модель
            openai_api_key=api_key,     # Используем ключ вашего провайдера
            base_url=CUSTOM_BASE_URL,   # Передаем базовый URL вашего провайдера
            temperature=0.7,            # Небольшая креативность
            # request_timeout=60,       # Опционально: установите таймаут, если нужно
            # max_tokens=1024           # Опционально: ограничение на длину ответа
        )
        return llm
    except Exception as e:
        st.error(f"Ошибка при инициализации LLM: {e}")
        print(f"Ошибка при инициализации LLM: {e}")
        raise # Перевыбрасываем ошибку, чтобы остановить выполнение, если LLM не создался

def create_generation_chain():
    """
    Создает LangChain цепочку для генерации структурированных новостей.
    Цепочка включает: получение LLM, форматирование промпта, вызов LLM с Pydantic-моделью
    в качестве функции и парсинг результата в Pydantic-объект.
    """
    llm = get_llm() # Получаем настроенный LLM

    # Преобразуем Pydantic модель в формат, понятный OpenAI для function calling
    openai_functions = [convert_to_openai_function(NewsReport)]

    # Создаем парсер, который будет извлекать данные из ответа LLM
    # согласно нашей Pydantic модели NewsReport
    parser = PydanticOutputFunctionsParser(pydantic_schema=NewsReport)

    # Определяем шаблон промпта
    prompt = ChatPromptTemplate.from_messages([
        ("system", """Ты — остроумный и немного саркастичный редактор исторической газеты 'Хронографъ'.
Твоя задача — написать сводку новостей для выпуска газеты на заданную дату.
Используй следующие реальные исторические события как основу, но добавь детали, юмор, вымышленных персонажей или комментарии в стиле газеты {era_style} века.
Важно: Строго придерживайся формата JSON, который соответствует Pydantic-модели NewsReport, содержащей список статей NewsArticle.
Не добавляй никакого текста до или после JSON-объекта.

Реальные события (контекст):
{context}"""),
        ("user", "Пожалуйста, напиши новости для даты {date_input}. Используй примерно {num_articles} события из предоставленного контекста. Стиль: {era_style} век.")
    ])

    # Привязываем описание нашей Pydantic модели (в формате OpenAI function) к LLM,
    # указывая, что мы ожидаем вызов именно этой функции.
    llm_with_functions = llm.bind(
        functions=openai_functions,
        function_call={"name": "NewsReport"} # Указываем имя функции (совпадает с именем Pydantic модели)
        )

    # Создаем цепочку: Промпт -> LLM с функцией -> Парсер Pydantic
    chain = prompt | llm_with_functions | parser
    return chain

def generate_news(target_date: str, era_style: str = "XVIII", num_articles: int = 3) -> NewsReport | None:
    """
    Основная функция для генерации новостей.
    Получает релевантные события через RAG, формирует промпт,
    вызывает цепочку генерации и возвращает результат в виде Pydantic-объекта NewsReport.
    Возвращает None в случае ошибки или отсутствия данных.
    """
    print(f"Запрос на генерацию новостей для даты: {target_date}")

    try:
        retriever = get_retriever(k=num_articles + 2) # Запросим чуть больше контекста
    except Exception as e:
        st.error(f"Ошибка при инициализации RAG ретривера: {e}")
        print(f"Ошибка при инициализации RAG ретривера: {e}")
        return None # Не можем продолжить без ретривера

    # Ищем релевантные события по семантической близости к дате/описанию
    query = f"Исторические события около {target_date}"
    try:
        relevant_docs = retriever.invoke(query)
    except Exception as e:
        st.error(f"Ошибка при поиске релевантных документов в RAG: {e}")
        print(f"Ошибка при поиске релевантных документов в RAG: {e}")
        return None

    if not relevant_docs:
        st.warning(f"Не найдено релевантных исторических событий в базе для даты около {target_date}.")
        print(f"Не найдено релевантных событий для: {target_date}")
        # Возвращаем пустой отчет, чтобы UI мог это обработать
        return NewsReport(articles=[])

    context = "\n\n".join([doc.page_content for doc in relevant_docs])
    print(f"Найденный контекст для LLM:\n{context}")

    try:
        chain = create_generation_chain() # Создаем цепочку генерации
    except ValueError as ve: # Обработка ошибки из get_llm (например, нет ключа)
        st.error(f"Ошибка конфигурации LLM: {ve}")
        print(f"Ошибка конфигурации LLM: {ve}")
        return None
    except Exception as e: # Другие возможные ошибки при создании цепочки
        st.error(f"Ошибка при создании цепочки генерации: {e}")
        print(f"Ошибка при создании цепочки генерации: {e}")
        return None

    print(f"Вызов LLM ('{MODEL_NAME}') для генерации...")
    try:
        # Передаем все необходимые переменные в цепочку
        result: NewsReport = chain.invoke({
            "date_input": target_date,
            "era_style": era_style,
            "num_articles": num_articles,
            "context": context
        })
        print("Генерация LLM завершена успешно.")
        return result
    except Exception as e:
        # Обработка ошибок во время вызова LLM или парсинга ответа
        st.error(f"Произошла ошибка во время генерации или парсинга ответа LLM: {e}")
        print(f"Ошибка при вызове LLM ('{MODEL_NAME}') или парсинге: {e}")
        # Дополнительно можно вывести traceback для отладки, если нужно
        # import traceback
        # print(traceback.format_exc())
        return None # Возвращаем None при ошибке генерации

# --- Блок для локального тестирования ---
if __name__ == '__main__':
    # Убедитесь, что у вас есть .env файл с CUSTOM_LLM_API_KEY для локального запуска
    # И файл data/historical_events.csv
    from dotenv import load_dotenv
    load_dotenv()

    print("--- Локальное тестирование generator.py ---")
    test_date = "14 июля 1789"
    test_era = "XVIII"
    test_num = 2

    # Проверяем наличие данных RAG
    try:
        retriever_test = get_retriever()
        print("RAG ретривер инициализирован.")
        docs = retriever_test.invoke(f"События около {test_date}")
        if not docs:
            print(f"ПРЕДУПРЕЖДЕНИЕ: Не найдено документов RAG для '{test_date}'. Генерация может быть некачественной.")
        else:
             print(f"Найдено {len(docs)} документов RAG для '{test_date}'.")
    except Exception as e:
        print(f"ОШИБКА при инициализации RAG для теста: {e}. Убедитесь, что файл data/historical_events.csv существует.")
        exit() # Выход, если RAG не работает

    # Запускаем генерацию
    report = generate_news(test_date, era_style=test_era, num_articles=test_num)

    if report and report.articles:
        print(f"\n--- Сгенерированный отчет для {test_date} ({test_era} век) ---")
        for i, article in enumerate(report.articles):
            print(f"\n--- Статья {i+1} ---")
            print(f"Заголовок: {article.headline}")
            print(f"Рубрика: {article.rubric}")
            print(f"Дата/Место: {article.date_location}")
            print(f"Репортер: {article.reporter}")
            print(f"Текст:\n{article.body}")
        print("\n--- Конец отчета ---")
    elif report and not report.articles:
         print("\n--- Генерация завершилась, но не было найдено подходящих событий или LLM не смог создать новости. ---")
    else:
        print("\n--- Не удалось сгенерировать отчет из-за ошибки. ---")

    print("--- Тестирование завершено ---")
