# modules/generator.py
import streamlit as st
import os
import time
import json
import re # Импортируем re для ручного поиска JSON
from dotenv import load_dotenv

# Импорты LangChain
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
# Используем только PydanticOutputParser
from langchain_core.output_parsers import PydanticOutputParser
# Импортируем OutputParsingError из основного пакета exceptions
from langchain.exceptions import OutputParsingError
from langchain_core.runnables import RunnablePassthrough # Для отладки

# Импорты вашего проекта
from .models import NewsReport
from .rag import get_retriever

load_dotenv()

# --- Конфигурация API ---
API_KEY = st.secrets.get("FORGETAPI_KEY", os.getenv("FORGETAPI_KEY"))
BASE_URL = st.secrets.get("FORGETAPI_BASE_URL", os.getenv("FORGETAPI_BASE_URL", "https://forgetapi.ru/v1"))

# Максимальное количество попыток генерации
MAX_RETRIES = 2

# Глобальная переменная для отслеживания последней ошибки (для app.py)
last_error = None

def get_llm():
    """Инициализирует LLM с вашими параметрами."""
    if not API_KEY:
        raise ValueError("Не найден API ключ для 'forgetapi'. Добавьте FORGETAPI_KEY в секреты Streamlit или .env.")
    if not BASE_URL:
        raise ValueError("Не найден BASE_URL для 'forgetapi'. Добавьте FORGETAPI_BASE_URL в секреты Streamlit или .env.")

    llm = ChatOpenAI(
        model="gpt-3.5-turbo", # Укажите модель, которую поддерживает ваш провайдер
        openai_api_key=API_KEY,
        openai_api_base=BASE_URL, # Указываем ваш base_url
        temperature=0.7,
        request_timeout=120 # Увеличим таймаут
    )
    return llm

def create_generation_chain():
    """
    Создает LangChain цепочку с PydanticOutputParser.
    """
    llm = get_llm()
    pydantic_parser = PydanticOutputParser(pydantic_object=NewsReport)
    format_instructions = pydantic_parser.get_format_instructions()

    prompt = ChatPromptTemplate.from_messages([
        ("system", """Ты — остроумный и немного саркастичный редактор исторической газеты 'Хронографъ'.
Твоя задача — написать сводку новостей для выпуска газеты на заданную дату.
Используй следующие реальные исторические события как основу, но добавь детали, юмор, вымышленных персонажей или комментарии в стиле газеты {era_style} века.

ВАЖНО: Весь твой ответ ДОЛЖЕН быть ТОЛЬКО JSON объектом, без какого-либо другого текста до или после него.
JSON должен строго соответствовать следующей структуре (не включай ```json или ```):
{format_instructions}

Реальные события (контекст):
{context}"""),
        ("user", "Пожалуйста, напиши новости для даты {date_input}. Используй примерно {num_articles} события из контекста. Стиль: {era_style} век.")
    ])

    # Используем только PydanticOutputParser
    chain = prompt | llm | pydantic_parser

    # Раскомментируйте для отладки сырого вывода LLM ПЕРЕД парсером
    # chain = prompt | llm | RunnablePassthrough(lambda x: print(f"--- LLM Raw Output ---\n{x.content}\n---")) | pydantic_parser

    return chain

def generate_news(target_date: str, era_style: str = "XVIII", num_articles: int = 3) -> NewsReport:
    """Основная функция для генерации новостей с обработкой ошибок и повторами."""
    global last_error # Используем глобальную переменную
    last_error = None # Сбрасываем ошибку при каждом вызове
    print(f"Запрос на генерацию новостей для даты: {target_date}, стиль: {era_style}")

    # 1. Получаем контекст из RAG
    try:
        retriever = get_retriever(k=num_articles + 2) # Запросим чуть больше контекста
        query = f"События около {target_date}"
        relevant_docs = retriever.invoke(query)

        if not relevant_docs:
            st.warning(f"Не найдено релевантных исторических событий для даты '{target_date}'.")
            return NewsReport(articles=[]) # Возвращаем пустой список

        context = "\n\n".join([doc.page_content for doc in relevant_docs])
        print(f"Найденный контекст:\n{context[:500]}...") # Печатаем начало контекста

    except Exception as e:
        st.error(f"Ошибка при поиске событий в RAG: {e}")
        print(f"Полная ошибка RAG: {e}")
        last_error = e # Сохраняем ошибку
        return NewsReport(articles=[])

    # 2. Генерируем новости с помощью LLM и парсим
    chain = create_generation_chain()
    pydantic_parser = PydanticOutputParser(pydantic_object=NewsReport) # Нужен для инструкций
    format_instructions = pydantic_parser.get_format_instructions()

    for attempt in range(MAX_RETRIES):
        print(f"Попытка генерации {attempt + 1}/{MAX_RETRIES}...")
        try:
            # Формируем входные данные для цепочки
            chain_input = {
                "date_input": target_date,
                "era_style": era_style,
                "num_articles": num_articles,
                "context": context,
                "format_instructions": format_instructions # Передаем инструкции в промпт
            }
            # Вызываем цепочку
            result = chain.invoke(chain_input)

            # Проверяем тип результата (должен быть экземпляр NewsReport)
            if isinstance(result, NewsReport):
                print("Генерация и парсинг прошли успешно.")
                last_error = None # Сбрасываем ошибку при успехе
                return result # Возвращаем успешный результат
            else:
                # Это не должно происходить с PydanticOutputParser, но обрабатываем
                print(f"Неожиданный тип результата: {type(result)}. Результат: {result}")
                last_error = OutputParsingError(f"Неожиданный тип результата: {type(result)}")
                # Попытка ручного парсинга JSON из строки (если result - строка)
                if isinstance(result, str):
                    try:
                         # Ищем JSON внутри строки (может быть окружен текстом)
                         json_match = re.search(r'\{.*\}', result, re.DOTALL)
                         if json_match:
                            json_str = json_match.group(0)
                            parsed_json = json.loads(json_str)
                            # Валидируем и создаем Pydantic объект
                            news_report = NewsReport.parse_obj(parsed_json)
                            print("Удалось вручную распарсить и валидировать JSON из строки.")
                            last_error = None
                            return news_report
                         else:
                            print("Не удалось найти JSON в строке ответа.")
                            last_error = OutputParsingError("Не удалось найти JSON в строке ответа.")
                    except (json.JSONDecodeError, ValidationError) as manual_parse_error: # Импорт ValidationError нужен из pydantic
                        print(f"Ошибка ручного парсинга/валидации: {manual_parse_error}")
                        last_error = OutputParsingError(f"Ошибка ручного парсинга/валидации: {manual_parse_error}")

        except OutputParsingError as ope:
            # Используем OutputParsingError, импортированный из langchain.exceptions
            print(f"Ошибка парсинга на попытке {attempt + 1}: {ope}")
            raw_output = getattr(ope, 'llm_output', str(ope)) # Пытаемся получить сырой вывод из ошибки
            st.warning(f"Попытка {attempt + 1}: Не удалось разобрать ответ LLM. Пробуем снова...")
            print(f"--- Сырой вывод LLM (при ошибке парсинга) ---\n{raw_output}\n---")
            last_error = ope
            time.sleep(2) # Ждем перед повтором

        except Exception as e:
            print(f"Неожиданная ошибка на попытке {attempt + 1}: {e}")
            st.error(f"Произошла неожиданная ошибка при генерации: {e}")
            last_error = e
            # Если ошибка не связана с парсингом, повторная попытка может не помочь
            break # Прерываем цикл при других ошибках

    # Если все попытки не удались
    st.error(f"Не удалось сгенерировать новости после {MAX_RETRIES} попыток.")
    if last_error:
        st.error(f"Последняя ошибка: {last_error}")
        # Выводим сырой вывод LLM, если он был в ошибке парсинга
        raw_output = getattr(last_error, 'llm_output', None)
        if raw_output:
            st.text_area("Последний сырой ответ от LLM (для отладки):", raw_output, height=200)

    return NewsReport(articles=[]) # Возвращаем пустой результат

# --- Блок для локального тестирования ---
if __name__ == '__main__':
    from pydantic import ValidationError # Убедитесь, что этот импорт есть для блока main

    # Убедитесь, что у вас есть .env файл с FORGETAPI_KEY и FORGETAPI_BASE_URL для локального запуска
    print("Запуск локального теста генератора...")
    test_date = "14 июля 1789"
    try:
        # Установите переменные окружения для теста, если не используете .env
        # os.environ.setdefault("FORGETAPI_KEY", "ВАШ_КЛЮЧ")
        # os.environ.setdefault("FORGETAPI_BASE_URL", "https://forgetapi.ru/v1")

        report = generate_news(test_date, era_style="XVIII", num_articles=2)
        if report and report.articles:
            print(f"\n--- Сгенерированный отчет для {test_date} ---")
            for article in report.articles:
                print(f"\nЗаголовок: {article.headline}")
                print(f"Рубрика: {article.rubric}")
                print(f"Дата/Место: {article.date_location}")
                print(f"Репортер: {article.reporter}")
                print(f"Текст: {article.body}")
            print("--- Конец отчета ---")
        else:
            print("Не удалось сгенерировать новости (возможно, после всех попыток).")
    except ValueError as ve:
         print(f"Ошибка конфигурации: {ve}")
    except ImportError as ie:
         print(f"Ошибка импорта при тесте: {ie}") # Добавим отлов ImportError
    except Exception as e:
         print(f"Непредвиденная ошибка при тесте: {e}")