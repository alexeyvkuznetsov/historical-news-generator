# modules/generator.py
import streamlit as st
import os
import time
import json # Импортируем стандартный JSON парсер для диагностики
from dotenv import load_dotenv

# Импорты LangChain
from langchain_openai import ChatOpenAI # Используем ChatOpenAI для совместимости
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser, OutputFixingParser # Добавляем OutputFixingParser
from langchain_core.exceptions import OutputParsingError # Для отлова ошибок парсинга

# Импорты вашего проекта
from .models import NewsReport # Ваша Pydantic модель
from .rag import get_retriever # Ваш RAG модуль

load_dotenv()

# --- Конфигурация API ---
# Получаем данные из секретов Streamlit или переменных окружения
# Не храните ключи прямо в коде! Используйте секреты Streamlit или .env
API_KEY = st.secrets.get("FORGETAPI_KEY", os.getenv("FORGETAPI_KEY"))
BASE_URL = st.secrets.get("FORGETAPI_BASE_URL", os.getenv("FORGETAPI_BASE_URL", "https://forgetapi.ru/v1")) # Значение по умолчанию, если нет в секретах

# Максимальное количество попыток генерации
MAX_RETRIES = 2

def get_llm():
    """Инициализирует LLM с вашими параметрами."""
    if not API_KEY:
        raise ValueError("Не найден API ключ для 'forgetapi'. Добавьте FORGETAPI_KEY в секреты Streamlit или .env.")
    if not BASE_URL:
        raise ValueError("Не найден BASE_URL для 'forgetapi'. Добавьте FORGETAPI_BASE_URL в секреты Streamlit или .env.")

    llm = ChatOpenAI(
        model="gpt-3.5-turbo", # Укажите модель, которую поддерживает ваш провайдер
        openai_api_key=API_KEY,
        openai_api_base=BASE_URL, # <--- Указываем ваш base_url
        temperature=0.7,
        request_timeout=120 # Увеличим таймаут на всякий случай
        # max_tokens=... # Можно ограничить, если нужно
    )
    return llm

def create_generation_chain():
    """
    Создает LangChain цепочку для генерации новостей с парсингом из текста ответа.
    """
    llm = get_llm()

    # Создаем парсер, который ожидает JSON в строке ответа и парсит его по модели NewsReport
    pydantic_parser = PydanticOutputParser(pydantic_object=NewsReport)

    # Получаем инструкции по форматированию от парсера (часто содержит JSON Schema)
    format_instructions = pydantic_parser.get_format_instructions()

    # Определяем шаблон промпта
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

    # Создаем цепочку: Промпт -> LLM -> Парсер Pydantic
    # Добавляем OutputFixingParser: он попытается исправить JSON, если LLM немного ошиблась
    output_fixing_parser = OutputFixingParser.from_llm(parser=pydantic_parser, llm=llm)

    chain = prompt | llm | output_fixing_parser # Поменяли парсер
    # Для отладки можно посмотреть сырой вывод LLM перед парсером:
    # chain = prompt | llm | RunnablePassthrough(lambda x: print(f"LLM Raw Output:\n{x.content}\n---")) | output_fixing_parser

    return chain

def generate_news(target_date: str, era_style: str = "XVIII", num_articles: int = 3) -> NewsReport:
    """Основная функция для генерации новостей с обработкой ошибок и повторами."""
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
        return NewsReport(articles=[])

    # 2. Генерируем новости с помощью LLM и парсим
    chain = create_generation_chain()
    pydantic_parser = PydanticOutputParser(pydantic_object=NewsReport) # Нужен для инструкций
    format_instructions = pydantic_parser.get_format_instructions()

    last_error = None
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
                return result
            else:
                # Это не должно происходить с PydanticOutputParser, но на всякий случай
                print(f"Неожиданный тип результата: {type(result)}. Результат: {result}")
                last_error = OutputParsingError(f"Неожиданный тип результата: {type(result)}")
                # Попробуем вручную распарсить, если это строка
                if isinstance(result, str):
                    try:
                         # Ищем JSON внутри строки
                         json_match = re.search(r'\{.*\}', result, re.DOTALL)
                         if json_match:
                            json_str = json_match.group(0)
                            parsed_json = json.loads(json_str)
                            news_report = NewsReport.parse_obj(parsed_json)
                            print("Удалось вручную распарсить JSON из строки.")
                            return news_report
                         else:
                            print("Не удалось найти JSON в строке ответа.")
                            last_error = OutputParsingError("Не удалось найти JSON в строке ответа.")
                    except (json.JSONDecodeError, ValidationError) as manual_parse_error:
                        print(f"Ошибка ручного парсинга: {manual_parse_error}")
                        last_error = OutputParsingError(f"Ошибка ручного парсинга: {manual_parse_error}")

        except OutputParsingError as ope:
            print(f"Ошибка парсинга на попытке {attempt + 1}: {ope}")
            # Попробуем извлечь сырой вывод LLM из ошибки, если возможно
            raw_output = getattr(ope, 'llm_output', str(ope))
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
    import re
    from pydantic import ValidationError
    # Убедитесь, что у вас есть .env файл с FORGETAPI_KEY и FORGETAPI_BASE_URL для локального запуска
    print("Запуск локального теста генератора...")
    test_date = "14 июля 1789"
    try:
        # Устанавливаем переменные окружения для теста, если их нет
        os.environ.setdefault("FORGETAPI_KEY", "g1QUHzuBFbDpjSMGsab2bFnNqeBxgJG0") # Замените или удалите, если используете .env
        os.environ.setdefault("FORGETAPI_BASE_URL", "https://forgetapi.ru/v1")

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
            print("Не удалось сгенерировать новости.")
    except ValueError as ve:
         print(f"Ошибка конфигурации: {ve}")
    except Exception as e:
         print(f"Непредвиденная ошибка при тесте: {e}")