# modules/generator.py
import streamlit as st
import os
import time
import json
import re
from dotenv import load_dotenv

# Импорты LangChain
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
# Импортируем САМ МОДУЛЬ exceptions из langchain_core
from langchain_core import exceptions
from langchain_core.runnables import RunnablePassthrough
# Импортируем Pydantic ValidationError отдельно для ручного парсинга
from pydantic import ValidationError


# Импорты вашего проекта
from .models import NewsReport
from .rag import get_retriever

load_dotenv()

# --- Конфигурация API ---
# Используем ваш FORGETAPI_KEY и BASE_URL из секретов Streamlit или .env
API_KEY = st.secrets.get("FORGETAPI_KEY", os.getenv("FORGETAPI_KEY"))
BASE_URL = st.secrets.get("FORGETAPI_BASE_URL", os.getenv("FORGETAPI_BASE_URL", "https://forgetapi.ru/v1")) # Укажите URL по умолчанию, если нужно
MAX_RETRIES = 2
# Глобальная переменная для хранения последней ошибки (для app.py)
last_error = None

# --- Функция get_llm ---
def get_llm():
    """Инициализирует LLM с заданными параметрами."""
    if not API_KEY:
        # Используем st.error для отображения в UI Streamlit, если ключ не найден при запуске
        st.error("Критическая ошибка: Не найден API ключ для 'forgetapi'. Добавьте FORGETAPI_KEY в секреты Streamlit или в .env файл.")
        # Выбрасываем ValueError, чтобы остановить выполнение, если ключ абсолютно необходим
        raise ValueError("FORGETAPI_KEY не найден.")
    if not BASE_URL:
        # Аналогично для BASE_URL
        st.error("Критическая ошибка: Не найден BASE_URL для 'forgetapi'. Добавьте FORGETAPI_BASE_URL в секреты Streamlit или в .env файл.")
        raise ValueError("FORGETAPI_BASE_URL не найден.")

    llm = ChatOpenAI(
        model="gpt-3.5-turbo", # Или другая модель, доступная через ваш BASE_URL
        openai_api_key=API_KEY,
        openai_api_base=BASE_URL,
        temperature=0.7,
        request_timeout=120 # Увеличьте, если запросы часто прерываются по таймауту
    )
    return llm

# --- Функция create_generation_chain ---
def create_generation_chain():
    """Создает LangChain цепочку для генерации структурированных новостей."""
    llm = get_llm() # Получаем настроенный LLM
    # Создаем парсер на основе Pydantic модели NewsReport
    pydantic_parser = PydanticOutputParser(pydantic_object=NewsReport)
    # Получаем инструкции по форматированию для LLM
    format_instructions = pydantic_parser.get_format_instructions()

    # Определяем шаблон промпта с инструкциями и плейсхолдерами
    prompt = ChatPromptTemplate.from_messages([
         ("system", """Ты — остроумный и немного саркастичный редактор исторической газеты 'Хронографъ'.
Твоя задача — написать сводку новостей для выпуска газеты на заданную дату.
Используй следующие реальные исторические события как основу, но добавь детали, юмор, вымышленных персонажей или комментарии в стиле газеты {era_style} века.

ВАЖНО: Весь твой ответ ДОЛЖЕН быть ТОЛЬКО JSON объектом, без какого-либо другого текста до или после него.
JSON должен строго соответствовать следующей структуре (не включай ```json или ``` в свой ответ):
{format_instructions}

Реальные события (контекст):
{context}"""),
        # Пользовательский запрос с параметрами
        ("user", "Пожалуйста, напиши новости для даты {date_input}. Используй примерно {num_articles} события из контекста. Стиль: {era_style} век.")
    ])

    # Создаем цепочку: Промпт -> LLM -> Парсер Pydantic
    # Парсер автоматически попытается разобрать вывод LLM в объект NewsReport
    chain = prompt | llm | pydantic_parser
    # Опционально: можно добавить логгирование сырого вывода LLM перед парсером для отладки
    # chain = prompt | llm | RunnablePassthrough(lambda x: print(f"--- LLM Raw Output ---\n{x.content}\n---")) | pydantic_parser
    return chain

# --- Функция generate_news ---
def generate_news(target_date: str, era_style: str = "XVIII", num_articles: int = 3) -> NewsReport:
    """Основная функция для генерации новостей с обработкой ошибок и повторными попытками."""
    global last_error # Объявляем, что будем менять глобальную переменную
    last_error = None # Сбрасываем ошибку перед каждым новым запуском
    print(f"Запрос на генерацию новостей для даты: {target_date}, стиль: {era_style}")

    # 1. Получаем контекст из RAG
    try:
        retriever = get_retriever(k=num_articles + 2) # Запросим чуть больше контекста
        query = f"События около {target_date}"
        relevant_docs = retriever.invoke(query)
        if not relevant_docs:
            st.warning(f"Не найдено релевантных исторических событий для даты '{target_date}'. Генерация невозможна.")
            return NewsReport(articles=[]) # Возвращаем пустой отчет
        context = "\n\n".join([doc.page_content for doc in relevant_docs])
        print(f"Найденный контекст (первые 500 символов):\n{context[:500]}...")
    except Exception as e:
        st.error(f"Ошибка при поиске событий в RAG: {e}")
        print(f"Полная ошибка RAG: {e}") # Лог для детальной отладки
        last_error = e
        return NewsReport(articles=[]) # Возвращаем пустой отчет при ошибке RAG

    # 2. Генерируем новости с помощью LLM и парсера
    try:
        chain = create_generation_chain()
        # Получаем инструкции по форматированию еще раз (хотя они уже внутри chain)
        pydantic_parser = PydanticOutputParser(pydantic_object=NewsReport)
        format_instructions = pydantic_parser.get_format_instructions()
    except ValueError as ve: # Ловим ошибку инициализации LLM (например, нет ключа)
        st.error(f"Ошибка конфигурации LLM: {ve}")
        last_error = ve
        return NewsReport(articles=[])

    for attempt in range(MAX_RETRIES):
        print(f"Попытка генерации {attempt + 1}/{MAX_RETRIES}...")
        try:
            # Формируем входные данные для цепочки
            chain_input = {
                "date_input": target_date,
                "era_style": era_style,
                "num_articles": num_articles,
                "context": context,
                "format_instructions": format_instructions # Передаем инструкции в контекст промпта
            }
            # Запускаем цепочку
            result = chain.invoke(chain_input)

            # Проверяем, что результат имеет ожидаемый тип (NewsReport)
            if isinstance(result, NewsReport):
                print("Генерация и парсинг прошли успешно.")
                last_error = None # Сбрасываем ошибку при успехе
                return result # Возвращаем успешный результат
            else:
                # Если парсер вернул что-то другое (например, строку при ошибке)
                print(f"Неожиданный тип результата от парсера: {type(result)}. Результат: {result}")
                last_error = exceptions.OutputParsingError(f"Неожиданный тип результата от парсера: {type(result)}")
                # Попытка ручного извлечения JSON из строки (если result это строка)
                if isinstance(result, str):
                    try:
                         # Ищем JSON объект в строке
                         json_match = re.search(r'\{.*\}', result, re.DOTALL)
                         if json_match:
                            json_str = json_match.group(0)
                            parsed_json = json.loads(json_str)
                            # Валидируем вручную распарсенный JSON через Pydantic
                            news_report = NewsReport.model_validate(parsed_json) # Используем model_validate для Pydantic v2+
                            print("Удалось вручную распарсить и валидировать JSON из строки.")
                            last_error = None
                            return news_report
                         else:
                            print("Не удалось найти JSON в строке ответа.")
                            last_error = exceptions.OutputParsingError("Не удалось найти JSON в строке ответа.")
                    except (json.JSONDecodeError, ValidationError) as manual_parse_error:
                        print(f"Ошибка ручного парсинга/валидации JSON: {manual_parse_error}")
                        last_error = exceptions.OutputParsingError(f"Ошибка ручного парсинга/валидации JSON: {manual_parse_error}")
                # Если ручной парсинг не удался или тип был не строка, переходим к следующей попытке

        # Ловим специфичную ошибку парсинга от LangChain
        except exceptions.OutputParsingError as ope:
            print(f"Ошибка парсинга Pydantic на попытке {attempt + 1}: {ope}")
            # Пытаемся получить сырой вывод LLM из атрибутов ошибки, если он там есть
            raw_output = getattr(ope, 'llm_output', str(ope))
            # Выводим предупреждение в UI
            st.warning(f"Попытка {attempt + 1}: Не удалось разобрать ответ LLM. Пробуем снова...")
            print(f"--- Сырой вывод LLM (при ошибке парсинга) ---\n{raw_output}\n---")
            last_error = ope # Сохраняем ошибку
            # Ждем немного перед следующей попыткой
            time.sleep(2)

        # Ловим другие возможные ошибки (сетевые, API и т.д.)
        except Exception as e:
            print(f"Неожиданная ошибка на попытке {attempt + 1}: {e}")
            st.error(f"Произошла неожиданная ошибка при генерации: {e}")
            last_error = e # Сохраняем ошибку
            break # Прерываем цикл попыток при других ошибках

    # Если все попытки не удались
    st.error(f"Не удалось сгенерировать новости после {MAX_RETRIES} попыток.")
    if last_error:
        # Показываем последнюю ошибку в UI
        st.error(f"Детали последней ошибки: {last_error}")
        # Если в ошибке был сырой вывод, покажем его для отладки
        raw_output = getattr(last_error, 'llm_output', None)
        if raw_output:
            st.text_area("Последний сырой ответ от LLM (для отладки):", str(raw_output), height=200)

    # Возвращаем пустой отчет, если ничего не получилось
    return NewsReport(articles=[])

# --- Блок для локального тестирования ---
if __name__ == '__main__':
    print("Запуск локального теста генератора...")
    test_date = "14 июля 1789" # Пример даты
    try:
        # Убедитесь, что переменные окружения установлены для локального теста
        # Например, через os.environ или .env файл, который читается load_dotenv()
        # os.environ.setdefault("FORGETAPI_KEY", "ВАШ_КЛЮЧ_ЗДЕСЬ")
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
         print(f"Ошибка конфигурации при тесте: {ve}")
    except ImportError as ie:
         print(f"Ошибка импорта при тесте: {ie}")
    except Exception as e:
         print(f"Непредвиденная ошибка при тесте: {e}")