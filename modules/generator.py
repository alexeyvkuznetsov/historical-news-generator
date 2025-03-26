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
# УБИРАЕМ OutputFixingParser, ОСТАВЛЯЕМ ТОЛЬКО PydanticOutputParser
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.exceptions import OutputParsingError
from langchain_core.runnables import RunnablePassthrough # Для отладки

# Импорты вашего проекта
from .models import NewsReport
from .rag import get_retriever

load_dotenv()

API_KEY = st.secrets.get("FORGETAPI_KEY", os.getenv("FORGETAPI_KEY"))
BASE_URL = st.secrets.get("FORGETAPI_BASE_URL", os.getenv("FORGETAPI_BASE_URL", "https://forgetapi.ru/v1"))

MAX_RETRIES = 2

def get_llm():
    # ... (код get_llm остается без изменений) ...
    if not API_KEY:
        raise ValueError("Не найден API ключ для 'forgetapi'. Добавьте FORGETAPI_KEY в секреты Streamlit или .env.")
    if not BASE_URL:
        raise ValueError("Не найден BASE_URL для 'forgetapi'. Добавьте FORGETAPI_BASE_URL в секреты Streamlit или .env.")

    llm = ChatOpenAI(
        model="gpt-3.5-turbo", # Укажите вашу модель
        openai_api_key=API_KEY,
        openai_api_base=BASE_URL,
        temperature=0.7,
        request_timeout=120
    )
    return llm

def create_generation_chain():
    """
    Создает LangChain цепочку с PydanticOutputParser (без OutputFixingParser).
    """
    llm = get_llm()
    pydantic_parser = PydanticOutputParser(pydantic_object=NewsReport)
    format_instructions = pydantic_parser.get_format_instructions()

    prompt = ChatPromptTemplate.from_messages([
        # ... (Промпт остается без изменений, убедитесь, что {format_instructions} есть) ...
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
    """Основная функция генерации (логика повторов и RAG остается)."""
    global last_error # Используем глобальную переменную для ошибки (не лучший стиль, но для отладки сойдет)
    last_error = None
    print(f"Запрос на генерацию новостей для даты: {target_date}, стиль: {era_style}")

    # 1. Получаем контекст из RAG (код без изменений)
    try:
        retriever = get_retriever(k=num_articles + 2)
        query = f"События около {target_date}"
        relevant_docs = retriever.invoke(query)
        if not relevant_docs:
            st.warning(f"Не найдено релевантных исторических событий для даты '{target_date}'.")
            return NewsReport(articles=[])
        context = "\n\n".join([doc.page_content for doc in relevant_docs])
        print(f"Найденный контекст:\n{context[:500]}...")
    except Exception as e:
        st.error(f"Ошибка при поиске событий в RAG: {e}")
        print(f"Полная ошибка RAG: {e}")
        last_error = e
        return NewsReport(articles=[])

    # 2. Генерируем новости (используем chain без OutputFixingParser)
    chain = create_generation_chain()
    pydantic_parser = PydanticOutputParser(pydantic_object=NewsReport)
    format_instructions = pydantic_parser.get_format_instructions()

    for attempt in range(MAX_RETRIES):
        print(f"Попытка генерации {attempt + 1}/{MAX_RETRIES}...")
        try:
            chain_input = {
                "date_input": target_date,
                "era_style": era_style,
                "num_articles": num_articles,
                "context": context,
                "format_instructions": format_instructions
            }
            result = chain.invoke(chain_input)

            if isinstance(result, NewsReport):
                print("Генерация и парсинг прошли успешно.")
                last_error = None # Сбрасываем ошибку при успехе
                return result
            else:
                # Это не должно происходить, но обрабатываем
                print(f"Неожиданный тип результата: {type(result)}. Результат: {result}")
                last_error = OutputParsingError(f"Неожиданный тип результата: {type(result)}")
                # Попытка ручного парсинга (оставляем как было)
                if isinstance(result, str):
                    try:
                         json_match = re.search(r'\{.*\}', result, re.DOTALL)
                         if json_match:
                            json_str = json_match.group(0)
                            parsed_json = json.loads(json_str)
                            news_report = NewsReport.parse_obj(parsed_json)
                            print("Удалось вручную распарсить JSON из строки.")
                            last_error = None
                            return news_report
                         else:
                             print("Не удалось найти JSON в строке ответа.")
                             last_error = OutputParsingError("Не удалось найти JSON в строке ответа.")
                    except (json.JSONDecodeError, ValidationError) as manual_parse_error:
                        print(f"Ошибка ручного парсинга: {manual_parse_error}")
                        last_error = OutputParsingError(f"Ошибка ручного парсинга: {manual_parse_error}")

        except OutputParsingError as ope:
            print(f"Ошибка парсинга на попытке {attempt + 1}: {ope}")
            raw_output = getattr(ope, 'llm_output', str(ope))
            st.warning(f"Попытка {attempt + 1}: Не удалось разобрать ответ LLM. Пробуем снова...")
            print(f"--- Сырой вывод LLM (при ошибке парсинга) ---\n{raw_output}\n---")
            last_error = ope
            time.sleep(2)

        except Exception as e:
            print(f"Неожиданная ошибка на попытке {attempt + 1}: {e}")
            st.error(f"Произошла неожиданная ошибка при генерации: {e}")
            last_error = e
            break

    # Если все попытки не удались
    st.error(f"Не удалось сгенерировать новости после {MAX_RETRIES} попыток.")
    if last_error:
        st.error(f"Последняя ошибка: {last_error}")
        raw_output = getattr(last_error, 'llm_output', None)
        if raw_output:
            st.text_area("Последний сырой ответ от LLM (для отладки):", raw_output, height=200)

    return NewsReport(articles=[])

# --- Блок для локального тестирования (остается как был, но нужны импорты re и ValidationError) ---
if __name__ == '__main__':
    from pydantic import ValidationError # Добавить импорт
    # ... остальной код __main__ ...
    print("Запуск локального теста генератора...")
    # ...