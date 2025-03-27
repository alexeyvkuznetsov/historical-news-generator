# modules/generator.py
import streamlit as st
import os
import time
import json
import re
from dotenv import load_dotenv
# ---> Новые импорты <---
import datetime
import locale
from pydantic import ValidationError
# ---> Конец новых импортов <---


# Импорты LangChain
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core import exceptions
from langchain_core.runnables import RunnablePassthrough

# Импорты вашего проекта
from .models import NewsReport
from .rag import get_retriever

load_dotenv()

# --- Конфигурация API ---
API_KEY = st.secrets.get("FORGETAPI_KEY", os.getenv("FORGETAPI_KEY"))
BASE_URL = st.secrets.get("FORGETAPI_BASE_URL", os.getenv("FORGETAPI_BASE_URL", "https://forgetapi.ru/v1"))
MAX_RETRIES = 2
last_error = None

# ---> Установка локали для парсинга русских месяцев <---
try:
    # Попробуем установить русскую локаль. Может не сработать на всех системах (особенно в Streamlit Cloud без доп. настроек)
    locale.setlocale(locale.LC_TIME, 'ru_RU.UTF-8')
    print("Русская локаль (ru_RU.UTF-8) установлена успешно.")
except locale.Error:
    print("Предупреждение: Не удалось установить локаль ru_RU.UTF-8. Парсинг русских месяцев может не работать.")
    # Можно попробовать альтернативную локаль или оставить системную по умолчанию
    try:
        # Попытка использовать стандартную русскую локаль Windows
        locale.setlocale(locale.LC_TIME, 'russian')
        print("Русская локаль ('russian') установлена успешно.")
    except locale.Error:
        print("Предупреждение: Не удалось установить локаль 'russian'. Используется системная локаль по умолчанию.")
# ---> Конец установки локали <---


# --- Функция get_llm (без изменений) ---
def get_llm():
    if not API_KEY:
        st.error("Критическая ошибка: Не найден API ключ для 'forgetapi'.")
        raise ValueError("FORGETAPI_KEY не найден.")
    if not BASE_URL:
        st.error("Критическая ошибка: Не найден BASE_URL для 'forgetapi'.")
        raise ValueError("FORGETAPI_BASE_URL не найден.")
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        openai_api_key=API_KEY,
        openai_api_base=BASE_URL,
        temperature=0.7,
        request_timeout=120
    )
    return llm

# --- Функция create_generation_chain (без изменений) ---
def create_generation_chain():
    llm = get_llm()
    pydantic_parser = PydanticOutputParser(pydantic_object=NewsReport)
    format_instructions = pydantic_parser.get_format_instructions()
    prompt = ChatPromptTemplate.from_messages([
         ("system", """Ты — остроумный и немного саркастичный редактор исторической газеты 'Хронографъ'.
Твоя задача — написать сводку новостей для выпуска газеты на заданную дату.
Используй следующие реальные исторические события как основу, но добавь детали, юмор, вымышленных персонажей или комментарии в стиле газеты {era_style} века.
ВАЖНО: Весь твой ответ ДОЛЖЕН быть ТОЛЬКО JSON объектом, без какого-либо другого текста до или после него.
JSON должен строго соответствовать следующей структуре (не включай ```json или ``` в свой ответ):
{format_instructions}
Реальные события (контекст):
{context}"""),
        ("user", "Пожалуйста, напиши новости для даты {date_input}. Используй примерно {num_articles} события из контекста. Стиль: {era_style} век.")
    ])
    chain = prompt | llm | pydantic_parser
    return chain

# ---> Новая вспомогательная функция для парсинга дат <---
def parse_date_robust(date_str: str, fmt: str = "%Y-%m-%d") -> datetime.date | None:
    """Пытается распарсить строку даты, возвращает date или None."""
    try:
        return datetime.datetime.strptime(date_str, fmt).date()
    except (ValueError, TypeError):
        # Попытка распарсить только год, если формат не подошел
        try:
             year = int(date_str)
             # Возвращаем дату на середину года для событий, известных только по году
             # Это спорное решение, но позволяет включить их в годовой фильтр
             return datetime.date(year, 6, 15)
        except (ValueError, TypeError):
            print(f"Предупреждение: Не удалось распарсить дату '{date_str}' с форматом '{fmt}' или как год.")
            return None
# ---> Конец новой функции <---


# --- Функция generate_news (ИЗМЕНЕНИЯ в получении и фильтрации контекста) ---
def generate_news(target_date_str: str, era_style: str = "XVIII", num_articles: int = 3, date_window_days: int = 7) -> NewsReport:
    """Основная функция для генерации новостей с фильтрацией по дате."""
    global last_error
    last_error = None
    print(f"Запрос на генерацию: дата='{target_date_str}', стиль={era_style}, статьи={num_articles}, окно={date_window_days} дней")

    # 1. Парсим целевую дату от пользователя
    try:
        # Используем формат, который приходит из Streamlit date_input (%d %B %Y)
        target_date_obj = datetime.datetime.strptime(target_date_str, "%d %B %Y").date()
        print(f"Целевая дата распарсена: {target_date_obj}")
    except ValueError as e:
        st.error(f"Не удалось распознать введенную дату '{target_date_str}'. Ошибка: {e}")
        print(f"Ошибка парсинга целевой даты: {e}")
        last_error = e
        return NewsReport(articles=[])

    # 2. Получаем БОЛЬШЕ кандидатов из RAG
    try:
        # Запрашиваем больше документов, например, в 3 раза больше, чем нужно, но не менее 5
        rag_k = max(num_articles * 3, 10)
        print(f"Запрос к RAG: k={rag_k}")
        retriever = get_retriever(k=rag_k)
        query = f"Исторические события около {target_date_str}" # Запрос оставляем общим
        all_docs = retriever.invoke(query)
        print(f"RAG вернул {len(all_docs)} кандидатов.")
        if not all_docs:
            st.warning(f"RAG не вернул никаких событий для запроса '{query}'.")
            return NewsReport(articles=[])

    except Exception as e:
        st.error(f"Ошибка при поиске событий в RAG: {e}")
        print(f"Полная ошибка RAG: {e}")
        last_error = e
        return NewsReport(articles=[])

    # 3. Фильтруем кандидатов по дате
    filtered_docs = []
    print(f"Фильтрация по дате: окно +/- {date_window_days} дней от {target_date_obj}")
    for doc in all_docs:
        doc_date_str = doc.metadata.get('date')
        if not doc_date_str:
            print(f"Предупреждение: Документ без даты в метаданных: {doc.page_content[:50]}...")
            continue

        # Парсим дату из метаданных (ожидаем YYYY-MM-DD или только YYYY)
        doc_date_obj = parse_date_robust(doc_date_str) # Используем новую функцию

        if doc_date_obj:
            # Проверяем попадание в окно (если задано окно > 0)
            if date_window_days >= 0:
                delta = abs((doc_date_obj - target_date_obj).days)
                if delta <= date_window_days:
                    print(f"  [OK] Документ от {doc_date_obj} попадает в окно (+/-{date_window_days} дн.). Дельта: {delta} дн.")
                    filtered_docs.append(doc)
                # else:
                #     print(f"  [Пропуск] Документ от {doc_date_obj} не попадает в окно (+/-{date_window_days} дн.). Дельта: {delta} дн.")
            else: # Если date_window_days < 0, окно не применяем (берем все)
                 filtered_docs.append(doc)

        # Опционально: можно добавить логику для фильтрации по году/месяцу, если окно не задано

    print(f"После фильтрации по дате осталось {len(filtered_docs)} документов.")

    # Если после фильтрации ничего не осталось
    if not filtered_docs:
        st.warning(f"Не найдено событий в базе данных близко к дате '{target_date_str}' (в пределах +/- {date_window_days} дней).")
        # Можно попробовать ослабить фильтр или сообщить пользователю
        # Например, попробовать найти события за тот же месяц/год? (требует доп. логики)
        return NewsReport(articles=[])

    # 4. Ограничиваем количество и формируем контекст
    # Берем не больше num_articles документов из отфильтрованных
    final_docs = filtered_docs[:num_articles]
    context = "\n\n".join([doc.page_content for doc in final_docs])
    print(f"Итоговый контекст для LLM сформирован из {len(final_docs)} документов.")
    # print(f"Итоговый контекст:\n{context[:500]}...") # Раскомментировать для отладки

    # 5. Генерируем новости (остальная часть функции почти без изменений)
    try:
        chain = create_generation_chain()
        pydantic_parser = PydanticOutputParser(pydantic_object=NewsReport)
        format_instructions = pydantic_parser.get_format_instructions()
    except ValueError as ve:
        st.error(f"Ошибка конфигурации LLM: {ve}")
        last_error = ve
        return NewsReport(articles=[])

    for attempt in range(MAX_RETRIES):
        print(f"Попытка генерации LLM {attempt + 1}/{MAX_RETRIES}...")
        try:
            chain_input = {
                "date_input": target_date_str, # Передаем исходную строку даты
                "era_style": era_style,
                "num_articles": len(final_docs), # Передаем фактическое кол-во статей в контексте
                "context": context, # Передаем отфильтрованный контекст
                "format_instructions": format_instructions
            }
            result = chain.invoke(chain_input)

            if isinstance(result, NewsReport):
                print("Генерация и парсинг LLM прошли успешно.")
                last_error = None
                return result
            else:
                # Обработка неожиданного типа результата
                print(f"Неожиданный тип результата от парсера: {type(result)}. Результат: {result}")
                last_error = exceptions.OutputParsingError(f"Неожиданный тип результата: {type(result)}")
                if isinstance(result, str):
                    try:
                         json_match = re.search(r'\{.*\}', result, re.DOTALL)
                         if json_match:
                            json_str = json_match.group(0)
                            parsed_json = json.loads(json_str)
                            news_report = NewsReport.model_validate(parsed_json)
                            print("Удалось вручную распарсить и валидировать JSON из строки.")
                            last_error = None
                            return news_report
                         else:
                            print("Не удалось найти JSON в строке ответа.")
                            last_error = exceptions.OutputParsingError("Не удалось найти JSON в строке ответа.")
                    except (json.JSONDecodeError, ValidationError) as manual_parse_error:
                        print(f"Ошибка ручного парсинга/валидации JSON: {manual_parse_error}")
                        last_error = exceptions.OutputParsingError(f"Ошибка ручного парсинга/валидации JSON: {manual_parse_error}")

        except exceptions.OutputParsingError as ope:
            # Обработка ошибки парсинга
            print(f"Ошибка парсинга Pydantic на попытке {attempt + 1}: {ope}")
            raw_output = getattr(ope, 'llm_output', str(ope))
            st.warning(f"Попытка {attempt + 1}: Не удалось разобрать ответ LLM. Пробуем снова...")
            print(f"--- Сырой вывод LLM (при ошибке парсинга) ---\n{raw_output}\n---")
            last_error = ope
            time.sleep(2)

        except Exception as e:
            # Обработка других ошибок
            print(f"Неожиданная ошибка на попытке {attempt + 1}: {e}")
            st.error(f"Произошла неожиданная ошибка при генерации: {e}")
            last_error = e
            break

    # Если все попытки не удались
    st.error(f"Не удалось сгенерировать новости после {MAX_RETRIES} попыток.")
    if last_error:
        st.error(f"Детали последней ошибки: {last_error}")
        raw_output = getattr(last_error, 'llm_output', None)
        if raw_output:
            st.text_area("Последний сырой ответ от LLM (для отладки):", str(raw_output), height=200)

    return NewsReport(articles=[])

# --- Блок для локального тестирования (без изменений) ---
if __name__ == '__main__':
    # ... (остальной тестовый блок без изменений) ...
    print("Запуск локального теста генератора...")
    test_date = "07 Сентября 1812" # Пример даты
    try:
        report = generate_news(test_date, era_style="XIX", num_articles=2, date_window_days=3) # Тест с окном +/- 3 дня

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
         print(f"Ошибка конфигурации при тесте: {ve}")
    except ImportError as ie:
         print(f"Ошибка импорта при тесте: {ie}")
    except Exception as e:
         print(f"Непредвиденная ошибка при тесте: {e}")