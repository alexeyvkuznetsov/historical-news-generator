# app.py
import streamlit as st # Streamlit импорт должен быть одним из первых
import datetime
import sys
import os
import importlib.metadata # Для диагностики

# --- ПЕРЕМЕЩАЕМ set_page_config СЮДА ---
# Это должна быть ПЕРВАЯ команда Streamlit в скрипте
st.set_page_config(page_title="Исторический ВестникЪ", layout="wide")
# --- КОНЕЦ ПЕРЕМЕЩЕНИЯ ---


# --- БЛОК ДИАГНОСТИКИ ВЕРСИЙ (теперь идет ПОСЛЕ set_page_config) ---
st.write("--- Диагностика среды ---")
try:
    st.write(f"Версия Python: {sys.version}")
    st.write(f"Пути поиска модулей (sys.path): {sys.path}")
    st.write(f"Текущая рабочая директория: {os.getcwd()}")
    # Проверяем наличие папки modules и выводим содержимое
    if 'modules' in os.listdir('.'):
         st.write("Содержимое папки 'modules':", os.listdir('modules'))
    else:
        st.warning("Папка 'modules' не найдена в корневой директории!")
    # Проверка версий библиотек
    libs_to_check = [
        "streamlit", "pandas", "openai", "pydantic",
        "langchain", "langchain_core", "langchain_openai", "langchain_community",
        "sentence-transformers", "faiss-cpu"
    ]
    versions = {}
    for lib in libs_to_check:
        try:
            versions[lib] = importlib.metadata.version(lib)
        except importlib.metadata.PackageNotFoundError:
            versions[lib] = "НЕ НАЙДЕНА!"
    st.write("Версии установленных библиотек:")
    st.json(versions)
    # Попытка тестового импорта парсеров
    st.write("Попытка импорта Output Parsers...")
    from langchain_core.output_parsers import PydanticOutputParser
    st.write("✅ Успешно импортирован PydanticOutputParser")
    try:
        from langchain_core.output_parsers import OutputFixingParser # Попытка импорта для диагностики
        st.write("✅ Успешно импортирован OutputFixingParser (хотя он может быть не использован)")
    except ImportError as ie_fix:
        st.warning(f"⚠️ Ожидаемая ошибка при импорте OutputFixingParser (т.к. он перемещен): {ie_fix}")

except Exception as diag_error:
    st.error(f"Ошибка во время диагностики: {diag_error}")
st.write("--- Конец диагностики ---")
# --- КОНЕЦ БЛОКА ДИАГНОСТИКИ ---


# --- Импорт основного модуля приложения (ПОСЛЕ диагностики) ---
try:
    from modules.generator import generate_news, last_error # Импортируем и last_error
    from modules.models import NewsReport
    st.success("Импорт modules.generator прошел успешно.")
except ImportError as app_import_error:
    st.error(f"Критическая ошибка импорта 'modules.generator': {app_import_error}")
    st.stop()
except Exception as app_other_error:
     st.error(f"Другая ошибка при импорте 'modules.generator': {app_other_error}")
     st.stop()


# --- Основной UI приложения ---
st.title("📜 Исторический ВестникЪ 📰")
st.caption("Генератор псевдо-исторических новостей на базе LLM")

# --- Ввод данных пользователем ---
col1, col2 = st.columns([1, 2])

with col1:
    default_date = datetime.date(1789, 7, 14)
    selected_date = st.date_input(
        "Выберите дату для выпуска газеты:",
        value=default_date,
        min_value=datetime.date(1000, 1, 1),
        max_value=datetime.date.today()
    )
    selected_date_str = selected_date.strftime("%d %B %Y")

    era_options = ["XVIII", "XIX", "XVII", "XX"]
    selected_era = st.selectbox("Стиль какого века предпочитаете?", era_options, index=0)
    num_articles = st.slider("Количество новостей в сводке:", min_value=1, max_value=5, value=3)

    generate_button = st.button("✨ Сгенерировать ВестникЪ!")


# --- Область вывода ---
with col2:
    st.subheader(f"Выпускъ отъ {selected_date_str} ({selected_era} вѣкъ)")

    if generate_button:
        # Проверяем, был ли импорт generate_news успешным ранее
        if 'generate_news' in globals():
            with st.spinner(f"⏳ Редакція '{'Хронографъ'.upper()}' готовитъ свѣжій номеръ..."):
                # Вызываем функцию генерации
                news_report: NewsReport = generate_news(
                    target_date=selected_date_str,
                    era_style=selected_era,
                    num_articles=num_articles
                )

                # Отображаем результат
                if news_report and news_report.articles:
                    st.success("📰 Свѣжій номеръ готовъ!")
                    for i, article in enumerate(news_report.articles):
                        st.markdown(f"---")
                        st.markdown(f"### {i+1}. {article.headline}")
                        st.markdown(f"**Рубрика:** {article.rubric}")
                        st.markdown(f"*{article.date_location}*")
                        st.write(article.body)
                        st.caption(f"Репортажъ велъ: {article.reporter}")
                # Сообщение, если нет новостей, но и не было ошибки (из generate_news)
                elif last_error is None: # Проверяем глобальную переменную last_error
                     st.info("Не удалось сгенерировать новости (возможно, нет данных или событий).")
                # Сообщения об ошибках (st.error / st.warning) выводятся внутри generate_news

        else:
             st.error("Ошибка: функция генерации новостей не была загружена из-за проблем с импортом.")

    else:
        st.info("Выберите дату и нажмите кнопку для генерации новостей.")


# --- Подвал ---
st.markdown("---")
st.caption("Создано с использованием LLM и Streamlit.")