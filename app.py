# app.py
import streamlit as st
import datetime
import sys
import os

# --- БЛОК ДИАГНОСТИКИ ВЕРСИЙ (ДОБАВИТЬ В САМОМ НАЧАЛЕ) ---
st.write("--- Диагностика среды ---")
try:
    st.write(f"Версия Python: {sys.version}")
    st.write(f"Пути поиска модулей (sys.path): {sys.path}")
    st.write(f"Текущая рабочая директория: {os.getcwd()}")
    st.write("Содержимое текущей директории:", os.listdir('.'))
    # Проверяем наличие папки modules
    if 'modules' in os.listdir('.'):
        st.write("Содержимое папки 'modules':", os.listdir('modules'))
    else:
        st.warning("Папка 'modules' не найдена в корневой директории!")

    # Используем importlib.metadata для получения версий
    import importlib.metadata
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

    # Попытка импорта проблемных модулей прямо здесь для теста
    st.write("Попытка импорта Output Parsers...")
    from langchain_core.output_parsers import PydanticOutputParser
    st.write("✅ Успешно импортирован PydanticOutputParser")
    # Попробуем импортировать и второй, чтобы увидеть, падает ли он
    try:
        from langchain_core.output_parsers import OutputFixingParser
        st.write("✅ Успешно импортирован OutputFixingParser")
    except ImportError as ie_fix:
        st.error(f"❌ Ошибка при импорте OutputFixingParser: {ie_fix}")

except Exception as diag_error:
    st.error(f"Ошибка во время диагностики: {diag_error}")
st.write("--- Конец диагностики ---")
# --- КОНЕЦ БЛОКА ДИАГНОСТИКИ ---


# --- Остальной код приложения ---
# Попробуем импортировать generate_news ПОСЛЕ диагностики
try:
    from modules.generator import generate_news
    from modules.models import NewsReport
    st.success("Импорт modules.generator прошел успешно.") # Сообщение об успехе импорта
except ImportError as app_import_error:
    st.error(f"Критическая ошибка импорта 'modules.generator': {app_import_error}")
    # Можно остановить приложение, если основной модуль не грузится
    st.stop()
except Exception as app_other_error:
     st.error(f"Другая ошибка при импорте 'modules.generator': {app_other_error}")
     st.stop()


# --- Настройка страницы ---
st.set_page_config(page_title="Исторический ВестникЪ", layout="wide")
st.title("📜 Исторический ВестникЪ 📰")
st.caption("Генератор псевдо-исторических новостей на базе LLM")

# --- Ввод данных пользователем ---
# ... (код выбора даты, стиля, кнопки и т.д. остается как был) ...
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
                news_report: NewsReport = generate_news(
                    target_date=selected_date_str,
                    era_style=selected_era,
                    num_articles=num_articles
                )

                if news_report and news_report.articles:
                    st.success("📰 Свѣжій номеръ готовъ!")
                    for i, article in enumerate(news_report.articles):
                        st.markdown(f"---")
                        st.markdown(f"### {i+1}. {article.headline}")
                        # ... (остальной код отображения) ...
                        st.markdown(f"**Рубрика:** {article.rubric}")
                        st.markdown(f"*{article.date_location}*")
                        st.write(article.body)
                        st.caption(f"Репортажъ велъ: {article.reporter}")
                # Убрали else, т.к. сообщения об ошибках теперь внутри generate_news
                # Можно добавить сообщение, если просто не нашлось событий
                elif 'last_error' not in globals() or globals()['last_error'] is None:
                     st.info("Не удалось сгенерировать новости (возможно, нет данных).")

        else:
             st.error("Ошибка: функция генерации новостей не была загружена из-за проблем с импортом.")

    else:
        st.info("Выберите дату и нажмите кнопку для генерации новостей.")


# --- Подвал ---
st.markdown("---")
st.caption("Создано с использованием LLM и Streamlit.")

# Глобальная переменная для ошибки (хотя лучше передавать ее иначе)
last_error = None