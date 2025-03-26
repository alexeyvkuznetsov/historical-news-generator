# app.py
import streamlit as st
import datetime
from modules.generator import generate_news # Убедитесь, что импорт правильный
from modules.models import NewsReport

# --- Настройка страницы ---
st.set_page_config(page_title="Исторический ВестникЪ", layout="wide")

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
    # Формат для отображения и передачи в функцию генерации
    # Используем формат, который LLM легко поймет
    selected_date_str = selected_date.strftime("%d %B %Y")

    era_options = ["XVIII", "XIX", "XVII", "XX"]
    selected_era = st.selectbox("Стиль какого века предпочитаете?", era_options, index=0)
    num_articles = st.slider("Количество новостей в сводке:", min_value=1, max_value=5, value=3)

    generate_button = st.button("✨ Сгенерировать ВестникЪ!")

# --- Область вывода ---
with col2:
    st.subheader(f"Выпускъ отъ {selected_date_str} ({selected_era} вѣкъ)")

    if generate_button:
        with st.spinner(f"⏳ Редакція '{'Хронографъ'.upper()}' готовитъ свѣжій номеръ..."):
            # Вызов функции генерации происходит внутри блока try...except в самой функции
            news_report: NewsReport = generate_news(
                target_date=selected_date_str,
                era_style=selected_era,
                num_articles=num_articles
            )

            # Отображение результата или сообщения об ошибке (сообщения теперь внутри generate_news)
            if news_report and news_report.articles:
                st.success("📰 Свѣжій номеръ готовъ!")
                for i, article in enumerate(news_report.articles):
                    st.markdown(f"---")
                    st.markdown(f"### {i+1}. {article.headline}")
                    st.markdown(f"**Рубрика:** {article.rubric}")
                    st.markdown(f"*{article.date_location}*")
                    st.write(article.body)
                    st.caption(f"Репортажъ велъ: {article.reporter}")
            elif not last_error: # Если ошибок не было, но и новостей нет (например, не нашлось событий)
                 st.info("Не удалось сгенерировать новости для этой даты (возможно, нет данных).")
            # Сообщения об ошибках теперь выводятся внутри generate_news через st.error / st.warning

    else:
        st.info("Выберите дату и нажмите кнопку для генерации новостей.")

# --- Подвал ---
st.markdown("---")
st.caption("Создано с использованием LLM и Streamlit.")

# Переменная для отслеживания последней ошибки (если нужно вне generate_news)
# Инициализируем ее перед блоком if generate_button:
last_error = None