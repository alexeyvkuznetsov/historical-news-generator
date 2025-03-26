import streamlit as st
import datetime
from modules.generator import generate_news
from modules.models import NewsReport # Для type hinting

# --- Настройка страницы ---
st.set_page_config(page_title="Исторический ВестникЪ", layout="wide")

st.title("📜 Исторический ВестникЪ 📰")
st.caption("Генератор псевдо-исторических новостей на базе LLM")

# --- Ввод данных пользователем ---
col1, col2 = st.columns([1, 2])

with col1:
    # Используем date_input для удобного выбора даты
    default_date = datetime.date(1789, 7, 14)
    selected_date = st.date_input(
        "Выберите дату для выпуска газеты:",
        value=default_date,
        min_value=datetime.date(1000, 1, 1), # Ограничим разумно
        max_value=datetime.date.today()
    )
    selected_date_str = selected_date.strftime("%d %B %Y") # Формат для отображения и запроса

    # Опционально: выбор стиля эпохи (можно сделать более продвинутым)
    era_options = ["XVIII", "XIX", "XVII", "XX"] # Добавьте нужные
    selected_era = st.selectbox("Стиль какого века предпочитаете?", era_options, index=0)

    num_articles = st.slider("Количество новостей в сводке:", min_value=1, max_value=5, value=3)

    generate_button = st.button("✨ Сгенерировать ВестникЪ!")

# --- Область вывода ---
with col2:
    st.subheader(f"Выпускъ отъ {selected_date_str} ({selected_era} вѣкъ)")

    if generate_button:
        # Показываем индикатор загрузки во время генерации
        with st.spinner(f"⏳ Редакція '{'Хронографъ'.upper()}' готовитъ свѣжій номеръ..."):
            try:
                # Вызов функции генерации
                # Передаем строку с датой, понятную для поиска событий и LLM
                news_report: NewsReport = generate_news(
                    target_date=selected_date_str,
                    era_style=selected_era,
                    num_articles=num_articles
                )

                if news_report and news_report.articles:
                    st.success("📰 Свѣжій номеръ готовъ!")
                    # Отображение новостей
                    for i, article in enumerate(news_report.articles):
                        st.markdown(f"---")
                        st.markdown(f"### {i+1}. {article.headline}")
                        st.markdown(f"**Рубрика:** {article.rubric}")
                        st.markdown(f"*{article.date_location}*")
                        st.write(article.body)
                        st.caption(f"Репортажъ велъ: {article.reporter}")
                else:
                    st.warning("Не удалось найти событий или сгенерировать новости для этой даты. Попробуйте другую.")

            except FileNotFoundError as fnf:
                st.error(f"Ошибка: Не найден файл с историческими данными. {fnf}")
            except ValueError as ve:
                st.error(f"Ошибка конфигурации: {ve}")
            except Exception as e:
                st.error(f"Произошла непредвиденная ошибка: {e}")
                # Здесь можно добавить логирование для отладки
                # import traceback
                # st.error(traceback.format_exc())

    else:
        st.info("Выберите дату и нажмите кнопку для генерации новостей.")

# --- Подвал (опционально) ---
st.markdown("---")
st.caption("Создано в рамках курса 'Делаем свой AI-продукт'.")