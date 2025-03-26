# modules/rag.py
import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
import os
import streamlit as st # Импортируем Streamlit для @st.cache_resource
from dotenv import load_dotenv

load_dotenv() # Загрузка переменных из .env (для локального запуска)

DATA_PATH = "data/historical_events.csv"
# Выберите модель эмбеддингов (мультиязычная модель хороша для русских текстов)
# https://huggingface.co/sentence-transformers/paraphrase-multilingual-mpnet-base-v2
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

# Используем кэширование Streamlit для тяжелых объектов
# Это предотвратит повторную загрузку модели и создание индекса при каждом взаимодействии с UI
@st.cache_resource
def get_embeddings():
    """Загружает модель эмбеддингов (кэшируется)."""
    print("Инициализация модели эмбеддингов...")
    try:
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        print("Модель эмбеддингов загружена.")
        return embeddings
    except Exception as e:
        st.error(f"Не удалось загрузить модель эмбеддингов '{EMBEDDING_MODEL_NAME}': {e}")
        print(f"Полная ошибка загрузки эмбеддингов: {e}")
        return None # Возвращаем None, чтобы обозначить ошибку

def load_documents_from_csv(file_path):
    """Загружает данные из CSV и преобразует в документы LangChain."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Файл данных не найден: {file_path}")
    try:
        df = pd.read_csv(file_path)
        documents = []
        for index, row in df.iterrows():
            # Объединяем важные поля в текст документа для поиска
            content = f"Дата: {row['date']}. Событие: {row['event_description']}"
            if pd.notna(row.get('location')):
                 content += f". Место: {row['location']}"
            if pd.notna(row.get('category')):
                 content += f". Категория: {row['category']}"

            # Используем дату и другие поля как метаданные
            metadata = {"date": str(row['date']), "source": file_path} # Убедимся что дата это строка
            if pd.notna(row.get('location')):
                metadata["location"] = row['location']
            if pd.notna(row.get('category')):
                metadata["category"] = row['category']

            documents.append(Document(page_content=content, metadata=metadata))
        print(f"Загружено {len(documents)} документов из {file_path}")
        return documents
    except Exception as e:
        print(f"Ошибка при чтении или обработке CSV файла {file_path}: {e}")
        raise # Перевыбрасываем ошибку, чтобы она была видна выше

@st.cache_resource
def get_vector_store():
    """Создает или загружает векторное хранилище (кэшируется)."""
    print("Проверка и загрузка векторного хранилища...")
    embeddings = get_embeddings()
    if embeddings is None:
        st.error("Векторное хранилище не может быть создано без модели эмбеддингов.")
        return None

    try:
        print("Загрузка документов из CSV...")
        documents = load_documents_from_csv(DATA_PATH)
        if not documents:
             st.warning("Не удалось загрузить документы из CSV, векторное хранилище будет пустым.")
             # Можно создать пустой индекс или вернуть None, в зависимости от желаемого поведения
             # return None
             # Создадим пустой индекс, чтобы избежать ошибок дальше, но RAG не будет работать
             print("Создание пустого индекса FAISS.")
             # Создание пустого индекса требует хотя бы одного вектора размерности модели
             dummy_vector = embeddings.embed_query("пустышка")
             import numpy as np
             index = FAISS.from_embeddings([("пустышка", dummy_vector)], embeddings)
             return index


        print("Создание эмбеддингов для документов и индекса FAISS...")
        # Создаем FAISS индекс из документов
        # Это может занять время при первом запуске после очистки кэша Streamlit
        vector_store = FAISS.from_documents(documents, embeddings)
        print("Векторное хранилище FAISS готово.")
        return vector_store

    except FileNotFoundError as fnf:
        st.error(f"Ошибка доступа к данным для RAG: {fnf}")
        print(f"Ошибка FileNotFoundError при создании vector_store: {fnf}")
        return None
    except Exception as e:
        st.error(f"Неожиданная ошибка при создании векторного хранилища: {e}")
        print(f"Полная ошибка при создании vector_store: {e}")
        return None

def get_retriever(k=5):
    """Возвращает настроенный ретривер из кэшированного векторного хранилища."""
    vector_store = get_vector_store()
    if vector_store is None:
        # Если хранилище не создалось, ретривер тоже не создать
        st.error("Ошибка: Ретривер не может быть создан, так как векторное хранилище недоступно.")
        # Можно выбросить исключение или вернуть None, чтобы обработать в вызывающей функции
        raise RuntimeError("Векторное хранилище для RAG не инициализировано.")
        # return None

    # k - количество возвращаемых релевантных документов
    return vector_store.as_retriever(search_kwargs={"k": k})

# --- Блок для локального тестирования RAG ---
if __name__ == '__main__':
    print("Запуск локального теста RAG модуля...")
    # В локальном тесте кэширование Streamlit не работает,
    # объекты будут создаваться каждый раз.
    try:
        retriever = get_retriever()
        if retriever:
            print("Ретривер успешно создан.")
            query = "Что случилось во Франции в июле 1789?"
            # Используем invoke для получения документов
            results = retriever.invoke(query)
            print(f"\nРезультаты для запроса '{query}':")
            if results:
                for doc in results:
                    print(f"- {doc.page_content}")
                    print(f"  (Метаданные: {doc.metadata})\n")
            else:
                print("Документы не найдены.")
        else:
            print("Не удалось создать ретривер.")

    except RuntimeError as rte:
        print(f"Ошибка выполнения при тесте RAG: {rte}")
    except Exception as e:
        print(f"Непредвиденная ошибка при тестировании RAG: {e}")