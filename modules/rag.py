# modules/rag.py
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import os
from dotenv import load_dotenv

load_dotenv()

# --- Константы ---
# Путь к папке с предсозданным индексом FAISS (относительно корня проекта)
INDEX_LOAD_PATH = "faiss_index_historical"
# Имя модели эмбеддингов, которая использовалась для СОЗДАНИЯ индекса
# Оно нужно для корректной ЗАГРУЗКИ индекса
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

# Используем кэширование Streamlit для загрузки модели и индекса
@st.cache_resource
def get_embeddings_loader():
    """Загружает ТОЛЬКО модель эмбеддингов (нужна для FAISS.load_local)."""
    print("Инициализация модели эмбеддингов для загрузки индекса...")
    try:
        # Модель нужна для FAISS.load_local, но не для генерации эмбеддингов на лету
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        print("Модель эмбеддингов загружена.")
        return embeddings
    except Exception as e:
        st.error(f"Не удалось загрузить модель эмбеддингов '{EMBEDDING_MODEL_NAME}': {e}")
        print(f"Полная ошибка загрузки эмбеддингов: {e}")
        return None

@st.cache_resource
def get_vector_store():
    """Загружает предсозданный индекс FAISS (кэшируется)."""
    print(f"Попытка загрузки индекса FAISS из папки: {INDEX_LOAD_PATH}...")
    embeddings = get_embeddings_loader()
    if embeddings is None:
        st.error("Векторное хранилище не может быть загружено без модели эмбеддингов.")
        return None

    if not os.path.exists(INDEX_LOAD_PATH):
         st.error(f"Ошибка: Папка с индексом FAISS не найдена по пути: '{INDEX_LOAD_PATH}'. Убедитесь, что она существует и содержит файлы 'index.faiss' и 'index.pkl'.")
         print(f"Ошибка FileNotFoundError при загрузке vector_store: папка '{INDEX_LOAD_PATH}' не найдена.")
         return None
    # Проверим наличие файлов внутри папки
    faiss_file = os.path.join(INDEX_LOAD_PATH, "index.faiss")
    pkl_file = os.path.join(INDEX_LOAD_PATH, "index.pkl")
    if not os.path.exists(faiss_file) or not os.path.exists(pkl_file):
        st.error(f"Ошибка: В папке '{INDEX_LOAD_PATH}' отсутствуют необходимые файлы 'index.faiss' или 'index.pkl'.")
        print(f"Ошибка: Не найдены index.faiss или index.pkl в '{INDEX_LOAD_PATH}'")
        return None

    try:
        # Загружаем индекс с диска
        # allow_dangerous_deserialization=True необходимо для загрузки .pkl файла FAISS
        vector_store = FAISS.load_local(
            folder_path=INDEX_LOAD_PATH,
            embeddings=embeddings,
            allow_dangerous_deserialization=True
        )
        print("Предсозданный индекс FAISS успешно загружен.")
        return vector_store
    except Exception as e:
        st.error(f"Неожиданная ошибка при загрузке индекса FAISS из '{INDEX_LOAD_PATH}': {e}")
        print(f"Полная ошибка при загрузке vector_store: {e}")
        return None

def get_retriever(k=5):
    """Возвращает настроенный ретривер из загруженного векторного хранилища."""
    vector_store = get_vector_store()
    if vector_store is None:
        raise RuntimeError("Векторное хранилище для RAG не инициализировано.")

    return vector_store.as_retriever(search_kwargs={"k": k})

# --- Блок для локального тестирования RAG ---
if __name__ == '__main__':
    print("Запуск локального теста RAG модуля (с загрузкой индекса)...")
    try:
        retriever = get_retriever()
        if retriever:
            print("Ретривер успешно создан из загруженного индекса.")
            query = "Бородинское сражение"
            results = retriever.invoke(query)
            print(f"\nРезультаты для запроса '{query}':")
            if results:
                for doc in results:
                    print(f"- {doc.page_content}")
                    print(f"  (Метаданные: {doc.metadata})\n")
            else:
                print("Документы не найдены.")
        else:
            print("Не удалось создать ретривер (вероятно, индекс не загрузился).")

    except RuntimeError as rte:
        print(f"Ошибка выполнения при тесте RAG: {rte}")
    except Exception as e:
        print(f"Непредвиденная ошибка при тестировании RAG: {e}")