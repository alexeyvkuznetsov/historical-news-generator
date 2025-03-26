import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
import os
from dotenv import load_dotenv

load_dotenv() # Загрузка переменных из .env (для локального запуска)

DATA_PATH = "data/historical_events.csv"
# Выберите модель эмбеддингов (мультиязычная модель хороша для русских текстов)
# https://huggingface.co/sentence-transformers/paraphrase-multilingual-mpnet-base-v2
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

# Кэширование для ускорения загрузки модели и индекса
_vector_store = None
_embeddings = None

def get_embeddings():
    """Загружает модель эмбеддингов."""
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    return _embeddings

def load_documents_from_csv(file_path):
    """Загружает данные из CSV и преобразует в документы LangChain."""
    df = pd.read_csv(file_path)
    documents = []
    for index, row in df.iterrows():
        # Объединяем важные поля в текст документа для поиска
        content = f"Дата: {row['date']}. Событие: {row['event_description']}"
        if pd.notna(row.get('location')):
             content += f". Место: {row['location']}"
        if pd.notna(row.get('category')):
             content += f". Категория: {row['category']}"

        # Используем дату как метаданные, если понадобится точная фильтрация
        # Но для семантического поиска важнее описание в content
        metadata = {"date": row['date'], "source": file_path}
        if pd.notna(row.get('location')):
            metadata["location"] = row['location']

        documents.append(Document(page_content=content, metadata=metadata))
    return documents

def get_vector_store():
    """Создает или загружает векторное хранилище."""
    global _vector_store
    if _vector_store is None:
        if not os.path.exists(DATA_PATH):
            raise FileNotFoundError(f"Файл данных не найден: {DATA_PATH}")

        print("Загрузка документов...")
        documents = load_documents_from_csv(DATA_PATH)
        if not documents:
             raise ValueError("Не удалось загрузить документы из CSV.")

        print("Создание эмбеддингов и векторного хранилища FAISS...")
        embeddings = get_embeddings()
        # Создаем FAISS индекс из документов
        # Если данных много, это может занять время при первом запуске
        _vector_store = FAISS.from_documents(documents, embeddings)
        print("Векторное хранилище готово.")
        # Опционально: сохранение индекса для быстрого старта в будущем
        # _vector_store.save_local("faiss_index")
        # И тогда при старте можно проверять наличие папки и делать FAISS.load_local(...)
    return _vector_store

def get_retriever(k=5):
    """Возвращает настроенный ретривер."""
    vector_store = get_vector_store()
    # k - количество возвращаемых релевантных документов
    return vector_store.as_retriever(search_kwargs={"k": k})

if __name__ == '__main__':
    # Тестирование RAG модуля
    try:
        retriever = get_retriever()
        print("Ретривер успешно создан.")
        query = "Что случилось во Франции в июле 1789?"
        results = retriever.invoke(query)
        print(f"\nРезультаты для запроса '{query}':")
        for doc in results:
            print(f"- {doc.page_content}")
            print(f"  (Метаданные: {doc.metadata})\n")
    except Exception as e:
        print(f"Ошибка при инициализации или тестировании RAG: {e}")