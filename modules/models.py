from pydantic import BaseModel, Field
from typing import List

class NewsArticle(BaseModel):
    """Структура одной псевдо-исторической новости."""
    headline: str = Field(description="Яркий, привлекающий внимание заголовок новости в стиле эпохи.")
    date_location: str = Field(description="Дата и место события, как указано в газете (например, 'Парижъ, 14 Июля 1789 г.')")
    body: str = Field(description="Текст новости (2-4 предложения), основанный на реальном событии, но с добавлением юмора/вымысла.")
    rubric: str = Field(description="Рубрика газеты (например, 'Столичные Вѣсти', 'Заграничныя Извѣстія', 'Происшествія', 'Культура и Нравы')")
    reporter: str = Field(description="Вымышленное имя репортера (например, 'Нашъ собственный корреспондентъ М. Невраловъ')")

class NewsReport(BaseModel):
    """Структура для всего отчета с несколькими новостями."""
    articles: List[NewsArticle] = Field(description="Список сгенерированных новостных статей.")