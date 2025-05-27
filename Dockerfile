# Базовый образ с Python 3.12.10-slim
FROM python:3.12.10-slim

# Рабочая директория внутри контейнера
WORKDIR /app

# Установка утилиты uv
RUN pip install --no-cache-dir uv

# Копируем файлы для установки зависимостей
COPY pyproject.toml .

COPY README.md .

# Установка зависимостей из pyproject.toml
RUN uv sync

# Копируем весь код приложения
COPY . .

# По умолчанию порт приложения
ENV PORT=8000
EXPOSE 8000

# Команда запуска приложения
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "${PORT}"]
