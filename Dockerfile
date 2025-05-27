# Базовый образ с Python 3.12.10-slim
FROM python:3.12.10-slim

# Установка утилиты uv
RUN pip install --no-cache-dir uv

# Рабочая директория внутри контейнера
WORKDIR /app

# Копируем весь код приложения
COPY . .

# Установка зависимостей в системную среду
RUN uv sync

# По умолчанию порт приложения
ENV PORT=8000
EXPOSE 8000

# Команда запуска приложения
CMD ["uv", "run", "api"]
