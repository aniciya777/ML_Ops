FROM python:3.12.10-slim

RUN apt update

RUN apt install -y --no-install-recommends ffmpeg git && \
    apt clean

# Установка утилиты uv
RUN pip3 install uv

# Рабочая директория внутри контейнера
WORKDIR /app

COPY pyproject.toml .
COPY uv.lock .
COPY README.md .
RUN mkdir "src"

# Установка зависимостей в системную среду
RUN uv sync --locked

# По умолчанию порт приложения
ENV PORT=8000

EXPOSE 8000

ENV TF_CPP_MIN_LOG_LEVEL=2
ENV AUDIO_LENGTH=41600

RUN git config --global user.name  "aniciya777" \
 && git config --global user.email "kisaost777@gmail.com"

# Копируем весь код приложения
COPY . .

# Команда запуска пайплайна
ENTRYPOINT ["./entrypoint.sh"]
