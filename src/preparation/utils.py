from pydub import AudioSegment


# Функция для конвертации ogg в wav
def convert_ogg_to_wav(input_file: str, output_file: str) -> None:
    try:
        # Загружаем ogg файл
        audio = AudioSegment.from_file(input_file, format="ogg")

        # Сохраняем как wav
        audio.export(output_file, format="wav")
    except Exception as e:
        print(f"convert_ogg_to_wav: Ошибка при конвертации: {e}")
