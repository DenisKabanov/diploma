import os
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv(".env", override=True, verbose=True)

class Config:
    """
    A configuration class that retrieves environment variables and stores configuration settings.
    """

    MODELS_DIR = os.getenv("MODELS_DIR") # путь до папки с моделями

    MAX_SEQUENCE_LEN = int(os.getenv("MAX_SEQUENCE_LEN")) # оптимальное число токенов, на которые будет разбит документ (если не достаёт — padding, если перебор — truncation), определялось по гистограмме распределения числа токенов в текстах
    
    AVAILABLE_MODELS = [
        "T5"
    ]

    REAL_MODELS_NAMES = {
        "T5": "utrobinmv/t5_translate_en_ru_zh_small_1024"
    }

    SOURCE_LANG = [
        "Русский",
        "Английский"
    ]

    TARGET_LANG = [
        "Английский",
        "Русский"
    ]
