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
        "T5 finetuned",
        "T5 base",
        "T5 pruned unstructured 0.2",
        "T5 pruned unstructured 0.33",
        "T5 pruned unstructured 0.5",
        "T5 pruned structured 0.1",
        "T5 pruned structured 0.15",
        "T5 pruned structured 0.20",
        "T5 pruned structured 0.25",
        "mT5",
        "MBart",
        "M2M100",
        "Marian",
        "LlaMA3.2"
    ]

    REAL_MODELS_NAMES = {
        "T5 finetuned": "utrobinmv/t5_translate_en_ru_zh_small_1024_finetuned",
        "T5 base": "utrobinmv/t5_translate_en_ru_zh_small_1024",
        "T5 pruned unstructured 0.2": "t5_pruned_0.2_finetuned",
        "T5 pruned unstructured 0.33": "t5_pruned_0.33_finetuned",
        "T5 pruned unstructured 0.5": "t5_pruned_0.5_finetuned",
        "T5 pruned structured 0.1": "t5_pruned_structured_0.1_finetuned",
        "T5 pruned structured 0.15": "t5_pruned_structured_0.15_finetuned",
        "T5 pruned structured 0.20": "t5_pruned_structured_0.2_finetuned",
        "T5 pruned structured 0.25": "t5_pruned_structured_0.25_finetuned",
        "mT5": "cointegrated/rut5-base-multitask",
        "MBart": "facebook/mbart-large-50-many-to-many-mmt",
        "M2M100": "facebook/m2m100_418M",
        "Marian": "Helsinki-NLP/opus-mt-ine-ine",
        "LlaMA3.2": "Vikhrmodels/Vikhr-Llama-3.2-1B-Instruct"
    }

    AVAILABLE_RUNTIMES = [
        "PyTorch",
        # "ExecuTorch",
        "ONNX",
        "openVINO"
    ]

    SOURCE_LANG = [
        "Русский",
        "Английский"
    ]

    TARGET_LANG = [
        "Английский",
        "Русский"
    ]
