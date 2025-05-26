from config.config import Config
import time


def translate(input_text, model, tokenizer):
    """
    This function handles the API request based on the model (hosted or local) and streams the response.
    """
    time_start = time.time() # замеряем время начала
    try:
        tokens_encoded = tokenizer(input_text, max_length=Config.MAX_SEQUENCE_LEN, return_tensors="pt", truncation=True, padding=True) # токенизируем данные (max_length — максимальное число токенов в документе, return_tensors — тип возвращаемых данных, np для numpy.array, pt для torch.tensor; truncation и padding — обрезание лишних токенов и автозаполнение недостающих до max_length)
        tokens_generated = model.generate(**tokens_encoded)

        translation = tokenizer.batch_decode(tokens_generated, skip_special_tokens=True)[0]
        latency = time.time() - time_start
        tokens_count = tokens_encoded["input_ids"].shape[1]

        return translation, latency, tokens_count
    except Exception as e:
        latency = time.time() - time_start
        return e, latency, None
