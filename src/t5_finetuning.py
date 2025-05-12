import os # для взаимодействия с системой
import numpy as np # для работы с массивами
import pandas as pd # для удобной работы с датасетом
import random as random # для работы со случайностью
from dotenv import load_dotenv # для загрузки переменных окружения

from datasets import Dataset, load_dataset, load_from_disk # для работы с HuggingFace датасетами

import torch # для работы с моделями torch
from transformers import T5ForConditionalGeneration, T5Tokenizer # для работы с T5 моделью
from transformers import DataCollatorForSeq2Seq # для сборщика данных (чтобы)
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer # для обучения модели
from transformers import EarlyStoppingCallback # callback для ранней остановки обучения

import time # для отслеживания времени выполнения
import matplotlib.pyplot as plt # для построения графиков
import evaluate # для подсчёта метрик
from transformers.modelcard import parse_log_history # для парсинга логов обучения через Trainer




load_dotenv() # загрузка переменных окружения
DATA_DIR = os.getenv("DATA_DIR")
RESULTS_DIR = os.getenv("RESULTS_DIR")
MODELS_DIR = os.getenv("MODELS_DIR")
MODEL_NAME = os.getenv("MODEL_NAME")
DATASET_NAME_HF = os.getenv("DATASET_NAME_HF")
DATASET_NAME_LOC = os.getenv("DATASET_NAME_LOC")

MAX_SEQUENCE_LEN = int(os.getenv("MAX_SEQUENCE_LEN"))
FP16 = bool(int(os.getenv("FP16")))

RANDOM_STATE = int(os.getenv("RANDOM_STATE"))
TEST_SIZE = float(os.getenv("TEST_SIZE"))
TEST_MAX_SAMPLES = int(os.getenv("TEST_MAX_SAMPLES"))
TRAIN_MAX_SAMPLES = int(os.getenv("TRAIN_MAX_SAMPLES"))

EPOCHS = int(os.getenv("EPOCHS"))
EPOCHS_PATIENCE = int(os.getenv("EPOCHS_PATIENCE"))
LEARNING_RATE = float(os.getenv("LEARNING_RATE"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE"))

DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

print(f"Status: \n\
      DATA_DIR: {DATA_DIR}\n\
      RESULTS_DIR: {RESULTS_DIR}\n\
      MODELS_DIR: {MODELS_DIR}\n\
      MODEL_NAME: {MODEL_NAME}\n\
      DATASET_NAME_HF: {DATASET_NAME_HF}\n\
      DATASET_NAME_LOC: {DATASET_NAME_LOC}\n\
      MAX_SEQUENCE_LEN: {MAX_SEQUENCE_LEN}\n\
      FP16: {FP16}\n\
      RANDOM_STATE: {RANDOM_STATE}\n\
      TEST_SIZE: {TEST_SIZE}\n\
      TEST_MAX_SAMPLES: {TEST_MAX_SAMPLES}\n\
      TRAIN_MAX_SAMPLES: {TRAIN_MAX_SAMPLES}\n\
      EPOCHS: {EPOCHS}\n\
      EPOCHS_PATIENCE: {EPOCHS_PATIENCE}\n\
      LEARNING_RATE: {LEARNING_RATE}\n\
      BATCH_SIZE: {BATCH_SIZE}\n\
      DEVICE: {DEVICE}\n\
      ")



if not os.path.exists(MODELS_DIR + MODEL_NAME):
    print("Скачиваю и сохраняю модель...")
    model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)

    model.save_pretrained(MODELS_DIR + MODEL_NAME, from_pt=True) # сохранение модели
    tokenizer.save_pretrained(MODELS_DIR + MODEL_NAME) # сохранение токенизатора
else:
    print(f"Модель по пути {MODELS_DIR + MODEL_NAME} уже была сохранена ранее, используем её!")
    model = T5ForConditionalGeneration.from_pretrained(MODELS_DIR + MODEL_NAME)
    tokenizer = T5Tokenizer.from_pretrained(MODELS_DIR + MODEL_NAME)

params_all = sum(p.numel() for p in model.parameters())
params_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Число обучаемых параметров: {params_trainable} из {params_all}, то есть ~{params_trainable / params_all * 100:.2f}%.") # считаем число параметров




if not os.path.exists(DATA_DIR + DATASET_NAME_LOC):
    print("Скачиваю и сохраняю датасет...")
    dataset = load_dataset(DATASET_NAME_HF, name="eng_Latn-rus_Cyrl") # скачивание датасета, name — название subset_а с HuggingFace
    dataset.save_to_disk(DATA_DIR + DATASET_NAME_LOC) # локальное сохранение датасета (в формате arrow)
else:
    print(f"Датасет по пути {DATA_DIR + DATASET_NAME_LOC} уже был сохранён ранее, используем его!")
    dataset = load_from_disk(DATA_DIR + DATASET_NAME_LOC)

def preprocess_function(data: Dataset, random_state=RANDOM_STATE):
    random.seed(random_state) # Set the random number generator to a fixed sequence.
    samples_count = len(dataset["train"]) # общее число сэмплов в датасете

    reflected_idx = set(random.sample(range(0, samples_count), int(samples_count/2))) # индексы отражаемых сэмплов (set — для сортировки и удобного вычитания)
    regular_idx = set(range(0, samples_count)) - reflected_idx

    data["new_src"] = ["translate to ru: " + sample if idx in regular_idx else "translate to en: " + data["tgt"][idx] for idx, sample in enumerate(data["src"])]
    data["new_tgt"] = [sample if idx in regular_idx else data["src"][idx] for idx, sample in enumerate(data["tgt"])]
    model_inputs = tokenizer(data["new_src"], text_target=data["new_tgt"], max_length=MAX_SEQUENCE_LEN, return_tensors="pt", truncation=True, padding=True)
    return model_inputs

if not os.path.exists(DATA_DIR + DATASET_NAME_LOC + "_t5_processed"):
    print("Обрабатываю датасет и сохраняю...")
    dataset = dataset.map(preprocess_function, batched=True)
    dataset = dataset.remove_columns(["provenance", "src", "tgt"]) # удаление ненужной колонки
    dataset = dataset.rename_column("new_src", "src") # переименовываем колонку
    dataset = dataset.rename_column("new_tgt", "tgt") # переименовываем колонку
    dataset = dataset["train"].train_test_split(test_size=TEST_SIZE, shuffle=True, seed=RANDOM_STATE) # разбиение датасета на тестовую и обучающую выборки

    dataset.save_to_disk(DATA_DIR + DATASET_NAME_LOC + "_t5_processed") # локальное сохранение датасета (в формате arrow)
else:
    print(f"Датасет по пути {DATA_DIR + DATASET_NAME_LOC + '_t5_processed'} уже был сохранён ранее, используем его!")
    dataset = load_from_disk(DATA_DIR + DATASET_NAME_LOC + "_t5_processed")


dataset["train"] = dataset["train"].select(range(TRAIN_MAX_SAMPLES))
dataset["test"] = dataset["test"].select(range(TEST_MAX_SAMPLES))

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, return_tensors="pt") # сборщик данных для обучения, return_tensors — тип возвращаемого тензора (pt — Torch, tf — Tensorflow, np — Numpy)




metric_BLEU = evaluate.load("bleu") # загружаем метрику

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]

    return preds, labels

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    preds_decoded = tokenizer.batch_decode(preds, skip_special_tokens=True)

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    labels_decoded = tokenizer.batch_decode(labels, skip_special_tokens=True)

    preds_decoded, labels_decoded = postprocess_text(preds_decoded, labels_decoded)

    result = metric_BLEU.compute(predictions=preds_decoded, references=labels_decoded)
    result = {"bleu": result["bleu"]}

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["mean_len"] = np.mean(prediction_lens)
    result = {k: round(v, 5) for k, v in result.items()} # округляем float до 5 знаков после запятой
    return result

if not os.path.exists(RESULTS_DIR + MODEL_NAME + "_finetuned"):
    os.makedirs(RESULTS_DIR + MODEL_NAME + "_finetuned")




training_args = Seq2SeqTrainingArguments(
    output_dir=RESULTS_DIR + MODEL_NAME + "_finetuned",
    # evaluation_strategy ="epoch",
    evaluation_strategy="steps", # стратегия оценивания модели, "steps" - через несколько вызовов forward pass (нужен для работы EarlyStoping), "epoch" — по эпохам
    eval_steps=int(dataset["train"].shape[0] / BATCH_SIZE), # подсчёт метрик и сохранение происходят каждые eval_steps шагов (нужно для работы для работы EarlyStoping), имитирующие одну эпоху
    save_strategy="steps", # стратегия сохранения модели, "steps" - через несколько вызовов forward pass (нужен для работы EarlyStoping), "epoch" — по эпохам
    save_steps=int(dataset["train"].shape[0] / BATCH_SIZE), # для работы должен быть кратен eval_steps, если стратегии стоят как "steps"
    logging_strategy="steps", # стратегия подсчёта метрик на обучающей части
    logging_steps=int(dataset["train"].shape[0] / BATCH_SIZE),
    learning_rate=LEARNING_RATE, # шаг обучения
    per_device_train_batch_size=BATCH_SIZE, # размер батча при обучении
    per_device_eval_batch_size=BATCH_SIZE, # размер батча при валидации
    weight_decay=0.01,
    save_total_limit=3, # количество сохраняемых чекпоинтов
    num_train_epochs=EPOCHS,
    predict_with_generate=True,

    fp16=FP16, # проводить ли обучение в float16 вместо float32
    # bf16=False, # проводить обучение в float16 от google (так обучены некоторые модели google и они не умеют работать с обычным fp16 из-за возникающих ошибок переполнения)

    load_best_model_at_end=True, # загружать ли в конце обучения чекпоинт с лучшей метрикой (также 100% сохраняет лучший чекпоинт)
    metric_for_best_model="eval_bleu", # название метрики, по которой будет определяться лучший чекпоинт обучения
    greater_is_better=True, # должна ли отслеживаемая метрика увеличиваться
    use_cpu=True if DEVICE == "cpu" else False, # использовать ли для подсчёта устройства, отличные от CPU
    push_to_hub=False,
    report_to="none" # не запускать wandb backend
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=tokenizer, # tokenizer -> processing_class с определённой версии PyTorch
    # processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=EPOCHS_PATIENCE)] # callback_и для обучения, EarlyStoppingCallback — ранняя остановка
)

time_start = time.time() # замеряем время начала обучения
trainer.train()
print(f"Время, затраченное на обучение: {time.time()  - time_start} секунд.")

trainer.model.save_pretrained(MODELS_DIR + MODEL_NAME + "_finetuned", from_pt=True) # сохранение модели
trainer.tokenizer.save_pretrained(MODELS_DIR + MODEL_NAME + "_finetuned", from_pt=True) # сохранение токенизатора




history = parse_log_history(trainer.state.log_history)
history = pd.DataFrame(history[1]) # преобразовываем историю обучения в DataFrame (данные на самом деле про test часть, хоть в них и не указано 'eval_')
with open(f"{RESULTS_DIR}{MODEL_NAME}_finetuned/history.json", mode='w', encoding='utf-8') as file:
    history.to_json(file, orient='records', lines=True, force_ascii=False)

def plot_history(history):
    fig, axes = plt.subplots(ncols=3, figsize=(24,7)) # создаём фигуру с несколькими подграфиками (ncols и/или nrows) размера figsize

    # берём данные об обучении
    epochs = history["Epoch"].values
    loss_train = history["Training Loss"].values # значение loss при обучении
    loss_test = history["Validation Loss"].values # значение loss при валидации
    bleu_test = history["Bleu"].values # значение BLEU при валидации
    len_test = history["Mean Len"].values # среднее количество токенов предсказания при валидации
    best_epoch = bleu_test.argmax() + 1 # эпоха с наилучшей метрикой (+1 из-за нумерации с 0)

    axes[0].plot(epochs, loss_train, color="b", label="При обучении") # построение графика
    axes[0].plot(epochs, loss_test, color="r", label="При валидации") # построение графика
    axes[0].plot(best_epoch, loss_test[best_epoch-1], 'o', color="g", label="Сохранённая модель") # выводим точку, где была найдена лучшая модель
    # axes[0].set_xticklabels(axes[0].get_xticklabels(), ha='right', rotation=45) # добавление rotation для существующих xticks (get_xticklabels) с выравниванием по правому концу (ha='right')
    axes[0].set_title("Изменение значения loss функции в зависимости от эпохи обучения") # название фигуры
    axes[0].set_xlabel("Эпоха") # подпись по оси OX
    axes[0].set_ylabel("Loss") # подпись по оси OY
    axes[0].grid(True) # отображение сетки на графике
    axes[0].legend() # отображение подписей графиков

    axes[1].plot(epochs, bleu_test, color="r", label="При валидации") # построение графика
    axes[1].plot(best_epoch, bleu_test[best_epoch-1], 'o', color="g", label="Сохранённая модель") # выводим точку, где была найдена лучшая модель
    # axes[1].set_xticklabels(axes[1].get_xticklabels(), ha='right', rotation=45) # добавление rotation для существующих xticks (get_xticklabels) с выравниванием по правому концу (ha='right')
    axes[1].set_title("Изменение значения BLEU в зависимости от эпохи обучения") # название фигуры
    axes[1].set_xlabel("Эпоха") # подпись по оси OX
    axes[1].set_ylabel("BLEU") # подпись по оси OY
    axes[1].grid(True) # отображение сетки на графике
    axes[1].legend() # отображение подписей графиков

    axes[2].plot(epochs, len_test, color="r", label="При валидации") # построение графика
    axes[2].plot(best_epoch, len_test[best_epoch-1], 'o', color="g", label="Сохранённая модель") # выводим точку, где была найдена лучшая модель
    # axes[2].set_xticklabels(axes[2].get_xticklabels(), ha='right', rotation=45) # добавление rotation для существующих xticks (get_xticklabels) с выравниванием по правому концу (ha='right')
    axes[2].set_title("Изменение среднего числа токенов предсказания в зависимости от эпохи обучения") # название фигуры
    axes[2].set_xlabel("Эпоха") # подпись по оси OX
    axes[2].set_ylabel("Количество токенов предсказания") # подпись по оси OY
    axes[2].grid(True) # отображение сетки на графике
    axes[2].legend() # отображение подписей графиков

    plt.savefig(f"{RESULTS_DIR}{MODEL_NAME}_finetuned/graph.png", dpi="figure", bbox_inches="tight", facecolor="white") # сохранение графика
    plt.show() # показ фигуры

plot_history(history)