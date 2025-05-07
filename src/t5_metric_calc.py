import os # для взаимодействия с системой
import numpy as np # для работы с массивами
import pandas as pd # для удобной работы с датасетом
import random as random # для работы со случайностью
from dotenv import load_dotenv # для загрузки переменных окружения
import json # для сохранения и загрузки объектов
from tqdm.auto import tqdm # для отслеживания прогресса

from datasets import Dataset, load_dataset, load_from_disk # для работы с HuggingFace датасетами

import torch # для работы с моделями torch
from transformers import T5ForConditionalGeneration, T5Tokenizer # для работы с T5 моделью

import time # для отслеживания времени выполнения
import matplotlib.pyplot as plt # для построения графиков
import seaborn as sns # для построения красивых графиков
import evaluate # для подсчёта метрик




load_dotenv() # загрузка переменных окружения
DATA_DIR = os.getenv("DATA_DIR")
RESULTS_DIR = os.getenv("RESULTS_DIR")
MODELS_DIR = os.getenv("MODELS_DIR")
MODEL_NAME = os.getenv("MODEL_NAME")
DATASET_NAME_HF = os.getenv("DATASET_NAME_HF")
DATASET_NAME_LOC = os.getenv("DATASET_NAME_LOC")

MAX_SEQUENCE_LEN = int(os.getenv("MAX_SEQUENCE_LEN"))

RANDOM_STATE = int(os.getenv("RANDOM_STATE"))
TEST_SIZE = float(os.getenv("TEST_SIZE"))
TEST_MAX_SAMPLES = int(os.getenv("TEST_MAX_SAMPLES"))
TRAIN_MAX_SAMPLES = int(os.getenv("TRAIN_MAX_SAMPLES"))

DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

print(f"Status: \n\
      DATA_DIR: {DATA_DIR}\n\
      RESULTS_DIR: {RESULTS_DIR}\n\
      MODELS_DIR: {MODELS_DIR}\n\
      MODEL_NAME: {MODEL_NAME}\n\
      DATASET_NAME_HF: {DATASET_NAME_HF}\n\
      DATASET_NAME_LOC: {DATASET_NAME_LOC}\n\
      MAX_SEQUENCE_LEN: {MAX_SEQUENCE_LEN}\n\
      RANDOM_STATE: {RANDOM_STATE}\n\
      TEST_SIZE: {TEST_SIZE}\n\
      TEST_MAX_SAMPLES: {TEST_MAX_SAMPLES}\n\
      TRAIN_MAX_SAMPLES: {TRAIN_MAX_SAMPLES}\n\
      DEVICE: {DEVICE}\n\
      ")




model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)

if not os.path.exists(RESULTS_DIR + MODEL_NAME):
    print("Сохраняю модель...")
    model.save_pretrained(MODELS_DIR + MODEL_NAME, from_pt=True) # сохранение модели
    tokenizer.save_pretrained(MODELS_DIR + MODEL_NAME) # сохранение токенизатора
else:
    print(f"Модель по пути {RESULTS_DIR + MODEL_NAME} уже была сохранена ранее!")



dataset = load_dataset(DATASET_NAME_HF, name="eng_Latn-rus_Cyrl") # скачивание датасета, name — название subset_а с HuggingFace
dataset.save_to_disk(DATA_DIR + DATASET_NAME_LOC) # локальное сохранение датасета (в формате arrow)

def preprocess_function(data: Dataset, random_state=RANDOM_STATE):
    random.seed(random_state) # Set the random number generator to a fixed sequence.
    samples_count = len(dataset["train"]) # общее число сэмплов в датасете

    reflected_idx = set(random.sample(range(0, samples_count), int(samples_count/2))) # индексы отражаемых сэмплов (set — для сортировки и удобного вычитания)
    regular_idx = set(range(0, samples_count)) - reflected_idx

    data["new_src"] = ["translate to ru: " + sample if idx in regular_idx else "translate to en: " + data["tgt"][idx] for idx, sample in enumerate(data["src"])]
    data["new_tgt"] = [sample if idx in regular_idx else data["src"][idx] for idx, sample in enumerate(data["tgt"])]
    model_inputs = tokenizer(data["new_src"], text_target=data["new_tgt"], max_length=MAX_SEQUENCE_LEN, return_tensors="pt", truncation=True, padding=True)
    return model_inputs

dataset = dataset.map(preprocess_function, batched=True)
dataset = dataset.remove_columns(["provenance", "src", "tgt"]) # удаление ненужной колонки
dataset = dataset.rename_column("new_src", "src") # переименовываем колонку
dataset = dataset.rename_column("new_tgt", "tgt") # переименовываем колонку
dataset = dataset["train"].train_test_split(test_size=TEST_SIZE, shuffle=True, seed=RANDOM_STATE) # разбиение датасета на тестовую и обучающую выборки

if not os.path.exists(DATA_DIR + DATASET_NAME_LOC + "_t5_processed"):
    print("Сохраняю обработанный датасет...")
    dataset.save_to_disk(DATA_DIR + DATASET_NAME_LOC + "_t5_processed") # локальное сохранение датасета (в формате arrow)
else:
    print(f"Датасет по пути {DATA_DIR + DATASET_NAME_LOC + '_t5_processed'} уже был сохранён ранее!")

dataset["train"] = dataset["train"].select(range(TRAIN_MAX_SAMPLES))
dataset["test"] = dataset["test"].select(range(TEST_MAX_SAMPLES))




vocab_len = tokenizer.vocab_size # размер словаря
print(f"Размер словаря токенизатора: {vocab_len}, с учётом особых токенов: {len(tokenizer.get_vocab())}.")

word2id = {} # словарь соответствия слова его id_шнику
id2word = {} # словарь соответствия id_шника слову
for word, id in tokenizer.get_vocab().items(): # идём по словам в словаре
    word2id[word] = id # заполняем словарь соответствия слова его id_шнику
    id2word[id] = word # заполняем словарь соответствия id_шника слову




def translate(model, tokenizer, texts) -> tuple:
    tokens_count = [] # список под количество токенов в тексте
    latency = [] # список под величину задержки между запуском модели и выводом ответа
    translations = [] # список под переводы

    if isinstance(texts, str): # если пришёл объект типа строки
        texts = [texts] # делаем из объекта список с одним элементом
    elif isinstance(texts, Dataset) or isinstance(texts, dict): # если пришёл объект типа Dataset или словарь (полученный с помощью среза объекта Dataset)
        texts = texts["src"] # берём из него только текста, что нужно переводить

    model.eval() # перевод модели в режим оценивания (dropout перестаёт работать, а BatchNorm собирать статистику)

    with torch.no_grad(): # отключаем подсчёт градиентов
        #================================ Быстрый перевод батчами ==========================================
        # tokens_encoded = tokenizer(texts, max_length=tokenizer.max_len_single_sentence, return_tensors="pt", truncation=True, padding=True) # токенизируем данные (max_length — максимальное число токенов в документе, return_tensors — тип возвращаемых данных, np для numpy.array, pt для torch.tensor; truncation и padding — обрезание лишних токенов и автозаполнение недостающих до max_length)
        # tokens_generated = model.generate(**tokens_encoded)
        # translations = tokenizer.batch_decode(tokens_generated, skip_special_tokens=True) # декодирование последовательности токенов (аналог .decode, но для работы с несколькими последовательностями сразу), skip_special_tokens — выводить ли специальные токены
        #-------------------------------- Перевод для учёта времени работы ---------------------------------
        for text in tqdm(texts):
            time_start = time.time() # замеряем время начала работы  с моделью

            tokens_encoded = tokenizer(text, max_length=tokenizer.max_len_single_sentence, return_tensors="pt", truncation=True, padding=True) # токенизируем данные (max_length — максимальное число токенов в документе, return_tensors — тип возвращаемых данных, np для numpy.array, pt для torch.tensor; truncation и padding — обрезание лишних токенов и автозаполнение недостающих до max_length)
            tokens_count.append(tokens_encoded["input_ids"].shape[1]) # запоминаем количество токенов

            tokens_generated = model.generate(**tokens_encoded) # генерируем новую последовательность токенов (переводим текст)
            translations.append(tokenizer.decode(tokens_generated[0], skip_special_tokens=True)) # декодирование последовательности токенов, skip_special_tokens — выводить ли специальные токены

            latency.append(time.time()  - time_start)
        #---------------------------------------------------------------------------------------------------
    return tokens_count, latency, translations


tokens_count, latency, translations = translate(model, tokenizer, dataset["test"])

metric_BLEU = evaluate.load("bleu") # загружаем метрику

def compute_metrics(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels] # создаём список возможных переводов

    metric = metric_BLEU.compute(predictions=preds, references=labels)["bleu"]
    print(f"Значение метрики BLEU: {metric}")

bleu = compute_metrics(translations, dataset["test"]["tgt"])

if not os.path.exists(RESULTS_DIR + MODEL_NAME):
    os.makedirs(RESULTS_DIR + MODEL_NAME)

with open(RESULTS_DIR + MODEL_NAME + "/tokens_count.json", mode='w', encoding='utf-8') as f: # открываем файл для записи (w — не побитовой)
    json.dump(tokens_count, f, ensure_ascii=False, indent=4) # сохраняем объект в файл f
with open(RESULTS_DIR + MODEL_NAME + "/latency.json", mode='w', encoding='utf-8') as f: # открываем файл для записи (w — не побитовой)
    json.dump(latency, f, ensure_ascii=False, indent=4) # сохраняем объект в файл f
with open(RESULTS_DIR + MODEL_NAME + "/translations.json", mode='w', encoding='utf-8') as f: # открываем файл для записи (w — не побитовой)
    json.dump(translations, f, ensure_ascii=False, indent=4) # сохраняем объект в файл f
with open(RESULTS_DIR + MODEL_NAME + "/BLEU.json", mode='w', encoding='utf-8') as f: # открываем файл для записи (w — не побитовой)
    json.dump(bleu, f, ensure_ascii=False, indent=4) # сохраняем объект в файл f


results = pd.DataFrame({"Tokens count": tokens_count, "Latency": latency}) # собираем данные в DataFrame
results.sort_values(by=["Tokens count"], inplace=True)
results.reset_index(drop=True, inplace=True) # обновляем индексы, так как они остались от предыдущего варианта датасета (inplace=True - перезаписываем существующий датасет)

stats = results.groupby(by="Tokens count", as_index=True).agg(mean=("Latency", "mean"),
                                                              std=("Latency", "std")
                                                             )

outlier_indexes = []

for index in results.index:
    tokens_count_, latency_ = results.loc[index]
    if np.abs(latency_ - stats.loc[tokens_count_]["mean"]) > 3 * stats.loc[tokens_count_]["std"]:
        outlier_indexes.append(index)

results.drop(outlier_indexes, inplace=True)

stats = results.groupby(by="Tokens count", as_index=True).agg(mean=("Latency", "mean"),
                                                              std=("Latency", "std")
                                                             )

a, b = np.polyfit(stats.index[:75], stats["mean"][:75], deg=1) # считаем линейную аппроксимацию (deg=1), [:75] — так как после идёт большой разброс по времени из-за малой представленности в датасете
print(f"Затрачиваемое время увеличивается, в среднем, на {a:.5f} секунд за каждый новый токен, при этом модель работает не менее {b:.5f} секунд.")




plt.figure(figsize=(25,10)) # задание размера фигуры
sns.boxplot(x=results["Tokens count"], y=results["Latency"], native_scale=True, showfliers=False) # строим "ящики с усами", showfliers — отображать ли выбросы, native_scale — воспринимать ли ось X как непрерывную (а не категориальную)

x = np.linspace(stats.index[0], stats.index[-1])
# x = stats.index.to_numpy() # рассмотренные значения по оси x для линейного графика
y = a * x + b
sns.lineplot(x=x, y=y, color="red", label="Аппроксимация")
# sns.pointplot(x=x, y=y, color="red", label="Аппроксимация") # pointplot в данном случае идёт как аналог линейного графика, у которого ось X воспринимается "категориально", то есть так же, как в boxplot

plt.xticks(rotation=45, ha='right') # поворот на 45 градусов подписей под осью OX (ha='right' ~ правый конец соответствует колонке)
plt.title("Зависимость времени перевода от размера текста") # название фигуры
plt.xlabel("Количество токенов") # подпись по оси x
plt.ylabel("Задержка перевода (latency, sec)") # подпись по оси y
plt.legend() # отображение подписей графиков
plt.savefig(f"{RESULTS_DIR}{MODEL_NAME}/latency_graph.png", dpi="figure", bbox_inches=None) # сохранение графика
plt.show() # показ фигуры