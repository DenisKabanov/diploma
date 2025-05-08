import os # для взаимодействия с системой
from dotenv import load_dotenv # для загрузки переменных окружения

import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from torch.nn.utils import prune




load_dotenv() # загрузка переменных окружения
RESULTS_DIR = os.getenv("RESULTS_DIR")
MODELS_DIR = os.getenv("MODELS_DIR")
MODEL_NAME = os.getenv("MODEL_NAME")

MAX_SEQUENCE_LEN = int(os.getenv("MAX_SEQUENCE_LEN"))
PRUNE_AMOUNT = float(os.getenv("PRUNE_AMOUNT"))

DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

print(f"Status: \n\
      RESULTS_DIR: {RESULTS_DIR}\n\
      MODELS_DIR: {MODELS_DIR}\n\
      MODEL_NAME: {MODEL_NAME}\n\
      MAX_SEQUENCE_LEN: {MAX_SEQUENCE_LEN}\n\
      PRUNE_AMOUNT: {PRUNE_AMOUNT}\n\
      DEVICE: {DEVICE}\n\
      ")




model = T5ForConditionalGeneration.from_pretrained(MODELS_DIR + MODEL_NAME)
tokenizer = T5Tokenizer.from_pretrained(MODELS_DIR + MODEL_NAME)




params_all = sum(p.numel() for p in model.parameters())
params_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Число обучаемых параметров: {params_trainable} из {params_all}, то есть ~{params_trainable / params_all * 100:.2f}%.") # считаем число параметров

size_model = 0
for param in model.parameters():
    if param.data.is_floating_point():
        size_model += param.numel() * torch.finfo(param.data.dtype).bits
    else:
        size_model += param.numel() * torch.iinfo(param.data.dtype).bits
print(f"Вес модели: {size_model / 8e6:.2f} / MB")




def prune_model(model, pruning_amount=0.2): # pruning_amount — Controls the fraction of weights to prune (e.g., 0.2 means 20% of the weights will be pruned).
    print(f"Прунинг {pruning_amount}% параметров модели.")
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            prune.l1_unstructured(module, name='weight', amount=pruning_amount)
            # prune.random_unstructured(module, name="weight", amount=pruning_amount)
            # прунинг создаёт weight_orig и weight_mask параметры, в которые складирует оригинальные веса и максу из нулей и единиц прунинга, комбинация же самой маски и изначальных весов записывается в атрибут weight (не настоящие веса до вызова prune.remove)
            
            prune.remove(module, 'weight') # удаляем атрибут, а не параметр ~ записываем веса в соответствии с прунингом

prune_model(model, pruning_amount=PRUNE_AMOUNT)




prefix = "translate to en: "
src_text = prefix + "Цель разработки — предоставить пользователям личного синхронного переводчика."

tokens_encoded = tokenizer(src_text, max_length=MAX_SEQUENCE_LEN, return_tensors="pt", truncation=True, padding=True) # токенизируем данные (max_length — максимальное число токенов в документе, return_tensors — тип возвращаемых данных, np для numpy.array, pt для torch.tensor; truncation и padding — обрезание лишних токенов и автозаполнение недостающих до max_length)

tokens_generated = model.generate(**tokens_encoded)

# output = tokenizer.batch_decode(tokens_generated, skip_special_tokens=True)
output = tokenizer.batch_decode(tokens_generated, skip_special_tokens=False)
print(output)




if not os.path.exists(MODELS_DIR + f"t5_pruned_{PRUNE_AMOUNT}"):
    print("Сохраняю модель...")
    model.save_pretrained(MODELS_DIR + f"t5_pruned_{PRUNE_AMOUNT}", from_pt=True) # сохранение модели
    tokenizer.save_pretrained(MODELS_DIR + f"t5_pruned_{PRUNE_AMOUNT}") # сохранение токенизатора
else:
    print(f"Модель по пути {MODELS_DIR + f't5_pruned_{PRUNE_AMOUNT}'} уже была сохранена ранее!")
