# Машинный перевод в режиме реального времени
Дипломная работа на тему "Real-Time Machine Translation". 
* В ней исследуются  ***LLM на основе трансформеров***, что могут работать в real-time на CPU (T5, mT5, MBart, M2M100, Marian, LlaMA3.2). 
* Проводится ***fine-tuning*** модели T5 на датасете с Hugging Face.
* Анализируются различные варианты ***прунинга*** модели с последующим повторным дообучением для возвращения потерянных метрик BLEU, Latency.
* Рассматривается ***инференс модели в нескольких рантаймах*** (PyTorch, ExecuTorch, ONNX, openVINO).
* Производится общее сравнение всех полученных результатов.

**Используемые языки:**
* Python
* Bash


## Project Structure:
    .
    ├── data                                    # общая папка под данные
    │   ├── polynews-parallel                   # изначальный датасет с Hugging Face
    │   │   └── ...
    │   ├── polynews-parallel_t5_processed      # датасет, предобработанный для T5 модели
    │   │   └── ...
    │   └── README.md                           # файл, содержащий основную информацию о датасетах
    ├── GUI                                     # подпроект, отвечающий за приложение для взаимодействия пользователя с моделями
    │   └── ...
    ├── images                                  # папка под изображения
    │   ├── data_folder.png               
    │   └── models_folder.png             
    ├── models                                  # папка под модели
    │   ├── <model_name>                        # папка под конкретную модель "model_name"
    │   │   └── ...
    │   ├── ...
    │   └── README.md                           # файл, содержащий основную информацию о моделях
    ├── results                                 # папка под результаты исследования
    │   ├── <model_name>                        # папка под результаты конкретной модели "model_name"
    │   │   └── ...
    │   └── ...
    ├── src
    │   ├── node_exe                            # папка под bash-скрипты для запуска вычислений на кластере
    │   │   ├── nn_finetuning.sbatch            # bash-скрипт для обучения
    │   │   ├── nn_metric_calc.sbatch           # bash-скрипт для подсчёта метрик
    │   │   ├── nn_pruning.sbatch               # bash-скрипт для неструктурированного прунинга
    │   │   └── nn_pruning_structured.sbatch    # bash-скрипт для структурированного прунинга
    │   ├── notebooks                           # папка под Jupyter notebooks с подсчётами метрик/простыми вычислениями
    │   │   ├── baseline                        # папка с notebook_ами для замера изначальных метрик моделей T5, mT5, MBart, M2M100, Marian, LlaMA3.2
    │   │   │   └── ...
    │   │   ├── finetuning                      # папка с notebook_ами для замера метрик слегка дообученных моделей T5, mT5, Marian
    │   │   │   └── ...
    │   │   ├── metrics.ipynb                   # notebook для сравнения работы изначальных моделей и вариантов прунинга T5
    │   │   └── runtimes.ipynb                  # notebook для сравнения работы модели T5 в разных рантаймах
    │   ├── t5_finetuning.py                    # python скрипт для обучения T5 моделей
    │   ├── t5_metric_calc.py                   # python скрипт для подсчёта метрик
    │   ├── t5_pruning.py                       # python скрипт для неструктурированного прунинга T5 моделей
    │   └── t5_pruning_structured.py            # python скрипт для структурированного прунинга T5 моделей
    ├── .env                                    # файл, содержащий особые переменные окружения
    ├── requirements.txt                        # файл со списком необходимых библиотек для работы проекта
    ├── diploma.pdf                             # файл с текстом дипломной работы
    ├── presentation.pptx                       # файл с презентацией дипломной работы
    └── README.md                               # файл, содержащий основную информацию о проекте


## Requirements:
Файл `requirements.txt` содержит необходимые библиотеки Python для запуска вложенных файлов.

Они могут быть установлены следующей командой:
```
pip install --user -r requirements.txt
```

Либо с созданием окружения в **Conda**:
```
conda create --name <env_name> python=3.9.12
conda activate <env_name>
pip install -r requirements.txt
```


## Setup:
Перед запуском любого скрипта, как через `python ./src/<script_name>.py` , так и через `sbatch ./src/node_exe/<script_name>.sbatch`, нужно провести его настройку в файле `.env`.

Для запуска вычислений на класете используются bash-скрипты из папки `./src/node_exe/`:
```
sbatch ./src/node_exe/<script_name>.sbatch
```
Например, следующий скрипт добавит дообучение модели (описанной в файле `.env`) в очередь на кластере:
```
sbatch ./src/node_exe/nn_finetuning.sbatch
```
##

Для обычного запуска скриптов (из папки `./src/`) можно воспользоваться командой:
```
python ./src/<script_name>.py
```
Например, для дообучения модели (описанной в файле `.env`):
```
python ./src/t5_finetuning.py
```
##

При работе на суперкомпьютере (кластере):
* srun - позволяет выполнить задачу в интерактивном режиме, обычно используется для отладки
* squeue - отобразить все задачи в очереди.
* mj - отобразить состояние только своих задач.
* mj --start - отобразить ориентировочное время запуска своих ожидающих задач.
* scancel <job_id> - прекратить выполнение своей задачи с указанным id.

Для отключения от кластера в VS Code (команду вводить в терминале виртуальной машины):
```
pkill -u $USER -f vscode-server
```
##

Для запуска GUI из соответствующей папки `./GUI/` выполнить команду:
```
python -m streamlit run ./main.py
```


## Troubleshooting
* Ошибка `ImportError: DLL load failed while importing _ssl: Не найден указанный модуль.` связана с отсутствием библиотек `libcrypto-1_1-x64.dll` и `libssl-1_1-x64.dll` в переменной окружения `PATH`. Исправляется добавлением пути до папки с ними в `PATH`.

    Например, на Linux (или git-bash):
    ```
    export PATH=$PATH:/c/Users/<user_name>/anaconda3/envs/<env_name>/Library/bin
    ```

    На Windows (PowerShell):
    ```
    $Env:PATH += ";C:\Users\<user_name>\anaconda3\envs\<env_name>\Library\bin"
    ```

* Ошибка `ModuleNotFoundError: No module named 'optimum.intel'` связана с расположением модуля openvino. Исправляется добавлением пути в `PATH`.

    Например, на Linux (или git-bash):
    ```
    export PATH=$PATH:/c/Users/<user_name>/anaconda3/envs/<env_name>/Lib/site-packages/openvino/libs
    ```

    На Windows (PowerShell):
    ```
    $Env:PATH += ";C:\Users\<user_name>\anaconda3\envs\<env_name>\Lib\site-packages\openvino\libs"
    ```