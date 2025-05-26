#!/bin/bash
#SBATCH --job-name=nn_metric_calc           # Название задачи
#SBATCH --error=nn_metric_calc%j.err        # Файл для вывода ошибок
#SBATCH --output=nn_metric_calc%j.log       # Лог-файл
#SBATCH --time=1-12:0                       # Максимальное время выполнения
#SBATCH --constraint="type_a"
#SBATCH --cpus-per-task 4                   # Количество выделенных ядер CPU
#SBATCH --nodes 1                           # Все CPU и GPU будут на одном вычислительном узле
#SBATCH --gpus 0                            # Требуемое кол-во GPU
#SBATCH --mail-user dskabanov@edu.hse.ru    # Почта, на которую будет отправлена информация о запуске

module load CUDA/11.4 Python/Anaconda       # Загрузка модулей

conda activate nn_finetuning

python ./t5_metric_calc.py