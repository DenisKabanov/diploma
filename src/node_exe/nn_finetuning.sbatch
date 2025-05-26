#!/bin/bash
#SBATCH --job-name=nn_finetuning            # Название задачи
#SBATCH --error=nn_finetuning%j.err         # Файл для вывода ошибок
#SBATCH --output=nn_finetuning%j.log        # Лог-файл
#SBATCH --time=3-12:0                       # Максимальное время выполнения
#SBATCH --cpus-per-task 1                   # Количество выделенных ядер CPU
#SBATCH --nodes 1                           # Все CPU и GPU будут на одном вычислительном узле
#SBATCH --gpus 1                            # Требуемое кол-во GPU
#SBATCH --mail-user dskabanov@edu.hse.ru    # Почта, на которую будет отправлена информация о запуске

module load CUDA/11.4 Python/Anaconda       # Загрузка модулей

conda activate nn_finetuning

python ../t5_finetuning.py