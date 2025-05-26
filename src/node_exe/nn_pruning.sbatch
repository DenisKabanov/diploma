#!/bin/bash
#SBATCH --job-name=nn_pruning               # Название задачи
#SBATCH --error=nn_pruning%j.err            # Файл для вывода ошибок
#SBATCH --output=nn_pruning%j.log           # Лог-файл
#SBATCH --time=0-1:0                        # Максимальное время выполнения
#SBATCH --cpus-per-task 4                   # Количество выделенных ядер CPU
#SBATCH --nodes 1                           # Все CPU и GPU будут на одном вычислительном узле
#SBATCH --gpus 0                            # Требуемое кол-во GPU
#SBATCH --mail-user dskabanov@edu.hse.ru    # Почта, на которую будет отправлена информация о запуске

module load CUDA/11.4 Python/Anaconda       # Загрузка модулей

conda activate nn_finetuning

python ./src/t5_pruning.py