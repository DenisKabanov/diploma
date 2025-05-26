# GUI для моделей-переводчиков

Часть проекта, отвечающая за GUI приложения для взаимодействия с пользователем.

## Project Structure:
```
GUI
├── config
│   ├── __init__.py
│   └── config.py                  # Конфигурации для запуска моделей
├── images                         # Папка под изображения
│   └── translator.png             
├── src
│   ├── api
│   │   ├── __init__.py
│   │   └── model_integration.py   # Обработка запросов к моделям
│   │
│   ├── utils
│   │   ├── __init__.py
│   │   └── prompt_templates.py    # Промпты для моделей
│   ├── __init__.py
│   └── app.py                     # Основная логика Streamlit приложения 
├── .env                           # Файл с переменными окружения
├── main.py                        # файл для запуска графического приложения
└── README.md                      # файл, содержащий основную информацию о проекте
```

## Setup:
Перед запуском рекомендуется проверить файл конфигурации `GUI/config/config.py` и `.env`.

Запуск GUI из одноимённой папки:
```bash
streamlit run main.py
```

## Examples:
![Пример GUI](./images/example.png)