import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
from src.api.model_integration import translate
from src.utils.prompt_templates import get_translation_prompt
from config.config import Config

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer # для работы моделями
# from optimum.executorch import ExecuTorchModelForSeq2SeqLM # ExecuTorch runtime
from optimum.onnxruntime import ORTModelForSeq2SeqLM # ONNX runtime
# os.environ["PATH"] += r";c:\Users\User\anaconda3\envs\gpu\Lib\site-packages\openvino\libs"
from optimum.intel.openvino import OVModelForSeq2SeqLM # openVINO runtime

# изначальные объекты модели и токенизатор не определены
st.session_state["model"] = None
st.session_state["tokenizer"] = None
st.session_state["last_model"] = None
st.session_state["last_runtime"] = None
st.session_state["metrics"] = pd.DataFrame(columns=["Model name", "Runtime", "Tokens count", "Latency", "Color"])

def setup_page():
    """
    Sets up the page with custom styles and page configuration.
    """
    st.set_page_config(
        page_title="Демоверсия приложения-переводчика",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(
        """
        <style>
        :root {
            --color: #4e8cff;
            --color-light: #e6f0ff;
            --color-dark: #1a3a6c;
            --gradient-start: #4e54c8;
            --gradient-end: #8f94fb;
        }
        .stApp {
            margin: auto;
            background-color: var(--background-color);
            color: var(--text-color);
        }
        .logo-container {
            display: flex;
            justify-content: center;
            margin-bottom: 1rem;
        }
        .logo-container img {
            width: 200px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main():
    setup_page()

    # Header section with title and subtitle
    st.markdown(
        """
        <div style="text-align: center;">
            <h1 class="header-title">Приложение-переводчик</h1>
            <p class="header-subtitle">Основанное на применении нейронных сетей</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


    # лого
    st.markdown( 
        fr"""
        <div class="logo-container">
            <img src="https://media.istockphoto.com/id/493553442/ru/%D1%84%D0%BE%D1%82%D0%BE/%D0%B3%D0%BB%D0%BE%D0%B1%D0%B0%D0%BB%D1%8C%D0%BD%D1%8B%D0%B5-%D0%BA%D0%BE%D0%BC%D0%BC%D1%83%D0%BD%D0%B8%D0%BA%D0%B0%D1%86%D0%B8%D0%B8-%D0%BC%D0%B5%D0%B6%D0%B4%D1%83%D0%BD%D0%B0%D1%80%D0%BE%D0%B4%D0%BD%D1%8B%D0%B9-%D0%BE%D0%B1%D0%BC%D0%B5%D0%BD-%D1%81%D0%BE%D0%BE%D0%B1%D1%89%D0%B5%D0%BD%D0%B8%D1%8F%D0%BC%D0%B8-%D0%B8-%D0%BF%D0%B5%D1%80%D0%B5%D0%B2%D0%BE%D0%B4%D0%B0-%D0%BA%D0%BE%D0%BD%D1%86%D0%B5%D0%BF%D1%86%D0%B8%D0%B8.jpg?s=612x612&w=0&k=20&c=Hv3FB7JqEWM_eCdKdzagbeWFf3V2NUdWKkAA2R0JH-o=" alt="Logo">
        </div>
        """,
        unsafe_allow_html=True,
    )


    with st.sidebar: # боковая панель для настроек
        st.title("⚙️Настройки")
        print(f"Модель: {st.session_state['last_model']}, рантайм: {st.session_state['last_runtime']}")


        last_model = st.session_state["last_model"]
        last_runtime = st.session_state["last_runtime"]

        model_name = st.selectbox("Выбор модели", Config.AVAILABLE_MODELS)
        model_path = Config.REAL_MODELS_NAMES[model_name]
        if last_model != model_name: # проверка, что нужная модель ещё не загружена
            if not os.path.exists(Config.MODELS_DIR + model_path):
                print(f"Скачиваю и сохраняю модель {model_name}...")
                st.session_state["model"] = AutoModelForSeq2SeqLM.from_pretrained(model_path)
                st.session_state["tokenizer"] = AutoTokenizer.from_pretrained(model_path)

                st.session_state["model"].save_pretrained(Config.MODELS_DIR + model_path, from_pt=True) # сохранение модели
                st.session_state["tokenizer"].save_pretrained(Config.MODELS_DIR + model_path) # сохранение токенизатора
            else:
                print(f"Загружаю сохранённую модель {model_name}...")
                st.session_state["model"] = AutoModelForSeq2SeqLM.from_pretrained(Config.MODELS_DIR + model_path)
                st.session_state["tokenizer"] = AutoTokenizer.from_pretrained(Config.MODELS_DIR + model_path)
            last_model = model_name
            last_runtime = "PyTorch"
            
        runtime = st.selectbox("Выбор рантайма", Config.AVAILABLE_RUNTIMES)
        if last_runtime != runtime:
            print(f"Конвертирую модель из рантайма {last_runtime} в {runtime}...")

            if runtime == "PyTorch":
                print(f"Найдена уже сохранённая вариация модели в рантайме {runtime}...")
                st.session_state["model"] = AutoModelForSeq2SeqLM.from_pretrained(Config.MODELS_DIR + model_path)
            # elif runtime == "ExecuTorch":
            #     st.session_state["model"] = ExecuTorchModelForSeq2SeqLM.from_pretrained(Config.MODELS_DIR + model_path, export=True, recipe="xnnpack", attn_implementation="custom_sdpa") # export — для моделей трансформеров с версии optimum>=2.0, from_transformers — для optimum<2.0
            elif runtime == "ONNX":
                if os.path.exists(Config.MODELS_DIR + model_path + "_ONNX"):
                    print(f"Найдена уже сохранённая вариация модели в рантайме {runtime}...")
                    st.session_state["model"] = ORTModelForSeq2SeqLM.from_pretrained(Config.MODELS_DIR + model_path + "_ONNX") # export — для моделей трансформеров с версии optimum>=2.0, from_transformers — для optimum<2.0
                else:
                    st.session_state["model"] = ORTModelForSeq2SeqLM.from_pretrained(Config.MODELS_DIR + model_path, export=True) # export — для моделей трансформеров с версии optimum>=2.0, from_transformers — для optimum<2.0
                    st.session_state["model"].save_pretrained(Config.MODELS_DIR + model_path + "_ONNX")
            elif runtime == "openVINO":
                if os.path.exists(Config.MODELS_DIR + model_path + "_openVINO"):
                    print(f"Найдена уже сохранённая вариация модели в рантайме {runtime}...")
                    st.session_state["model"] = OVModelForSeq2SeqLM.from_pretrained(Config.MODELS_DIR + model_path + "_openVINO") # export — для моделей трансформеров с версии optimum>=2.0, from_transformers — для optimum<2.0
                else:
                    st.session_state["model"] = OVModelForSeq2SeqLM.from_pretrained(Config.MODELS_DIR + model_path, export=True) # export — для моделей трансформеров с версии optimum>=2.0, from_transformers — для optimum<2.0
                    st.session_state["model"].save_pretrained(Config.MODELS_DIR + model_path + "_openVINO")

            last_runtime = runtime # обновляем последний рантайм


        source_lang = st.selectbox("Язык оригинала", Config.SOURCE_LANG)
        target_lang = st.selectbox("Целевой язык", Config.TARGET_LANG)

        # Обновляем имена используемых модели и рантайма
        st.session_state["last_model"] = last_model
        st.session_state["last_runtime"] = last_runtime


    main_container = st.container(border=True) # основной контейнер для текста и перевода

    with main_container:
        st.header("Введите текст для перевода")
        text = st.text_area(
            "Оригинальный текст",
            "Цель разработки — предоставить пользователям личного синхронного переводчика.",
            height=200,
        )
        st.caption(f"Количество символов: {len(text)}")

        if st.button("Перевести", type="primary"):
            if text:
                tab1, tab2 = st.tabs(
                    [
                        "Перевод",
                        "Анализ эффективности"
                    ]
                )

                translation_prompt = get_translation_prompt(text, model_name, source_lang, target_lang)
                translation, latency, tokens_count = translate(translation_prompt, st.session_state["model"], st.session_state["tokenizer"])
                
                last_model = st.session_state["last_model"]
                last_runtime = st.session_state["last_runtime"]
                records_count = len(st.session_state["metrics"]) # количество уже сделанных переводов
                
                new_row = {"Model name": last_model, "Runtime": last_runtime, "Tokens count": tokens_count, "Latency": latency, "Color": colors[last_runtime]} # новая запись про метрики
                st.session_state["metrics"].loc[records_count] = new_row # запоминаем метрики
                metrics = st.session_state["metrics"]


                # Tab 1: Translation
                with tab1:
                    st.subheader("Перевод")
                    translation_container = st.empty()

                    if isinstance(translation, str):
                        translation_container.markdown(translation)
                    else:
                        error_message = f"Error: {str(translation)}"
                        translation_container.error(error_message)


                # Tab 2: Analytics
                with tab2:
                    st.subheader("Анализ эффективности")
                    container_metrics = st.container(border=False) # placeholder (контейнер) для данных с фиксированной позицией (и рамкой border=True)

                    holder_approx = st.empty() # placeholder (контейнер) для данных с фиксированной позицией (и рамкой border=True)

                    if isinstance(translation, str):
                        columns_models = container_metrics.columns(spec=len(metrics["Model name"].unique())) # делим контейнер на spec колонок (или в пропорции spec)
                        holders_models = {}


                        approximations = {}
                        for i, column in enumerate(columns_models):
                            model_name_ = metrics["Model name"].unique()[i]
                            column.title(f"Модель {model_name_}") # добавление заголовка к контейнеру
                            holders_models[model_name_] = column.empty() # пустой подконтейнер, что может вмещать только один объект (для самообновления графиков)

                            plt.figure(figsize=(10, 5))
                            plt.ylim(0, metrics["Latency"].max() + 1)
                            data = metrics[metrics["Model name"] == model_name_]

                            for runtime_name_ in data["Runtime"].unique():
                                data_ = data[data["Runtime"] == runtime_name_]
                                a, b = np.polyfit(data_["Tokens count"], data_["Latency"], deg=1) # считаем линейную аппроксимацию (deg=1)
                                approximations[f"{model_name_} - {runtime_name_}"] = [a, b]
                        
                            sns.boxplot(data=data, x="Tokens count", y="Latency", hue="Runtime", palette=colors, native_scale=True, showfliers=False, width=0.5) # строим "ящики с усами", showfliers — отображать ли выбросы, native_scale — воспринимать ли ось X как непрерывную (а не категориальную)

                            plt.xticks(rotation=45, ha='right') # поворот на 45 градусов подписей под осью OX (ha='right' ~ правый конец соответствует колонке)
                            plt.title("Зависимость времени перевода от размера текста") # название фигуры
                            plt.xlabel("Количество токенов") # подпись по оси x
                            plt.ylabel("Задержка перевода (latency, sec)") # подпись по оси y
                            holders_models[model_name_].pyplot(plt) # создаём линейный график на основе данных в словаре session_state
                            plt.close() # отключает повторное отображение графика из-за наложения их в Matplotlib

                        
                        plt.figure(figsize=(25,10)) # задание размера фигуры
                        x = np.linspace(1, 256)
                        for variation in approximations.keys():
                            a, b = approximations[variation]
                            y = a * x + b
                            sns.lineplot(x=x, y=y, label=variation, linewidth=2.0)
                        plt.axis([10, 256, 0, 20])
                        plt.xticks(rotation=45, ha='right') # поворот на 45 градусов подписей под осью OX (ha='right' ~ правый конец соответствует колонке)
                        plt.title("Зависимость времени перевода от количества токенов") # название фигуры
                        plt.xlabel("Количество токенов") # подпись по оси x
                        plt.ylabel("Задержка перевода (latency, sec)") # подпись по оси y
                        holder_approx.pyplot(plt) # создаём линейный график на основе данных в словаре session_state
                        plt.close() # отключает повторное отображение графика из-за наложения их в Matplotlib
                    
                    else:
                        error_message = f"Error: {str(translation)}"
                        container_metrics.error(error_message)


    with st.sidebar: # боковая панель для дополнительной информации
        st.subheader("Информация")
        st.info("Это приложение является демонстрационной версией основных возможностей.")


# colors = {"PyTorch": "blue", "ONNX": "orange", "openVINO": "green"}
colors = {"PyTorch": sns.color_palette()[0], "ONNX": sns.color_palette()[1], "openVINO": sns.color_palette()[2]}

if __name__ == "__main__":
    main()
