import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from src.api.model_integration import translate
from src.utils.prompt_templates import get_translation_prompt
from config.config import Config

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer # для работы моделями

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
    global model, tokenizer
    setup_page()

    for key in ["tokens_count", "latency"]:
        if key not in st.session_state.keys(): # если ключа нет в словаре сессии
            st.session_state[key] = [] # создаём пустой список под данные


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


    # logo
    st.markdown(
        fr"""
        <div class="logo-container">
            <img src="https://media.istockphoto.com/id/493553442/ru/%D1%84%D0%BE%D1%82%D0%BE/%D0%B3%D0%BB%D0%BE%D0%B1%D0%B0%D0%BB%D1%8C%D0%BD%D1%8B%D0%B5-%D0%BA%D0%BE%D0%BC%D0%BC%D1%83%D0%BD%D0%B8%D0%BA%D0%B0%D1%86%D0%B8%D0%B8-%D0%BC%D0%B5%D0%B6%D0%B4%D1%83%D0%BD%D0%B0%D1%80%D0%BE%D0%B4%D0%BD%D1%8B%D0%B9-%D0%BE%D0%B1%D0%BC%D0%B5%D0%BD-%D1%81%D0%BE%D0%BE%D0%B1%D1%89%D0%B5%D0%BD%D0%B8%D1%8F%D0%BC%D0%B8-%D0%B8-%D0%BF%D0%B5%D1%80%D0%B5%D0%B2%D0%BE%D0%B4%D0%B0-%D0%BA%D0%BE%D0%BD%D1%86%D0%B5%D0%BF%D1%86%D0%B8%D0%B8.jpg?s=612x612&w=0&k=20&c=Hv3FB7JqEWM_eCdKdzagbeWFf3V2NUdWKkAA2R0JH-o=" alt="Logo">
        </div>
        """,
        unsafe_allow_html=True,
    )


    # Sidebar for settings
    with st.sidebar:
        st.title("⚙️Настройки")
        model_name = st.selectbox("Выбор модели", Config.AVAILABLE_MODELS)
        model_ckeckpoint = Config.REAL_MODELS_NAMES[model_name]

        if (not hasattr(model, "config")) or (not model.config._name_or_path.endswith(model_ckeckpoint)): # проверка, что нужная модель ещё не загружена
            model = AutoModelForSeq2SeqLM.from_pretrained(Config.MODELS_DIR + model_ckeckpoint)
            tokenizer = AutoTokenizer.from_pretrained(Config.MODELS_DIR + model_ckeckpoint)

        source_lang = st.selectbox("Язык оригинала", Config.SOURCE_LANG)
        target_lang = st.selectbox("Целевой язык", Config.TARGET_LANG)


    # Main container with border
    main_container = st.container(border=True)


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
                translation, latency, tokens_count = translate(translation_prompt, model, tokenizer)


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
                    efficiency_container = st.empty()

                    if isinstance(translation, str):
                        st.session_state["latency"].append(latency)
                        st.session_state["tokens_count"].append(tokens_count)
                        
                        plt.figure(figsize=(15, 5))
                        sns.boxplot(x=st.session_state["tokens_count"], y=st.session_state["latency"], native_scale=True, showfliers=False) # строим "ящики с усами", showfliers — отображать ли выбросы, native_scale — воспринимать ли ось X как непрерывную (а не категориальную)
                        plt.xticks(rotation=45, ha='right') # поворот на 45 градусов подписей под осью OX (ha='right' ~ правый конец соответствует колонке)
                        plt.title("Зависимость времени перевода от размера текста") # название фигуры
                        plt.xlabel("Количество токенов") # подпись по оси x
                        plt.ylabel("Задержка перевода (latency, sec)") # подпись по оси y
                        efficiency_container.pyplot(plt) # создаём линейный график на основе данных в словаре session_state
                    else:
                        error_message = f"Error: {str(translation)}"
                        efficiency_container.error(error_message)


    # Sidebar for additional information and feedback
    with st.sidebar:
        st.subheader("Информация")
        st.info("Это приложение является демонстрационной версией основных возможностей.")


model, tokenizer = None, None # изначальные объекты модели и токенизатор не существуют
if __name__ == "__main__":
    main()
