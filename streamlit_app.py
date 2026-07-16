import json
import os

import requests
import streamlit as st


API_URL = st.sidebar.text_input("API URL", os.getenv("API_URL", "http://localhost:8000"))

st.set_page_config(page_title="Artifact Vision MVP", layout="wide")
st.title("Artifact Vision MVP")
st.caption("Первичная классификация находок и поиск визуально похожих эталонов")

analyze_tab, add_tab = st.tabs(["Анализ", "Добавить эталон"])

with analyze_tab:
    uploaded = st.file_uploader("Фотография новой находки", type=["jpg", "jpeg", "png", "webp"])
    top_k = st.slider("Количество аналогов", min_value=1, max_value=20, value=7)
    use_ollama = st.checkbox("Добавить визуальное описание через Ollama")

    if st.button("Проанализировать", type="primary", disabled=uploaded is None):
        response = requests.post(
            f"{API_URL}/v1/analyze",
            files={"file": (uploaded.name, uploaded.getvalue(), uploaded.type)},
            data={"top_k": top_k, "use_ollama": str(use_ollama).lower()},
            timeout=300,
        )
        if response.ok:
            result = response.json()
            col1, col2, col3 = st.columns(3)
            col1.metric("Тип", result["artifact_type"]["label"] or "—", result["artifact_type"]["confidence"])
            col2.metric("Материал", result["material"]["label"] or "—", result["material"]["confidence"])
            col3.metric("Эпоха", result["era"]["label"] or "—", result["era"]["confidence"])
            st.subheader("Датировка")
            st.json(result["dating"])
            st.subheader("Похожие эталоны")
            st.dataframe(result["similar_artifacts"], use_container_width=True)
            if result.get("visual_observation"):
                st.subheader("Наблюдаемые признаки")
                st.json(result["visual_observation"])
            for warning in result.get("warnings", []):
                st.warning(warning)
        else:
            st.error(response.text)

with add_tab:
    reference = st.file_uploader(
        "Фотография эталонной находки",
        type=["jpg", "jpeg", "png", "webp"],
        key="reference",
    )
    title = st.text_input("Название / инвентарный номер")
    artifact_type = st.text_input("Тип изделия")
    material = st.text_input("Материал")
    era = st.text_input("Эпоха")
    col1, col2 = st.columns(2)
    year_from = col1.number_input("Год от", value=None, step=1, placeholder="Например -300")
    year_to = col2.number_input("Год до", value=None, step=1, placeholder="Например -100")
    description = st.text_area("Экспертное описание")
    object_group_id = st.text_input("ID физического объекта")

    if st.button("Добавить в базу", disabled=reference is None):
        metadata = {
            "title": title or None,
            "artifact_type": artifact_type or None,
            "material": material or None,
            "era": era or None,
            "year_from": year_from,
            "year_to": year_to,
            "description": description or None,
            "object_group_id": object_group_id or None,
        }
        response = requests.post(
            f"{API_URL}/v1/artifacts",
            files={"file": (reference.name, reference.getvalue(), reference.type)},
            data={"metadata": json.dumps(metadata, ensure_ascii=False)},
            timeout=300,
        )
        if response.ok:
            st.success("Эталон добавлен")
            st.json(response.json())
        else:
            st.error(response.text)
