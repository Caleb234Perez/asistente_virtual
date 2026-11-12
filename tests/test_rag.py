"""Pruebas de búsqueda, generación y manejo de errores del modelo.

No requieren Ollama ni el modelo de embeddings: ambos se simulan.
"""

import json

import numpy as np
import pytest

import src.llm as llm
import src.rag as rag
import src.search as search
from src.llm import ErrorModelo
from src.rag import (
    MENSAJE_SIN_INFORMACION,
    normalizar_paginas,
    responder_pregunta,
)


# ---------------------------------------------------------
# Búsqueda semántica
# ---------------------------------------------------------

def test_busqueda_ordena_por_similitud(monkeypatch):
    monkeypatch.setattr(
        search, "generar_embedding",
        lambda pregunta: np.array([1.0, 0.0])
    )

    chunks = [
        {"pagina": 1, "texto": "lejano", "embedding": np.array([0.0, 1.0])},
        {"pagina": 2, "texto": "cercano", "embedding": np.array([1.0, 0.1])},
        {"pagina": 3, "texto": "medio", "embedding": np.array([1.0, 1.0])},
    ]

    resultados = search.buscar_fragmentos("x", chunks, top_k=2)

    assert [r["pagina"] for r in resultados] == [2, 3]
    assert resultados[0]["similitud"] > resultados[1]["similitud"]


# ---------------------------------------------------------
# Normalización de fuentes
# ---------------------------------------------------------

@pytest.mark.parametrize("entrada, esperado", [
    ([1, 2], {1, 2}),
    (["2"], {2}),
    (["Página 3"], {3}),
    (2, {2}),
    ([2.0], {2}),
    ([], set()),
    (None, set()),
    (["sin número"], set()),
    ([True], set()),
])
def test_normalizar_paginas(entrada, esperado):
    assert normalizar_paginas(entrada) == esperado


# ---------------------------------------------------------
# Flujo RAG con modelo simulado
# ---------------------------------------------------------

RESULTADOS_SIMULADOS = [
    {"pagina": 2, "texto": "30 participantes", "similitud": 0.9},
    {"pagina": 1, "texto": "resumen", "similitud": 0.7},
    {"pagina": 3, "texto": "retos", "similitud": 0.5},
]


@pytest.fixture
def busqueda_simulada(monkeypatch):
    monkeypatch.setattr(
        rag, "buscar_fragmentos",
        lambda pregunta, chunks, top_k=3: RESULTADOS_SIMULADOS[:top_k]
    )


def simular_modelo(monkeypatch, datos):
    monkeypatch.setattr(
        rag, "generar_respuesta_estructurada",
        lambda prompt: datos
    )


def test_respuesta_con_fuentes_validas(monkeypatch, busqueda_simulada):
    simular_modelo(monkeypatch, {"respuesta": "Fueron 30.", "fuentes": [2]})

    resultado = responder_pregunta("¿Cuántos participantes?", [])

    assert resultado["tiene_respuesta"]
    assert [f["pagina"] for f in resultado["fuentes"]] == [2]


def test_fuentes_como_texto_se_reconocen(monkeypatch, busqueda_simulada):
    simular_modelo(monkeypatch, {"respuesta": "Fueron 30.", "fuentes": ["2"]})

    resultado = responder_pregunta("¿Cuántos participantes?", [])

    assert [f["pagina"] for f in resultado["fuentes"]] == [2]


def test_pagina_no_recuperada_se_descarta(monkeypatch, busqueda_simulada):
    # El modelo cita la página 9, que no estaba en el contexto.
    simular_modelo(monkeypatch, {"respuesta": "Fueron 30.", "fuentes": [2, 9]})

    resultado = responder_pregunta("¿Cuántos participantes?", [])

    assert [f["pagina"] for f in resultado["fuentes"]] == [2]


def test_sin_informacion_no_muestra_fuentes(monkeypatch, busqueda_simulada):
    # Aunque el modelo cite páginas, si rechaza la pregunta no hay fuentes.
    simular_modelo(
        monkeypatch,
        {"respuesta": MENSAJE_SIN_INFORMACION, "fuentes": [1]}
    )

    resultado = responder_pregunta("¿Capital de Francia?", [])

    assert not resultado["tiene_respuesta"]
    assert resultado["fuentes"] == []


def test_respuesta_vacia_se_trata_como_sin_informacion(
    monkeypatch, busqueda_simulada
):
    simular_modelo(monkeypatch, {"fuentes": [2]})

    resultado = responder_pregunta("pregunta", [])

    assert resultado["respuesta"] == MENSAJE_SIN_INFORMACION
    assert not resultado["tiene_respuesta"]


def test_rechazo_con_mensaje_modificado_se_detecta(
    monkeypatch, busqueda_simulada
):
    # Caso P18 de la evaluación: el modelo cambió el final del mensaje.
    simular_modelo(monkeypatch, {
        "respuesta": "No encontré información suficiente en los "
                     "documentos proporcionados para determinar el costo.",
        "fuentes": []
    })

    resultado = responder_pregunta("¿Cuánto cuesta SPL?", [])

    assert not resultado["tiene_respuesta"]


def test_campo_encontrada_falso_indica_rechazo(
    monkeypatch, busqueda_simulada
):
    simular_modelo(monkeypatch, {
        "encontrada": False,
        "respuesta": "El documento no menciona ese dato.",
        "fuentes": [2]
    })

    resultado = responder_pregunta("pregunta", [])

    assert not resultado["tiene_respuesta"]
    assert resultado["fuentes"] == []


def test_respuesta_sin_fuentes_usa_el_fragmento_mas_similar(
    monkeypatch, busqueda_simulada
):
    # Casos P08 y P14 de la evaluación: respuesta correcta sin páginas.
    simular_modelo(monkeypatch, {
        "encontrada": True,
        "respuesta": "Tenían en promedio 20.43 años.",
        "fuentes": []
    })

    resultado = responder_pregunta("¿Qué edad tenían?", [])

    assert [f["pagina"] for f in resultado["fuentes"]] == [2]
    assert resultado["fuentes_inferidas"]


def test_fuentes_citadas_no_se_marcan_como_inferidas(
    monkeypatch, busqueda_simulada
):
    simular_modelo(monkeypatch, {"respuesta": "Fueron 30.", "fuentes": [1]})

    resultado = responder_pregunta("¿Cuántos participantes?", [])

    assert not resultado["fuentes_inferidas"]


def test_fuentes_incluyen_fragmentos_de_su_pagina(
    monkeypatch, busqueda_simulada
):
    simular_modelo(monkeypatch, {"respuesta": "Fueron 30.", "fuentes": [2]})

    resultado = responder_pregunta("¿Cuántos participantes?", [])

    assert resultado["fuentes"][0]["fragmentos"] == ["30 participantes"]


def test_prompt_incluye_contexto_y_pregunta(monkeypatch, busqueda_simulada):
    prompts = []

    def modelo(prompt):
        prompts.append(prompt)
        return {"respuesta": "ok", "fuentes": []}

    monkeypatch.setattr(rag, "generar_respuesta_estructurada", modelo)

    responder_pregunta("¿Cuántos participantes?", [])

    assert "[Página 2]" in prompts[0]
    assert "30 participantes" in prompts[0]
    assert "¿Cuántos participantes?" in prompts[0]


# ---------------------------------------------------------
# Errores del modelo (Ollama)
# ---------------------------------------------------------

def test_ollama_apagado_genera_error_controlado(monkeypatch):
    def chat(**kwargs):
        raise ConnectionError("connection refused")

    monkeypatch.setattr(llm.ollama, "chat", chat)

    with pytest.raises(ErrorModelo, match="Ollama"):
        llm.generar_respuesta_estructurada("hola")


def test_modelo_no_descargado_genera_error_controlado(monkeypatch):
    def chat(**kwargs):
        raise llm.ollama.ResponseError("model not found", 404)

    monkeypatch.setattr(llm.ollama, "chat", chat)

    with pytest.raises(ErrorModelo, match="ollama pull"):
        llm.generar_respuesta_estructurada("hola")


def test_json_invalido_genera_error_controlado(monkeypatch):
    monkeypatch.setattr(
        llm.ollama, "chat",
        lambda **kwargs: {"message": {"content": "esto no es JSON"}}
    )

    with pytest.raises(ErrorModelo):
        llm.generar_respuesta_estructurada("hola")


def test_json_valido_se_interpreta(monkeypatch):
    contenido = json.dumps({"respuesta": "Sí", "fuentes": [1]})
    monkeypatch.setattr(
        llm.ollama, "chat",
        lambda **kwargs: {"message": {"content": contenido}}
    )

    assert llm.generar_respuesta_estructurada("hola") == {
        "respuesta": "Sí", "fuentes": [1]
    }
