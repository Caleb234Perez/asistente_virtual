import re

from src.search import buscar_fragmentos
from src.llm import generar_respuesta_estructurada


MENSAJE_SIN_INFORMACION = (
    "No encontré información suficiente en los documentos "
    "proporcionados para responder esta pregunta."
)


def normalizar_paginas(fuentes):
    """Convierte las páginas reportadas por el modelo a enteros.

    El modelo puede devolver [2], ["2"], ["Página 2"] o un solo valor.
    Los valores que no contienen un número se descartan.
    """

    if not isinstance(fuentes, list):
        fuentes = [fuentes]

    paginas = set()

    for fuente in fuentes:
        if isinstance(fuente, bool):
            continue

        if isinstance(fuente, (int, float)):
            paginas.add(int(fuente))
            continue

        for numero in re.findall(r"\d+", str(fuente)):
            paginas.add(int(numero))

    return paginas


def responder_pregunta(pregunta, chunks, top_k=3):

    resultados = buscar_fragmentos(
        pregunta,
        chunks,
        top_k=top_k
    )

    contexto_partes = []

    for resultado in resultados:
        contexto_partes.append(
            f"[Página {resultado['pagina']}]\n"
            f"{resultado['texto']}"
        )

    contexto = "\n\n".join(contexto_partes)

    prompt = f"""
Eres un asistente documental.

Utiliza EXCLUSIVAMENTE el contexto proporcionado para responder.

Debes devolver un objeto JSON con esta estructura exacta:

{{
    "respuesta": "texto de la respuesta",
    "fuentes": [1, 2]
}}

REGLAS:

1. Busca primero información que responda directamente la pregunta.
2. Si la respuesta aparece explícitamente en el contexto, úsala.
3. No utilices conocimiento externo.
4. No inventes información.
5. En "fuentes" incluye únicamente las páginas que respaldan
   directamente la respuesta.
6. No incluyas una página solamente porque tenga información
   relacionada con el tema.
7. Si no existe información suficiente, devuelve:

{{
    "respuesta": "{MENSAJE_SIN_INFORMACION}",
    "fuentes": []
}}

CONTEXTO:

{contexto}

PREGUNTA:

{pregunta}
"""

    datos = generar_respuesta_estructurada(prompt)

    respuesta = datos.get("respuesta")

    if not isinstance(respuesta, str) or not respuesta.strip():
        respuesta = MENSAJE_SIN_INFORMACION

    paginas_citadas = normalizar_paginas(
        datos.get("fuentes", [])
    )

    fuentes = []

    for resultado in resultados:

        if resultado["pagina"] in paginas_citadas:

            if not any(
                fuente["pagina"] == resultado["pagina"]
                for fuente in fuentes
            ):
                fuentes.append(resultado)

    tiene_respuesta = (
        respuesta.strip().lower()
        != MENSAJE_SIN_INFORMACION.lower()
    )

    if not tiene_respuesta:
        fuentes = []

    return {
        "respuesta": respuesta,
        "fuentes": fuentes,
        "tiene_respuesta": tiene_respuesta
    }