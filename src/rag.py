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


def determinar_si_respondio(datos, respuesta):
    """Indica si el modelo respondió o rechazó la pregunta.

    Se usa el campo "encontrada" del JSON cuando el modelo lo incluye.
    Si no, se revisa si la respuesta comienza con el mensaje de rechazo:
    el modelo a veces lo modifica ("...para determinar el costo..."),
    por lo que no basta con una comparación exacta.
    """

    texto_de_rechazo = respuesta.strip().lower().startswith(
        "no encontré información suficiente"
    )

    encontrada = datos.get("encontrada")

    if isinstance(encontrada, bool):
        return encontrada and not texto_de_rechazo

    return not texto_de_rechazo


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
    "encontrada": true,
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
7. Si respondes la pregunta, "fuentes" debe contener al menos
   una página.
8. La lista de referencias bibliográficas solo indica obras
   citadas: sus autores NO son los autores del documento.
9. Si no existe información suficiente, devuelve:

{{
    "encontrada": false,
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

    tiene_respuesta = determinar_si_respondio(datos, respuesta)

    fuentes_inferidas = False

    if not tiene_respuesta:
        fuentes = []

    elif not fuentes and resultados:
        # El modelo respondió pero no citó páginas válidas: se usa la
        # página del fragmento más similar a la pregunta.
        fuentes = [resultados[0]]
        fuentes_inferidas = True

    return {
        "respuesta": respuesta,
        "fuentes": fuentes,
        "tiene_respuesta": tiene_respuesta,
        "fuentes_inferidas": fuentes_inferidas
    }