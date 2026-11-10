"""Evaluación del asistente documental.

Mide dos cosas con el conjunto de preguntas de evaluacion/preguntas.json:

1. Recuperación (siempre): si la página que contiene la respuesta aparece
   entre los fragmentos recuperados.
     - Acierto@1: la primera página recuperada es una página esperada.
     - Acierto@3: alguna de las 3 primeras es una página esperada
       (el asistente usa top_k=3, así que esta es la métrica clave).
     - MRR: promedio de 1/posición de la primera página correcta.

2. Generación (opcional, con --con-llm; requiere Ollama en ejecución):
     - Preguntas con respuesta: el asistente responde y cita al menos una
       página correcta y ninguna incorrecta.
     - Preguntas fuera de contenido: el asistente rechaza la pregunta.
   El texto de cada respuesta se guarda en el CSV para revisarlo a mano.

Uso, desde la raíz del proyecto:

    python -m evaluacion.evaluar
    python -m evaluacion.evaluar --con-llm
"""

import argparse
import csv
import json
import os
import time
from datetime import datetime

from src.pdf_reader import extraer_paginas_pdf
from src.text_processor import crear_chunks
from src.embeddings import generar_embeddings_chunks
from src.search import buscar_fragmentos


CARPETA = os.path.dirname(os.path.abspath(__file__))
TOP_K = 3


def paginas_unicas_en_orden(resultados):
    paginas = []

    for resultado in resultados:
        if resultado["pagina"] not in paginas:
            paginas.append(resultado["pagina"])

    return paginas


def evaluar_recuperacion(pregunta, chunks):
    resultados = buscar_fragmentos(
        pregunta["pregunta"], chunks, top_k=len(chunks)
    )
    ranking = paginas_unicas_en_orden(resultados)
    top = paginas_unicas_en_orden(resultados[:TOP_K])

    esperadas = set(pregunta["paginas_esperadas"])

    posicion = next(
        (i for i, p in enumerate(ranking, start=1) if p in esperadas),
        None
    )

    return {
        "paginas_recuperadas": top,
        "similitud_max": round(resultados[0]["similitud"], 3),
        "acierto_1": int(bool(top) and top[0] in esperadas),
        "acierto_3": int(any(p in esperadas for p in top)),
        "rr": (1 / posicion) if posicion else 0.0,
    }


def evaluar_generacion(pregunta, chunks):
    # Se importa aquí para que la evaluación de recuperación funcione
    # aunque Ollama no esté instalado.
    from src.rag import responder_pregunta

    inicio = time.perf_counter()
    resultado = responder_pregunta(pregunta["pregunta"], chunks, top_k=TOP_K)
    segundos = time.perf_counter() - inicio

    esperadas = set(pregunta["paginas_esperadas"])
    citadas = {f["pagina"] for f in resultado["fuentes"]}

    if esperadas:
        correcto = (
            resultado["tiene_respuesta"]
            and bool(citadas)
            and citadas <= esperadas
        )
    else:
        correcto = not resultado["tiene_respuesta"]

    return {
        "respondio": int(resultado["tiene_respuesta"]),
        "paginas_citadas": sorted(citadas),
        "fuentes_inferidas": int(resultado.get("fuentes_inferidas", False)),
        "comportamiento_correcto": int(correcto),
        "segundos": round(segundos, 1),
        "respuesta": resultado["respuesta"].replace("\n", " "),
    }


def porcentaje(valores):
    return f"{100 * sum(valores) / len(valores):.1f}%" if valores else "—"


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--con-llm", action="store_true",
        help="Evalúa también las respuestas generadas (requiere Ollama)."
    )
    parser.add_argument(
        "--preguntas", default=os.path.join(CARPETA, "preguntas.json")
    )
    args = parser.parse_args()

    with open(args.preguntas, encoding="utf-8") as archivo:
        conjunto = json.load(archivo)

    if not os.path.exists(conjunto["documento"]):
        raise SystemExit(
            f"No se encontró {conjunto['documento']}. "
            "Copia el PDF de prueba en la carpeta data/ y vuelve a ejecutar."
        )

    print(f"Preparando documento: {conjunto['documento']}")
    paginas = extraer_paginas_pdf(conjunto["documento"])
    chunks = generar_embeddings_chunks(crear_chunks(paginas))

    filas = []

    for pregunta in conjunto["preguntas"]:
        print(f"  {pregunta['id']}: {pregunta['pregunta']}")

        fila = {
            "id": pregunta["id"],
            "tipo": pregunta["tipo"],
            "pregunta": pregunta["pregunta"],
            "paginas_esperadas": pregunta["paginas_esperadas"],
        }

        if pregunta["paginas_esperadas"]:
            fila.update(evaluar_recuperacion(pregunta, chunks))
        else:
            # Sin página esperada no se mide recuperación, pero se registra
            # la similitud máxima: suele ser menor en preguntas ajenas.
            resultados = buscar_fragmentos(pregunta["pregunta"], chunks, 1)
            fila["similitud_max"] = round(resultados[0]["similitud"], 3)

        if args.con_llm:
            fila.update(evaluar_generacion(pregunta, chunks))

        filas.append(fila)

    # -----------------------------------------------------
    # Resumen
    # -----------------------------------------------------

    con_respuesta = [f for f in filas if f["paginas_esperadas"]]
    sin_respuesta = [f for f in filas if not f["paginas_esperadas"]]

    lineas = [
        "# Resultados de la evaluación",
        "",
        f"Fecha: {datetime.now():%Y-%m-%d %H:%M}",
        f"Documento: {conjunto['documento']} "
        f"({len(paginas)} páginas, {len(chunks)} fragmentos)",
        f"Preguntas: {len(filas)} "
        f"({len(con_respuesta)} con respuesta, "
        f"{len(sin_respuesta)} fuera de contenido)",
        "",
        "## Recuperación de información",
        "",
        "| Métrica | Valor |",
        "|---|---|",
        f"| Acierto@1 | {porcentaje([f['acierto_1'] for f in con_respuesta])} |",
        f"| Acierto@3 | {porcentaje([f['acierto_3'] for f in con_respuesta])} |",
        "| MRR | "
        f"{sum(f['rr'] for f in con_respuesta) / len(con_respuesta):.3f} |",
    ]

    if sin_respuesta:
        prom_con = sum(f["similitud_max"] for f in con_respuesta) / len(con_respuesta)
        prom_sin = sum(f["similitud_max"] for f in sin_respuesta) / len(sin_respuesta)
        lineas += [
            f"| Similitud máxima promedio (con respuesta) | {prom_con:.3f} |",
            f"| Similitud máxima promedio (fuera de contenido) | {prom_sin:.3f} |",
        ]

    if args.con_llm:
        lineas += [
            "",
            "## Generación de respuestas",
            "",
            "| Métrica | Valor |",
            "|---|---|",
            "| Respuestas con fuente correcta | "
            f"{porcentaje([f['comportamiento_correcto'] for f in con_respuesta])} |",
            "| Rechazo correcto fuera de contenido | "
            f"{porcentaje([f['comportamiento_correcto'] for f in sin_respuesta])} |",
            "| Tiempo promedio por respuesta | "
            f"{sum(f['segundos'] for f in filas) / len(filas):.1f} s |",
        ]

    lineas += ["", "## Detalle por tipo de pregunta", ""]
    lineas += ["| Tipo | Preguntas | Acierto@3 |", "|---|---|---|"]

    for tipo in dict.fromkeys(f["tipo"] for f in con_respuesta):
        grupo = [f for f in con_respuesta if f["tipo"] == tipo]
        lineas.append(
            f"| {tipo} | {len(grupo)} | "
            f"{porcentaje([f['acierto_3'] for f in grupo])} |"
        )

    resumen = "\n".join(lineas) + "\n"

    marca = datetime.now().strftime("%Y%m%d_%H%M")
    ruta_csv = os.path.join(CARPETA, f"resultados_{marca}.csv")
    ruta_md = os.path.join(CARPETA, f"resumen_{marca}.md")

    columnas = list(dict.fromkeys(c for f in filas for c in f))

    with open(ruta_csv, "w", newline="", encoding="utf-8-sig") as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=columnas)
        escritor.writeheader()
        escritor.writerows(filas)

    with open(ruta_md, "w", encoding="utf-8") as archivo:
        archivo.write(resumen)

    print()
    print(resumen)
    print(f"Detalle guardado en: {ruta_csv}")
    print(f"Resumen guardado en: {ruta_md}")


if __name__ == "__main__":
    main()
