"""Visualización de páginas del PDF con el texto de las fuentes resaltado.

Se usa pypdfium2 para dibujar la página como imagen y para ubicar en ella
el texto de los fragmentos que respaldan una respuesta.
"""

import io
import threading

import pypdfium2 as pdfium
from PIL import Image, ImageDraw


# pdfium no admite que dos hilos lo usen al mismo tiempo, y Streamlit
# puede ejecutar la aplicación en varios hilos.
_candado = threading.Lock()

COLOR_RESALTADO = (255, 214, 0, 95)
COLOR_BORDE = (230, 170, 0, 255)
LONGITUD_MINIMA_LINEA = 12
PALABRAS_POR_VENTANA = 6


def _normalizar(texto):
    return " ".join(texto.split())


def _consultas_de_linea(linea):
    """Devuelve la línea completa y, como respaldo, ventanas de palabras.

    La extracción de pypdf y la de pdfium pueden diferir en espacios o
    guiones; buscar trozos más cortos aumenta las coincidencias.
    """

    palabras = linea.split()

    ventanas = [
        " ".join(palabras[i:i + PALABRAS_POR_VENTANA])
        for i in range(0, len(palabras), PALABRAS_POR_VENTANA)
        if len(palabras[i:i + PALABRAS_POR_VENTANA]) >= 3
    ]

    return linea, ventanas


def _ocurrencias(pagina_texto, consulta):
    buscador = pagina_texto.search(consulta, match_case=False)

    try:
        while True:
            ocurrencia = buscador.get_next()

            if ocurrencia is None:
                break

            yield ocurrencia
    finally:
        buscador.close()


def _primera_desde(pagina_texto, consulta, cursor):
    """Primera ocurrencia a partir de cursor; si no hay, la primera."""

    primera = None

    for indice, cantidad in _ocurrencias(pagina_texto, consulta):
        if primera is None:
            primera = (indice, cantidad)

        if indice >= cursor:
            return indice, cantidad

    return primera


def _rectangulos_de_ocurrencia(pagina_texto, indice, cantidad):
    total = pagina_texto.count_rects(indice, cantidad)

    return [pagina_texto.get_rect(i) for i in range(total)]


def ubicar_texto(pagina_texto, texto):
    """Rectángulos (en puntos del PDF) que cubren el texto en la página.

    Las líneas se buscan en orden, avanzando un cursor, para no resaltar
    una frase repetida en otra parte de la página.
    """

    rectangulos = []
    cursor = 0

    for linea_original in texto.splitlines():
        linea = _normalizar(linea_original)

        if len(linea) < LONGITUD_MINIMA_LINEA:
            continue

        completa, ventanas = _consultas_de_linea(linea)

        ocurrencia = _primera_desde(pagina_texto, completa, cursor)

        if ocurrencia is not None:
            indice, cantidad = ocurrencia
            rectangulos += _rectangulos_de_ocurrencia(
                pagina_texto, indice, cantidad
            )
            cursor = indice + cantidad
            continue

        for ventana in ventanas:
            ocurrencia = _primera_desde(pagina_texto, ventana, cursor)

            if ocurrencia is not None:
                indice, cantidad = ocurrencia
                rectangulos += _rectangulos_de_ocurrencia(
                    pagina_texto, indice, cantidad
                )
                cursor = indice + cantidad

    return rectangulos


def _unir_por_renglon(cajas):
    """Une las cajas de un mismo renglón en una sola franja continua.

    La búsqueda puede dejar huecos entre palabras; uniéndolas, el
    resaltado se ve como el de un lector de PDF.
    """

    renglones = []

    for caja in sorted(cajas, key=lambda c: (c[1], c[0])):
        centro = (caja[1] + caja[3]) / 2
        alto = caja[3] - caja[1]

        for renglon in renglones:
            centro_renglon = (renglon[1] + renglon[3]) / 2

            if abs(centro - centro_renglon) <= max(alto, 1) * 0.5:
                renglon[0] = min(renglon[0], caja[0])
                renglon[1] = min(renglon[1], caja[1])
                renglon[2] = max(renglon[2], caja[2])
                renglon[3] = max(renglon[3], caja[3])
                break
        else:
            renglones.append(list(caja))

    return [tuple(r) for r in renglones]


def _a_png(imagen):
    salida = io.BytesIO()
    imagen.save(salida, format="PNG")
    return salida.getvalue()


def renderizar_pagina(pdf_bytes, numero_pagina, textos_resaltados=(),
                      escala=1.6, margen=36):
    """Dibuja una página y resalta los textos indicados.

    Devuelve un diccionario con:
      - "pagina": PNG de la página completa.
      - "recorte": PNG de la zona resaltada (todo el ancho de la página),
        o None si no se encontró el texto.
      - "encontrado": True si se resaltó al menos una línea.
    """

    with _candado:
        documento = pdfium.PdfDocument(pdf_bytes)

        try:
            pagina = documento[numero_pagina - 1]
            pagina_texto = pagina.get_textpage()

            rectangulos = []

            for texto in textos_resaltados:
                rectangulos += ubicar_texto(pagina_texto, texto)

            izquierda, abajo, _, arriba = pagina.get_cropbox()
            imagen = pagina.render(scale=escala).to_pil().convert("RGBA")

            pagina_texto.close()
            pagina.close()
        finally:
            documento.close()

    def a_pixeles(rect):
        x0, y0, x1, y1 = rect
        return (
            (x0 - izquierda) * escala,
            (arriba - y1) * escala,
            (x1 - izquierda) * escala,
            (arriba - y0) * escala,
        )

    cajas = _unir_por_renglon([a_pixeles(r) for r in rectangulos])

    if cajas:
        capa = Image.new("RGBA", imagen.size, (0, 0, 0, 0))
        dibujo = ImageDraw.Draw(capa)

        for caja in cajas:
            dibujo.rectangle(caja, fill=COLOR_RESALTADO)

        imagen = Image.alpha_composite(imagen, capa)

        # Barra lateral que marca el párrafo, como en un lector de PDF.
        marca = ImageDraw.Draw(imagen)
        superior = min(c[1] for c in cajas)
        inferior = max(c[3] for c in cajas)
        borde_x = max(4, min(c[0] for c in cajas) - 10)
        marca.rectangle(
            (borde_x - 4, superior, borde_x, inferior),
            fill=COLOR_BORDE
        )

    imagen = imagen.convert("RGB")

    recorte = None

    if cajas:
        margen_px = margen * escala
        superior = max(0, min(c[1] for c in cajas) - margen_px)
        inferior = min(imagen.height, max(c[3] for c in cajas) + margen_px)
        recorte = _a_png(
            imagen.crop((0, int(superior), imagen.width, int(inferior)))
        )

    return {
        "pagina": _a_png(imagen),
        "recorte": recorte,
        "encontrado": bool(cajas),
    }


def contar_paginas(pdf_bytes):
    with _candado:
        documento = pdfium.PdfDocument(pdf_bytes)

        try:
            return len(documento)
        finally:
            documento.close()
