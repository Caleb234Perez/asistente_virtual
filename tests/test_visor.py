"""Pruebas del visor de PDF con texto resaltado."""

import io
import os

import pytest
from PIL import Image

from src.pdf_reader import extraer_paginas_pdf
from src.pdf_viewer import (
    _unir_por_renglon,
    contar_paginas,
    renderizar_pagina,
)
from src.text_processor import crear_chunks
from tests.conftest import RUTA_PDF_PRUEBA


requiere_pdf = pytest.mark.skipif(
    not os.path.exists(RUTA_PDF_PRUEBA),
    reason="Falta data/documento_prueba.pdf"
)


@pytest.fixture
def pdf_bytes():
    with open(RUTA_PDF_PRUEBA, "rb") as archivo:
        return archivo.read()


def fragmento_con(texto, pagina):
    chunks = crear_chunks(extraer_paginas_pdf(RUTA_PDF_PRUEBA))

    return next(
        c["texto"] for c in chunks
        if texto in c["texto"] and c["pagina"] == pagina
    )


def es_png(datos):
    return Image.open(io.BytesIO(datos)).format == "PNG"


@requiere_pdf
def test_cuenta_las_paginas_del_pdf(pdf_bytes):
    assert contar_paginas(pdf_bytes) == 3


@requiere_pdf
def test_pagina_sin_resaltado(pdf_bytes):
    vista = renderizar_pagina(pdf_bytes, 1)

    assert es_png(vista["pagina"])
    assert vista["recorte"] is None
    assert not vista["encontrado"]


@requiere_pdf
@pytest.mark.parametrize("texto, pagina", [
    ("Yueru Lang", 1),
    ("170 minutes", 2),
    ("ethical concerns", 3),
])
def test_resalta_el_fragmento_en_su_pagina(pdf_bytes, texto, pagina):
    vista = renderizar_pagina(pdf_bytes, pagina, [fragmento_con(texto, pagina)])

    assert vista["encontrado"]
    assert es_png(vista["recorte"])

    pagina_completa = Image.open(io.BytesIO(vista["pagina"]))
    recorte = Image.open(io.BytesIO(vista["recorte"]))

    # El recorte muestra solo la zona del párrafo, con el mismo ancho.
    assert recorte.width == pagina_completa.width
    assert recorte.height < pagina_completa.height


@requiere_pdf
def test_texto_ajeno_no_se_resalta(pdf_bytes):
    vista = renderizar_pagina(
        pdf_bytes, 1, ["Este texto no aparece en ninguna parte del documento"]
    )

    assert not vista["encontrado"]


def test_une_cajas_del_mismo_renglon():
    cajas = [(10, 100, 50, 112), (60, 101, 90, 112), (10, 120, 80, 132)]

    assert _unir_por_renglon(cajas) == [
        (10, 100, 90, 112),
        (10, 120, 80, 132),
    ]
