"""Pruebas del procesamiento documental: lectura de PDF y fragmentación."""

import io

import pytest
from pypdf import PdfWriter

from src.pdf_reader import extraer_paginas_pdf, ErrorPDF
from src.text_processor import crear_chunks
from tests.conftest import RUTA_PDF_PRUEBA


# ---------------------------------------------------------
# Lectura de PDF
# ---------------------------------------------------------

def test_extrae_las_tres_paginas_del_documento_de_prueba():
    paginas = extraer_paginas_pdf(RUTA_PDF_PRUEBA)

    assert [p["pagina"] for p in paginas] == [1, 2, 3]
    assert all(p["texto"] for p in paginas)


def test_el_texto_extraido_contiene_contenido_esperado():
    paginas = extraer_paginas_pdf(RUTA_PDF_PRUEBA)

    assert "Socratic" in paginas[0]["texto"]
    assert "30 Chinese college students" in paginas[1]["texto"]


def test_pdf_sin_texto_devuelve_lista_vacia():
    # PDF con una página en blanco: equivale a un PDF escaneado sin OCR.
    escritor = PdfWriter()
    escritor.add_blank_page(width=612, height=792)

    archivo = io.BytesIO()
    escritor.write(archivo)
    archivo.seek(0)

    assert extraer_paginas_pdf(archivo) == []


def test_archivo_que_no_es_pdf_genera_error_controlado():
    archivo = io.BytesIO(b"esto no es un PDF")

    with pytest.raises(ErrorPDF):
        extraer_paginas_pdf(archivo)


def test_pdf_protegido_con_contrasena_genera_error_controlado():
    escritor = PdfWriter()
    escritor.add_blank_page(width=612, height=792)
    escritor.encrypt(user_password="secreta", algorithm="RC4-128")

    archivo = io.BytesIO()
    escritor.write(archivo)
    archivo.seek(0)

    with pytest.raises(ErrorPDF):
        extraer_paginas_pdf(archivo)


# ---------------------------------------------------------
# Fragmentación
# ---------------------------------------------------------

def test_fragmentos_conservan_el_numero_de_pagina():
    paginas = [
        {"pagina": 1, "texto": "a" * 1500},
        {"pagina": 2, "texto": "b" * 300},
    ]

    chunks = crear_chunks(paginas)

    assert {c["pagina"] for c in chunks} == {1, 2}
    assert all(set(c["texto"]) == {"a"} for c in chunks if c["pagina"] == 1)


def test_fragmentos_respetan_tamano_y_solapamiento():
    texto = "".join(str(i % 10) for i in range(2500))
    chunks = crear_chunks(
        [{"pagina": 1, "texto": texto}],
        tamano_chunk=1000,
        solapamiento=200
    )

    assert all(len(c["texto"]) <= 1000 for c in chunks)
    # El final del primer fragmento se repite al inicio del segundo.
    assert chunks[0]["texto"][-200:] == chunks[1]["texto"][:200]


def test_documento_de_prueba_genera_trece_fragmentos():
    # Coincide con lo registrado en el manual técnico (sección 9.1).
    paginas = extraer_paginas_pdf(RUTA_PDF_PRUEBA)

    assert len(crear_chunks(paginas)) == 13


def test_paginas_vacias_no_generan_fragmentos():
    assert crear_chunks([{"pagina": 1, "texto": ""}]) == []
