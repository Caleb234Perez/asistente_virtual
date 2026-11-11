"""Pruebas del procesamiento documental: lectura de PDF y fragmentación."""

import io
import os

import pytest
from pypdf import PdfWriter

from src.pdf_reader import extraer_paginas_pdf, excluir_referencias, ErrorPDF
from src.text_processor import crear_chunks
from tests.conftest import RUTA_PDF_PRUEBA


# El PDF de prueba no se sube al repositorio (ver .gitignore). Si no está
# en data/, las pruebas que lo usan se omiten en lugar de fallar.
requiere_pdf = pytest.mark.skipif(
    not os.path.exists(RUTA_PDF_PRUEBA),
    reason="Falta data/documento_prueba.pdf"
)


# ---------------------------------------------------------
# Lectura de PDF
# ---------------------------------------------------------

@requiere_pdf
def test_extrae_las_tres_paginas_del_documento_de_prueba():
    paginas = extraer_paginas_pdf(RUTA_PDF_PRUEBA)

    assert [p["pagina"] for p in paginas] == [1, 2, 3]
    assert all(p["texto"] for p in paginas)


@requiere_pdf
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


@requiere_pdf
def test_documento_de_prueba_genera_once_fragmentos_sin_referencias():
    paginas = extraer_paginas_pdf(RUTA_PDF_PRUEBA)

    assert len(crear_chunks(paginas)) == 11


@requiere_pdf
def test_documento_de_prueba_genera_trece_fragmentos_con_referencias():
    # Valor registrado en el manual técnico antes de excluir referencias.
    paginas = extraer_paginas_pdf(RUTA_PDF_PRUEBA, omitir_referencias=False)

    assert len(crear_chunks(paginas)) == 13


@requiere_pdf
def test_documento_de_prueba_no_incluye_referencias():
    paginas = extraer_paginas_pdf(RUTA_PDF_PRUEBA)
    texto = " ".join(p["texto"] for p in paginas)

    assert "Future research should employ" in texto
    assert "Dalim, S. F." not in texto


# ---------------------------------------------------------
# Exclusión de referencias
# ---------------------------------------------------------

def paginas_de(*textos):
    return [
        {"pagina": i, "texto": t} for i, t in enumerate(textos, start=1)
    ]


@pytest.mark.parametrize("encabezado", [
    "References", "REFERENCIAS", "7. Referencias", "Bibliografía",
    "Referencias bibliográficas", "Bibliography:",
])
def test_reconoce_encabezados_de_referencias(encabezado):
    paginas = paginas_de("Introducción", f"Conclusión.\n{encabezado}\nAutor, A. (2020).")

    resultado = excluir_referencias(paginas)

    assert resultado[-1]["texto"] == "Conclusión."


def test_descarta_paginas_posteriores_a_las_referencias():
    paginas = paginas_de("Intro", "Resultados", "Fin.\nReferences\nA, B.", "Más citas")

    resultado = excluir_referencias(paginas)

    assert [p["pagina"] for p in resultado] == [1, 2, 3]


def test_pagina_solo_con_referencias_se_elimina():
    paginas = paginas_de("Intro", "Resultados", "References\nA, B. (2020).")

    assert [p["pagina"] for p in excluir_referencias(paginas)] == [1, 2]


def test_ignora_referencias_en_la_primera_mitad():
    # Un índice al inicio que menciona "Referencias" no debe cortar el texto.
    paginas = paginas_de("Índice\nReferencias\n", "Capítulo 1", "Capítulo 2", "Capítulo 3")

    assert excluir_referencias(paginas) == paginas


def test_no_corta_la_palabra_dentro_de_una_oracion():
    paginas = paginas_de("Intro", "Las references del modelo son útiles.")

    assert excluir_referencias(paginas) == paginas


def test_paginas_vacias_no_generan_fragmentos():
    assert crear_chunks([{"pagina": 1, "texto": ""}]) == []
