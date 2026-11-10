from pypdf import PdfReader
from pypdf.errors import PdfReadError


class ErrorPDF(Exception):
    """El archivo no pudo leerse como PDF."""


def extraer_paginas_pdf(origen_pdf):
    try:
        lector = PdfReader(origen_pdf)

        if lector.is_encrypted:
            # Algunos PDFs están cifrados sin contraseña de apertura.
            try:
                descifrado = lector.decrypt("")
            except Exception as error:
                raise ErrorPDF(
                    "El PDF está cifrado y no pudo abrirse."
                ) from error

            if not descifrado:
                raise ErrorPDF(
                    "El PDF está protegido con contraseña."
                )

        paginas = []

        for numero_pagina, pagina in enumerate(lector.pages, start=1):
            texto = pagina.extract_text()

            if texto and texto.strip():
                paginas.append({
                    "pagina": numero_pagina,
                    "texto": texto.strip()
                })

        return paginas

    except ErrorPDF:
        raise

    except (PdfReadError, ValueError, KeyError) as error:
        raise ErrorPDF(
            "No fue posible leer el archivo. "
            "Verifica que sea un PDF válido y que no esté dañado."
        ) from error