import math
import re

from pypdf import PdfReader
from pypdf.errors import PdfReadError


# Encabezado de la sección de referencias en su propia línea, con número
# opcional: "References", "7. Referencias", "BIBLIOGRAFÍA", etc.
PATRON_REFERENCIAS = re.compile(
    r"^[ \t]*(?:\d+\.?[ \t]*)?"
    r"(?:references|referencias|referencias bibliogr[aá]ficas|"
    r"bibliography|bibliograf[ií]a|works cited|literatura citada)"
    r"[ \t]*:?[ \t]*$",
    re.IGNORECASE | re.MULTILINE
)


class ErrorPDF(Exception):
    """El archivo no pudo leerse como PDF."""


def excluir_referencias(paginas):
    """Elimina la sección de referencias bibliográficas y lo que le sigue.

    Los nombres de autores citados se confundían con los autores del
    documento. Solo se considera el último encabezado que esté en la
    segunda mitad del documento, para no cortar un índice que mencione
    "Referencias" al principio.
    """

    if not paginas:
        return paginas

    primera_valida = math.ceil(len(paginas) / 2) - 1

    for indice in range(len(paginas) - 1, primera_valida - 1, -1):
        coincidencias = list(
            PATRON_REFERENCIAS.finditer(paginas[indice]["texto"])
        )

        if not coincidencias:
            continue

        corte = coincidencias[-1].start()
        texto_previo = paginas[indice]["texto"][:corte].strip()

        resultado = paginas[:indice]

        if texto_previo:
            resultado.append({
                "pagina": paginas[indice]["pagina"],
                "texto": texto_previo
            })

        return resultado

    return paginas


def extraer_paginas_pdf(origen_pdf, omitir_referencias=True):
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

        if omitir_referencias:
            paginas = excluir_referencias(paginas)

        return paginas

    except ErrorPDF:
        raise

    except (PdfReadError, ValueError, KeyError) as error:
        raise ErrorPDF(
            "No fue posible leer el archivo. "
            "Verifica que sea un PDF válido y que no esté dañado."
        ) from error