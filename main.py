import sys

from src.pdf_reader import extraer_paginas_pdf, ErrorPDF
from src.text_processor import crear_chunks
from src.embeddings import generar_embeddings_chunks
from src.rag import responder_pregunta
from src.llm import ErrorModelo


def main():
    ruta_pdf = "data/documento_prueba.pdf"

    print("Preparando asistente documental...")

    try:
        paginas = extraer_paginas_pdf(ruta_pdf)
    except (ErrorPDF, FileNotFoundError) as error:
        print(f"Error: {error}")
        sys.exit(1)

    if not paginas:
        print(
            "Error: el documento no contiene texto extraíble. "
            "Verifica que no esté compuesto solo por imágenes."
        )
        sys.exit(1)

    chunks = crear_chunks(paginas)
    chunks = generar_embeddings_chunks(chunks)

    print("\nDocumento preparado correctamente.")

    pregunta = input("\nEscribe tu pregunta: ")

    print("\nGenerando respuesta...")

    try:
        resultado = responder_pregunta(
            pregunta,
            chunks
        )
    except ErrorModelo as error:
        print(f"Error: {error}")
        sys.exit(1)

    print("\nRespuesta:")
    print("=" * 60)
    print(resultado["respuesta"])

    print("\nFuentes:")
    print("=" * 60)

    if resultado["fuentes"]:
        paginas_mostradas = set()

        for fuente in resultado["fuentes"]:
            pagina = fuente["pagina"]

            if pagina not in paginas_mostradas:
                print(
                    f"documento_prueba.pdf - Página {pagina}"
                )
                paginas_mostradas.add(pagina)

    else:
        print(
            "No se identificaron fuentes suficientes "
            "para responder la consulta."
        )


if __name__ == "__main__":
    main()