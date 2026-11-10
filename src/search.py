import numpy as np

from src.embeddings import generar_embedding


def buscar_fragmentos(pregunta, chunks, top_k=3):
    embedding_pregunta = generar_embedding(pregunta)

    resultados = []

    for chunk in chunks:
        embedding_chunk = chunk["embedding"]

        similitud = np.dot(
            embedding_pregunta,
            embedding_chunk
        ) / (
            np.linalg.norm(embedding_pregunta)
            * np.linalg.norm(embedding_chunk)
        )

        resultados.append({
            "pagina": chunk["pagina"],
            "texto": chunk["texto"],
            "similitud": float(similitud)
        })

    resultados.sort(
        key=lambda resultado: resultado["similitud"],
        reverse=True
    )

    return resultados[:top_k]