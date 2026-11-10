_modelo = None


def obtener_modelo():
    global _modelo

    if _modelo is None:
        # Se importa aquí para no cargar la librería (que es pesada)
        # hasta que realmente se necesita.
        from sentence_transformers import SentenceTransformer

        print("Cargando modelo de embeddings...")

        _modelo = SentenceTransformer(
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )

    return _modelo


def generar_embedding(texto):
    modelo = obtener_modelo()
    return modelo.encode(texto)


def generar_embeddings_chunks(chunks):
    modelo = obtener_modelo()

    textos = [chunk["texto"] for chunk in chunks]

    embeddings = modelo.encode(
        textos,
        show_progress_bar=True
    )

    for chunk, embedding in zip(chunks, embeddings):
        chunk["embedding"] = embedding

    return chunks