def crear_chunks(paginas, tamano_chunk=1000, solapamiento=200):
    chunks = []

    for pagina in paginas:
        texto = pagina["texto"]
        numero_pagina = pagina["pagina"]

        inicio = 0

        while inicio < len(texto):
            fin = inicio + tamano_chunk
            fragmento = texto[inicio:fin].strip()

            if fragmento:
                chunks.append({
                    "pagina": numero_pagina,
                    "texto": fragmento
                })

            inicio += tamano_chunk - solapamiento

    return chunks