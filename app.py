import streamlit as st

from src.pdf_reader import extraer_paginas_pdf, ErrorPDF
from src.text_processor import crear_chunks
from src.embeddings import generar_embeddings_chunks
from src.rag import responder_pregunta
from src.llm import ErrorModelo
from src.pdf_viewer import renderizar_pagina, contar_paginas


@st.cache_data(show_spinner=False, max_entries=64)
def pagina_resaltada(pdf_bytes, numero_pagina, textos=()):
    return renderizar_pagina(pdf_bytes, numero_pagina, textos)


def limpiar_documento():
    st.session_state.chunks = None
    st.session_state.nombre_documento = None
    st.session_state.numero_paginas = 0
    st.session_state.numero_fragmentos = 0
    st.session_state.mensajes = []
    st.session_state.pdf_bytes = None
    st.session_state.total_paginas_pdf = 0


def mostrar_en_documento(pagina, fragmentos):
    """Muestra la página citada con los fragmentos usados resaltados."""

    if not st.session_state.get("pdf_bytes"):
        return

    with st.expander(f"🔎 Ver página {pagina} en el documento"):

        with st.spinner("Cargando página..."):
            vista = pagina_resaltada(
                st.session_state.pdf_bytes,
                pagina,
                tuple(fragmentos)
            )

        if vista["encontrado"]:
            pestana_parrafo, pestana_pagina = st.tabs(
                ["Párrafo resaltado", "Página completa"]
            )

            with pestana_parrafo:
                st.image(vista["recorte"])
                st.caption(
                    "En amarillo, el texto que el asistente leyó "
                    "para responder."
                )

            with pestana_pagina:
                st.image(vista["pagina"])

        else:
            st.image(vista["pagina"])
            st.caption(
                "No fue posible ubicar el texto exacto en la página; "
                "se muestra la página completa."
            )


def mostrar_fuentes(fuentes):
    st.caption("Fuentes")

    for fuente in fuentes:

        # Mensajes antiguos guardaban solo el número de página.
        if isinstance(fuente, int):
            fuente = {"pagina": fuente, "fragmentos": []}

        st.markdown(
            f"""
            <div class="source-card">
            📄 {st.session_state.nombre_documento}
            · Página {fuente["pagina"]}
            </div>
            """,
            unsafe_allow_html=True
        )

        mostrar_en_documento(fuente["pagina"], fuente["fragmentos"])


# ---------------------------------------------------------
# CONFIGURACIÓN
# ---------------------------------------------------------

st.set_page_config(
    page_title="Asistente Documental",
    page_icon="📚",
    layout="wide"
)


# ---------------------------------------------------------
# ESTILOS
# ---------------------------------------------------------

st.markdown(
    """
    <style>

    .block-container {
        padding-top: 3.5rem;
        padding-bottom: 3rem;
    }

    .main-title {
        font-size: 2.4rem;
        font-weight: 700;
        line-height: 1.25;
        margin-top: 0.5rem;
        margin-bottom: 0.25rem;
    }
    .subtitle {
        color: #777;
        font-size: 1rem;
        margin-bottom: 2rem;
    }

    .document-card {
        padding: 1rem;
        border: 1px solid rgba(128, 128, 128, 0.25);
        border-radius: 12px;
        margin-bottom: 1rem;
    }

    .source-card {
        padding: 0.65rem 0.9rem;
        border: 1px solid rgba(128, 128, 128, 0.25);
        border-radius: 8px;
        margin-bottom: 0.5rem;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ---------------------------------------------------------
# ESTADO DE SESIÓN
# ---------------------------------------------------------

if "chunks" not in st.session_state:
    st.session_state.chunks = None

if "nombre_documento" not in st.session_state:
    st.session_state.nombre_documento = None

if "numero_paginas" not in st.session_state:
    st.session_state.numero_paginas = 0

if "numero_fragmentos" not in st.session_state:
    st.session_state.numero_fragmentos = 0

if "mensajes" not in st.session_state:
    st.session_state.mensajes = []

if "archivo_actual" not in st.session_state:
    st.session_state.archivo_actual = None

if "pdf_bytes" not in st.session_state:
    st.session_state.pdf_bytes = None

if "total_paginas_pdf" not in st.session_state:
    st.session_state.total_paginas_pdf = 0


# ---------------------------------------------------------
# ENCABEZADO
# ---------------------------------------------------------

st.markdown(
    '<p class="main-title">📚 Asistente Documental</p>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <p class="subtitle">
    Consulta información contenida en documentos PDF mediante
    Inteligencia Artificial Generativa.
    </p>
    """,
    unsafe_allow_html=True
)


# ---------------------------------------------------------
# BARRA LATERAL
# ---------------------------------------------------------

with st.sidebar:

    st.header("📄 Documento")

    archivo = st.file_uploader(
        "Selecciona un archivo PDF",
        type=["pdf"]
    )

    # Detectar cambio de documento (nombre y tamaño, para distinguir
    # dos archivos diferentes que tengan el mismo nombre)
    if archivo is not None:

        identificador = (archivo.name, archivo.size)

        if (
            st.session_state.archivo_actual is not None
            and identificador != st.session_state.archivo_actual
        ):
            limpiar_documento()

        st.session_state.archivo_actual = identificador

        st.caption(
            f"Tamaño: {archivo.size / (1024 * 1024):.2f} MB"
        )

        if st.button(
            "⚙️ Procesar documento",
            use_container_width=True
        ):

            paginas = None

            with st.spinner("Extrayendo texto del documento..."):

                try:
                    paginas = extraer_paginas_pdf(archivo)
                except ErrorPDF as error:
                    limpiar_documento()
                    st.error(str(error))

            if paginas is None:
                # El error ya se mostró al intentar leer el PDF.
                pass

            elif not paginas:

                limpiar_documento()

                st.error(
                    "No se pudo extraer texto del documento. "
                    "Verifica que el PDF contenga texto seleccionable "
                    "y no esté compuesto únicamente por imágenes "
                    "escaneadas."
                )

            else:

                with st.spinner(
                    "Generando representaciones del documento..."
                ):

                    chunks = crear_chunks(paginas)

                    try:
                        chunks = generar_embeddings_chunks(chunks)
                    except Exception:
                        chunks = None

                if chunks is None:

                    limpiar_documento()

                    st.error(
                        "No fue posible generar las representaciones "
                        "del documento. La primera ejecución descarga "
                        "el modelo de embeddings, por lo que requiere "
                        "conexión a internet."
                    )

                else:

                    st.session_state.chunks = chunks
                    st.session_state.nombre_documento = archivo.name
                    st.session_state.numero_paginas = len(paginas)
                    st.session_state.numero_fragmentos = len(chunks)
                    st.session_state.mensajes = []
                    st.session_state.pdf_bytes = archivo.getvalue()
                    st.session_state.total_paginas_pdf = contar_paginas(
                        st.session_state.pdf_bytes
                    )

                    st.success("Documento listo.")

    else:

        st.session_state.archivo_actual = None


    # -----------------------------------------------------
    # INFORMACIÓN DEL DOCUMENTO
    # -----------------------------------------------------

    if st.session_state.chunks is not None:

        st.divider()

        st.subheader("Información")

        st.write(
            f"📄 **{st.session_state.nombre_documento}**"
        )

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Páginas",
                st.session_state.numero_paginas
            )

        with col2:
            st.metric(
                "Fragmentos",
                st.session_state.numero_fragmentos
            )

        st.success("🟢 Listo para consultas")

    st.divider()

    st.subheader("🤖 Modelo")

    st.write("**Mistral 7B**")
    st.caption("Ejecutado localmente mediante Ollama")

    st.info(
        "🔒 El procesamiento y la generación de respuestas "
        "se realizan localmente."
    )


# ---------------------------------------------------------
# PANTALLA INICIAL
# ---------------------------------------------------------

if st.session_state.chunks is None:

    st.subheader("👋 Bienvenido")

    st.write(
        "Para comenzar, selecciona un documento PDF desde "
        "la barra lateral y pulsa **Procesar documento**."
    )

    st.info(
        "El documento debe contener texto seleccionable. "
        "Los archivos compuestos únicamente por imágenes "
        "escaneadas no pueden procesarse en esta versión."
    )

    st.stop()


# ---------------------------------------------------------
# ÁREA DE CONSULTA
# ---------------------------------------------------------

st.subheader("💬 Consulta tu documento")

st.caption(
    "Realiza preguntas sobre el contenido del documento cargado."
)


# ---------------------------------------------------------
# VISOR DEL DOCUMENTO
# ---------------------------------------------------------

if st.session_state.pdf_bytes:

    with st.expander("📄 Ver documento"):

        total = st.session_state.total_paginas_pdf

        numero = st.number_input(
            f"Página (1 a {total})",
            min_value=1,
            max_value=total,
            value=1,
            step=1,
            key="visor_pagina"
        )

        with st.spinner("Cargando página..."):
            vista = pagina_resaltada(
                st.session_state.pdf_bytes,
                int(numero)
            )

        st.image(vista["pagina"])


# ---------------------------------------------------------
# HISTORIAL
# ---------------------------------------------------------

for mensaje in st.session_state.mensajes:

    with st.chat_message(mensaje["rol"]):

        if mensaje.get("error"):
            st.error(mensaje["contenido"])
        else:
            st.write(mensaje["contenido"])

        if (
            mensaje["rol"] == "assistant"
            and mensaje.get("fuentes")
        ):
            mostrar_fuentes(mensaje["fuentes"])


# ---------------------------------------------------------
# ENTRADA DEL USUARIO
# ---------------------------------------------------------

pregunta = st.chat_input(
    "Escribe una pregunta sobre el documento..."
)


if pregunta:

    # Guardar pregunta
    st.session_state.mensajes.append(
        {
            "rol": "user",
            "contenido": pregunta
        }
    )

    # Mostrar pregunta
    with st.chat_message("user"):
        st.write(pregunta)


    # -----------------------------------------------------
    # GENERAR RESPUESTA
    # -----------------------------------------------------

    with st.chat_message("assistant"):

        resultado = None
        fuentes_mostradas = []

        with st.spinner("Analizando el documento..."):

            try:
                resultado = responder_pregunta(
                    pregunta,
                    st.session_state.chunks
                )
            except ErrorModelo as error:
                mensaje_error = str(error)

        if resultado is None:

            st.error(mensaje_error)

            st.session_state.mensajes.append(
                {
                    "rol": "assistant",
                    "contenido": mensaje_error,
                    "fuentes": [],
                    "error": True
                }
            )

        else:

            st.write(resultado["respuesta"])

            for fuente in resultado["fuentes"]:

                if not any(
                    f["pagina"] == fuente["pagina"]
                    for f in fuentes_mostradas
                ):
                    fuentes_mostradas.append({
                        "pagina": fuente["pagina"],
                        "fragmentos": fuente.get(
                            "fragmentos", [fuente["texto"]]
                        )
                    })

            if fuentes_mostradas:
                mostrar_fuentes(fuentes_mostradas)
            else:
                st.info(
                    "No se identificaron fuentes suficientes "
                    "para responder la consulta."
                )

            # Guardar respuesta
            st.session_state.mensajes.append(
                {
                    "rol": "assistant",
                    "contenido": resultado["respuesta"],
                    "fuentes": fuentes_mostradas
                }
            )
