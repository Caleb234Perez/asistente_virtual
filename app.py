import streamlit as st

from src.pdf_reader import extraer_paginas_pdf, ErrorPDF
from src.text_processor import crear_chunks
from src.embeddings import generar_embeddings_chunks
from src.rag import responder_pregunta
from src.llm import ErrorModelo


def limpiar_documento():
    st.session_state.chunks = None
    st.session_state.nombre_documento = None
    st.session_state.numero_paginas = 0
    st.session_state.numero_fragmentos = 0
    st.session_state.mensajes = []


def mostrar_fuentes(paginas):
    st.caption("Fuentes")

    for pagina in paginas:
        st.markdown(
            f"""
            <div class="source-card">
            📄 {st.session_state.nombre_documento}
            · Página {pagina}
            </div>
            """,
            unsafe_allow_html=True
        )


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
        paginas_fuente = []

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

                pagina = fuente["pagina"]

                if pagina not in paginas_fuente:
                    paginas_fuente.append(pagina)

            if paginas_fuente:
                mostrar_fuentes(paginas_fuente)
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
                    "fuentes": paginas_fuente
                }
            )
