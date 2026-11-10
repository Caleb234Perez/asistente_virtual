import os
import sys
import types


# Permite importar "src" al ejecutar pytest desde la raíz del proyecto.
sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)


# Las pruebas no se comunican con Ollama: todas las llamadas al modelo
# se sustituyen por respuestas simuladas. Si la librería no está instalada
# (por ejemplo, en una máquina sin Ollama), se crea un módulo mínimo para
# poder importar src.llm.
try:
    import ollama  # noqa: F401
except ImportError:
    modulo = types.ModuleType("ollama")

    class ResponseError(Exception):
        def __init__(self, error, status_code=-1):
            super().__init__(error)
            self.error = error
            self.status_code = status_code

    def chat(*args, **kwargs):
        raise ConnectionError("Ollama no disponible en pruebas")

    modulo.ResponseError = ResponseError
    modulo.chat = chat
    sys.modules["ollama"] = modulo


RUTA_PDF_PRUEBA = os.path.join(
    os.path.dirname(__file__), "..", "data", "documento_prueba.pdf"
)
