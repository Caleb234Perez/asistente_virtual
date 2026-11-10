import json

import ollama


MODELO = "mistral:7b"


class ErrorModelo(Exception):
    """Error al comunicarse con Ollama o al interpretar su respuesta."""


def _chat(prompt, **opciones):
    try:
        return ollama.chat(
            model=MODELO,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            **opciones
        )

    except ollama.ResponseError as error:
        if error.status_code == 404:
            raise ErrorModelo(
                f"El modelo '{MODELO}' no está disponible en Ollama. "
                f"Descárgalo con: ollama pull {MODELO}"
            ) from error

        raise ErrorModelo(
            f"Ollama devolvió un error: {error.error}"
        ) from error

    except (ConnectionError, OSError) as error:
        raise ErrorModelo(
            "No fue posible conectarse con Ollama. "
            "Verifica que Ollama esté instalado y en ejecución."
        ) from error

    except Exception as error:
        # Versiones anteriores de la librería lanzan errores de httpx
        # cuando Ollama no está en ejecución.
        raise ErrorModelo(
            "No fue posible obtener respuesta del modelo. "
            "Verifica que Ollama esté en ejecución. "
            f"Detalle: {error}"
        ) from error


def generar_respuesta(prompt):
    respuesta = _chat(prompt)

    return respuesta["message"]["content"]


def generar_respuesta_estructurada(prompt):
    respuesta = _chat(prompt, format="json")

    contenido = respuesta["message"]["content"]

    try:
        datos = json.loads(contenido)
    except json.JSONDecodeError as error:
        raise ErrorModelo(
            "El modelo devolvió una respuesta con formato inválido. "
            "Intenta formular la pregunta nuevamente."
        ) from error

    if not isinstance(datos, dict):
        raise ErrorModelo(
            "El modelo devolvió una respuesta con estructura inesperada."
        )

    return datos