# Asistente Documental

Prototipo de asistente que responde preguntas en lenguaje natural sobre documentos PDF mediante recuperación aumentada por generación (RAG). Todo el procesamiento se realiza localmente: los embeddings con Sentence Transformers y las respuestas con Mistral 7B a través de Ollama.

## Requisitos

- Python 3.13
- [Ollama](https://ollama.com) instalado y en ejecución
- Conexión a internet solo la primera vez (para descargar los modelos)

## Instalación

```bash
# 1. Crear y activar el entorno virtual
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Descargar el modelo generativo
ollama pull mistral:7b
```

El PDF de prueba no se incluye en el repositorio. Para `main.py`, las pruebas que lo usan y la evaluación, colócalo en `data/documento_prueba.pdf`.

## Funciones principales

- Preguntas en lenguaje natural (español o inglés) sobre un PDF.
- Respuestas con las páginas que las respaldan.
- **Ver en el documento:** cada fuente muestra la página del PDF con el párrafo usado para responder resaltado en amarillo.
- **Visor del documento:** permite hojear el PDF mientras se hacen preguntas.
- Rechazo de preguntas cuya respuesta no está en el documento.

## Ejecución

Interfaz web:

```bash
streamlit run app.py
```

Versión de terminal (usa `data/documento_prueba.pdf`):

```bash
python main.py
```

## Pruebas

Pruebas automáticas (no requieren Ollama ni descargar modelos):

```bash
python -m pytest
```

Evaluación de la recuperación con el conjunto de preguntas de `evaluacion/preguntas.json`:

```bash
python -m evaluacion.evaluar            # solo recuperación
python -m evaluacion.evaluar --con-llm  # también respuestas (requiere Ollama)
```

Los resultados se guardan en `evaluacion/` como CSV (detalle por pregunta) y Markdown (resumen).

## Estructura

```
├── app.py                 Interfaz Streamlit
├── main.py                Versión de terminal
├── requirements.txt
├── data/                  Documento de prueba (no incluido en el repo)
├── src/
│   ├── pdf_reader.py      Extracción de texto del PDF
│   ├── text_processor.py  Fragmentación del texto
│   ├── embeddings.py      Representaciones vectoriales
│   ├── search.py          Búsqueda por similitud coseno
│   ├── llm.py             Comunicación con Ollama
│   └── rag.py             Integración recuperación + generación
├── tests/                 Pruebas automáticas (pytest)
└── evaluacion/            Conjunto de preguntas y script de evaluación
```

## Limitaciones

- Solo procesa PDFs con texto seleccionable (los escaneados requieren OCR).
- Ollama debe estar en ejecución para generar respuestas.
- Prototipo de un solo usuario; no está pensado para despliegue productivo.
