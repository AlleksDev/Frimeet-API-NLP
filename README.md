# Independent NLP Service

Servicio backend NLP independiente para busqueda semantica, recomendaciones, ranking, embeddings, ChromaDB y redaccion de respuestas conversacionales con Llama via Groq.

Este servicio esta pensado para ser consumido por una API principal mediante REST. La app movil no debe llamar directamente a este servicio. La API principal mantiene autenticacion, usuarios, sesiones, permisos, reportes de respuestas y logica general de producto.

## 1. Descripcion del proyecto

El servicio NLP apoya una aplicacion movil para planear salidas con amigos, familia o pareja y recomendar lugares o publicaciones relacionadas. La primera region objetivo es Chiapas, Mexico, con posibilidad de crecer a todo Mexico.

La primera version queda lista como un modular monolith en FastAPI: modular por recursos, mantenible, preparada para produccion moderada y sin Docker por ahora.

## 2. Objetivo del servicio NLP

Responsabilidades principales:

- Procesar y normalizar texto de entrada.
- Generar embeddings de queries de usuario.
- Consultar ChromaDB como vector store.
- Recomendar y rankear lugares.
- Recomendar publicaciones.
- Exponer clusters de publicaciones previamente calculados.
- Usar Llama via Groq solo para embellecer la respuesta final del chat.

Fuera de alcance:

- Autenticacion de usuarios.
- Sesiones.
- Reportes de respuestas por usuarios.
- Permisos de producto.
- Llamadas directas desde la app movil.
- Generacion masiva de embeddings durante requests normales.
- Clustering de publicaciones durante requests normales.

## 3. Arquitectura general

La arquitectura es un modular monolith service:

- `places`: busqueda, recomendaciones, chat y ranking de lugares.
- `posts`: recomendaciones, clusters y ranking de publicaciones.
- `shared`: capacidades reutilizables de NLP, ChromaDB, configuracion, cache, errores, logging y seguridad.
- `jobs`: tareas pesadas o batch que se ejecutan fuera del flujo normal de requests.

Dentro de cada modulo se usa una arquitectura limpia ligera:

- `api`: routers y schemas Pydantic.
- `application/use_cases`: casos de uso.
- `application/ports`: interfaces hacia dependencias externas.
- `domain`: modelos del dominio.
- `infrastructure`: implementaciones concretas o mocks iniciales.

Los casos de uso no dependen directamente de ChromaDB, Groq, Llama ni proveedores concretos.

## 4. Estructura de carpetas

```text
app/
├── main.py
├── modules/
│   ├── places/
│   │   ├── api/
│   │   ├── application/
│   │   │   ├── use_cases/
│   │   │   └── ports/
│   │   ├── domain/
│   │   └── infrastructure/
│   └── posts/
│       ├── api/
│       ├── application/
│       │   ├── use_cases/
│       │   └── ports/
│       ├── domain/
│       └── infrastructure/
├── shared/
│   ├── nlp/
│   │   ├── embeddings/
│   │   ├── preprocessing/
│   │   ├── llm/
│   │   └── prompts/
│   ├── chroma/
│   ├── config/
│   ├── cache/
│   ├── errors/
│   ├── logging/
│   └── security/
└── jobs/
```

## 5. Flujo de busqueda de lugares

Endpoint:

```http
POST /places/search
```

Flujo:

1. Recibe `query`, ciudad, estado y filtros.
2. Limpia y normaliza el texto.
3. Genera embedding solo del query del usuario.
4. Consulta la coleccion de lugares en ChromaDB.
5. Aplica filtros por metadata cuando existan:
   - `city`
   - `state`
   - `category`
   - `price_range`
   - `is_active`
   - `occasion`
6. Rankea candidatos.
7. Devuelve lugares estructurados.

Importante: los embeddings de lugares se generan offline con jobs y se guardan en ChromaDB. La request solo genera el embedding del query.

## 6. Flujo de chat con Llama via Groq

Endpoint:

```http
POST /places/chat
```

Flujo:

1. Recibe mensaje, ciudad/region y filtros.
2. Preprocesa el texto.
3. Genera embedding del mensaje.
4. Busca lugares reales en ChromaDB o repositorio configurado.
5. Aplica ranking.
6. Construye un contexto limitado con candidatos reales.
7. Envia a Llama via Groq solo la intencion, region aproximada y lugares candidatos.
8. Llama redacta una respuesta conversacional.
9. Se valida la salida con una capa simple de guardrails.
10. La API responde con:
    - `response_id`
    - `nlp_trace_id`
    - `message`
    - `places`
    - `metadata`

Llama no decide recomendaciones, no hace busqueda y no debe inventar lugares. Las cards de la app deben salir del arreglo estructurado `places`, no del texto libre.

## 7. Flujo de embeddings con ChromaDB

La capa compartida esta en:

```text
app/shared/chroma/
app/shared/nlp/embeddings/
```

Hay una interfaz `EmbeddingProvider` con:

- `embed_text(text: str) -> list[float]`
- `embed_batch(texts: list[str]) -> list[list[float]]`

La version inicial usa `MockEmbeddingProvider`, deterministico y util para desarrollo/tests. En produccion se puede reemplazar por un provider real sin cambiar los casos de uso.

ChromaDB esta preparado para modo servidor via HTTP:

```text
CHROMA_HOST=localhost
CHROMA_PORT=8000
CHROMA_PLACES_COLLECTION=places_collection
CHROMA_POSTS_COLLECTION=posts_collection
```

Las colecciones de lugares y publicaciones deben mantenerse separadas.

## 8. Flujo de clustering de publicaciones

Endpoint:

```http
GET /posts/clusters
```

Este endpoint solo lee clusters ya calculados. El clustering no se ejecuta durante una request normal.

El job responsable es:

```bash
python -m app.jobs.rebuild_post_clusters
```

La implementacion inicial es un placeholder listo para conectar una fuente real de embeddings y persistencia de clusters.

## 9. Variables de entorno

Crear `.env` a partir de `.env.example`:

```bash
cp .env.example .env
```

Variables principales:

```text
ENV
API_HOST
API_PORT
GROQ_API_KEY
GROQ_MODEL
CHROMA_HOST
CHROMA_PORT
CHROMA_PLACES_COLLECTION
CHROMA_POSTS_COLLECTION
LOG_LEVEL
REQUEST_TIMEOUT_SECONDS
LLM_TIMEOUT_SECONDS
MAX_LLM_CONCURRENT_REQUESTS
```

`.env` esta incluido en `.gitignore` y no debe subirse al repositorio.

## 10. Instalar dependencias

Crear entorno virtual:

```bash
python -m venv .venv
```

Activar en Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Activar en macOS/Linux:

```bash
source .venv/bin/activate
```

Instalar dependencias:

```bash
pip install -r requirements.txt
```

## 11. Ejecutar la API sin Docker

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8080 --reload
```

Endpoints de sistema:

```http
GET /health
GET /ready
```

Documentacion local:

```text
http://localhost:8080/docs
```

## 12. Ejecutar ChromaDB en modo servidor

El servicio FastAPI usa `chromadb-client` para conectarse por HTTP y evitar acoplar cada worker a almacenamiento embebido. Para levantar el servidor ChromaDB local, instala el paquete completo de ChromaDB en un entorno separado o en una maquina preparada para ese proceso:

```bash
python -m venv .venv-chroma
source .venv-chroma/bin/activate
pip install -r requirements-chroma-server.txt
```

En Windows PowerShell:

```powershell
python -m venv .venv-chroma
.\.venv-chroma\Scripts\Activate.ps1
pip install -r requirements-chroma-server.txt
```

Luego ejecutar:

```bash
chroma run --host localhost --port 8000
```

Nota para Windows: si el paquete completo `chromadb` intenta compilar `chroma-hnswlib`, puede requerir Microsoft C++ Build Tools o una version de Python con wheel disponible. La API no necesita ese paquete completo para correr; solo lo necesita el proceso servidor de ChromaDB.

En esta version inicial los endpoints usan repositorios mock por defecto para poder desarrollar y probar sin una instancia real. Para conectar Chroma real, reemplaza las dependencias en:

```text
app/modules/places/api/dependencies.py
app/modules/posts/api/dependencies.py
```

por repositorios `ChromaPlaceVectorRepository` y `ChromaPostVectorRepository` usando `ChromaVectorStore`.

## 13. Ejecutar jobs

Reconstruir embeddings de lugares:

```bash
python -m app.jobs.rebuild_place_embeddings
```

Reconstruir embeddings de publicaciones:

```bash
python -m app.jobs.rebuild_post_embeddings
```

Reconstruir clusters de publicaciones:

```bash
python -m app.jobs.rebuild_post_clusters
```

Los jobs actuales son placeholders ejecutables. La intencion es conectar ahi la base de datos fuente, generar embeddings por lotes y hacer upsert a ChromaDB sin cargar trabajo pesado en requests normales.

## 14. Endpoints disponibles

```http
GET  /health
GET  /ready
POST /places/search
POST /places/recommendations
POST /places/chat
POST /posts/recommendations
GET  /posts/clusters
```

Ejemplo de search:

```json
{
  "query": "lugares tranquilos para cenar",
  "city": "Tuxtla Gutierrez",
  "filters": {
    "category": "restaurant",
    "is_active": true
  },
  "limit": 5
}
```

Ejemplo de chat:

```json
{
  "message": "quiero una cena tranquila con mi pareja",
  "city": "Tuxtla Gutierrez",
  "filters": {
    "occasion": "pareja",
    "is_active": true
  },
  "limit": 5
}
```

## 15. Consideraciones de seguridad

- No exponer API keys.
- No subir `.env`.
- La app movil nunca debe llamar directamente a Groq ni a este servicio NLP.
- Minimizar datos enviados a Groq.
- No enviar emails, telefonos, nombres completos ni ubicacion exacta si no es necesario.
- Hay middleware base para limitar tamano de request.
- Hay placeholder de rate limiting listo para evolucionar.
- Las llamadas a Groq usan timeout y limite de concurrencia.

## 16. Consideraciones de escalabilidad

- FastAPI usa endpoints async.
- Los modelos/providers se instancian una vez mediante dependencias cacheadas.
- No se generan embeddings masivos durante requests.
- ChromaDB esta pensado como servicio separado.
- `/places/search` funciona sin LLM.
- `/places/chat` usa LLM y esta protegido con timeout, concurrencia y fallback.
- Jobs offline manejan embeddings masivos y clustering.
- Hay cache TTL simple para busquedas frecuentes de lugares.

## 17. Por que Llama solo embellece respuestas

Las recomendaciones deben venir del sistema NLP:

- embeddings,
- similitud vectorial,
- filtros por metadata,
- ranking,
- datos reales en ChromaDB.

Llama solo redacta una respuesta mas amable y natural con los lugares ya seleccionados. Esto evita que el modelo invente lugares, horarios, precios, promociones o datos que no existen en el contexto.

## 18. Reportes de respuestas desde la API principal

El boton de reportar respuesta vive en la app movil, pero el reporte debe enviarse a la API principal.

Este servicio NLP solo devuelve metadatos para trazabilidad:

- `response_id`
- `nlp_trace_id`
- modelo usado
- proveedor LLM
- lugares usados como contexto
- timestamp

No se reporta automaticamente a Groq cuando un usuario reporta una respuesta. Los reportes deben quedar en el sistema del producto para revision interna y mejora de prompts, filtros y ranking.

## 19. Proximos pasos recomendados

- Conectar `ChromaPlaceVectorRepository` y `ChromaPostVectorRepository` en las dependencias.
- Crear jobs reales que lean lugares/posts desde la fuente oficial.
- Elegir provider real de embeddings y mantener la interfaz actual.
- Persistir trazas NLP en storage interno.
- Agregar autenticacion interna entre API principal y servicio NLP.
- Implementar rate limiting real por API principal, tenant o client id.
- Agregar observabilidad: request id, metricas, latencias y errores por provider.
- Endurecer guardrails del LLM con validacion mas estricta sobre entidades permitidas.
- Ampliar tests de integracion con una instancia ChromaDB de desarrollo.

## Tests

```bash
pytest
```

Los tests iniciales cubren:

- `/health`
- provider mock de embeddings
- busqueda de lugares con providers mock
- chat de lugares con LLM mock
- respuesta de chat con lugares estructurados
- fallback cuando falla el LLM
