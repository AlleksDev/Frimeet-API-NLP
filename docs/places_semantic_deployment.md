# Backfill y activacion del retriever semantico de Places

Esta guia despliega exclusivamente el retriever de Places. El modelo BERT de
intencion se configura por separado con `PLACES_CHAT_BERT_*` y no participa en
el backfill de `place_embeddings_semantic_v1`.

## Regla principal de seguridad

El proveedor, la dimension y las funciones SQL deben cambiar como un solo
perfil. Nunca mezcles un embedding de 768 dimensiones con las funciones
FastText de 300 dimensiones.

| Etapa | Proceso | Proveedor | Dimension | Lectura | Escritura |
| --- | --- | --- | ---: | --- | --- |
| Durante backfill | Space productivo | `fasttext` | 300 | `match_places` | funciones legacy |
| Durante backfill | Colab | `sentence_transformer` | 768 | no se usa | funciones `*_semantic_v1` |
| Despues de verificar | Space productivo | `sentence_transformer` | 768 | funciones `*_semantic_v1` | funciones `*_semantic_v1` |

Si el Space ya tiene `PLACES_EMBEDDING_PROVIDER=sentence_transformer` y todavia
usa `PLACES_PGVECTOR_MATCH_FUNCTION=match_places`, pausalo o vuelve
temporalmente al perfil FastText antes de desplegar el codigo nuevo.

## 1. Identificar exactamente el modelo

En el repositorio del retriever de Hugging Face abre **Files and versions**,
entra al historial del commit publicado y copia el SHA completo de 40
caracteres. No uses `main`: el contenido de `main` puede cambiar.

Conserva estos dos valores:

```text
MODEL_ID=tu-usuario/places-e5-retriever-v1
MODEL_SHA=0123456789abcdef0123456789abcdef01234567
```

`PLACES_EMBEDDING_MODEL_REVISION` controla los archivos exactos descargados.
`PLACES_EMBEDDING_VERSION` es la etiqueta que queda visible en PostgreSQL. Se
recomienda `places-e5-retriever-v1@<MODEL_SHA>`.

## 2. Preparar Colab

No se necesita GPU para el backfill. Con CPU usa lotes pequenos; una GPU solo
reduce el tiempo.

En **Colab > Secrets** crea y habilita el acceso al notebook para:

- `PGVECTOR_WRITER_PASSWORD`: password del rol `nlp_writer`.
- `MAIN_API_INTERNAL_TOKEN`: solo si la API principal protege el snapshot.
- `HF_TOKEN`: solo es obligatorio si el repositorio del retriever es privado.

No pongas passwords o tokens directamente en una celda.

Clona la rama o commit que contiene esta implementacion:

```python
!git clone --branch RAMA_CON_ESTOS_CAMBIOS \
  https://github.com/AlleksDev/Frimeet-API-NLP.git \
  /content/Frimeet-API-NLP
%cd /content/Frimeet-API-NLP
!git rev-parse HEAD
```

Configura solamente valores no secretos. Sustituye cada ejemplo por el valor
real del entorno:

```python
import os

MODEL_ID = "tu-usuario/places-e5-retriever-v1"
MODEL_SHA = "0123456789abcdef0123456789abcdef01234567"

os.environ["PLACES_EMBEDDING_MODEL"] = MODEL_ID
os.environ["PLACES_EMBEDDING_MODEL_REVISION"] = MODEL_SHA
os.environ["PLACES_EMBEDDING_VERSION"] = f"places-e5-retriever-v1@{MODEL_SHA}"
os.environ["PLACES_EMBEDDING_FIX_MISTRAL_REGEX"] = "true"
os.environ["PLACES_EMBEDDING_DEVICE"] = "cpu"
os.environ["PLACES_EMBEDDING_BATCH_SIZE"] = "8"

os.environ["MAIN_API_BASE_URL"] = "https://URL-DE-TU-API-PRINCIPAL"
os.environ["MAIN_API_PLACES_SEARCH_PATH"] = "/api/v1/places/search"
os.environ["MAIN_API_PLACES_PAGINATION_MODE"] = "cursor"

os.environ["PGVECTOR_HOST"] = "HOST-DE-RDS"
os.environ["PGVECTOR_PORT"] = "5432"
os.environ["PGVECTOR_DATABASE"] = "nlp_vectors"
os.environ["PGVECTOR_WRITER_USER"] = "nlp_writer"
os.environ["PGVECTOR_SSL_MODE"] = "require"
```

Si RDS restringe IPs, autoriza temporalmente la IP publica de ese runtime de
Colab como una regla `/32`. Nunca abras `0.0.0.0/0` y elimina la regla al
terminar.

## 3. Ensayo sin escritura

El modo `--semantic` configura automaticamente el proveedor de 768d, los
prefijos E5 y las funciones SQL de la tabla nueva:

```python
!python scripts/colab_initial_load_places.py \
  --semantic \
  --dry-run \
  --max-pages 1 \
  --page-limit 5 \
  --batch-size 5
```

El ensayo es correcto cuando termina con `errors=0`. En `--dry-run`,
`upserted=0` es esperado: los vectores se calculan, pero no se escriben.

Errores habituales:

- `401` o `403`: falta `MAIN_API_INTERNAL_TOKEN` o no tiene acceso al snapshot.
- timeout a RDS: falta la regla `/32`, la ruta de red o el puerto 5432.
- `Repository Not Found`: el ID es incorrecto o falta `HF_TOKEN` para un repo privado.
- dimension distinta de 768: se subio el artefacto equivocado o la migracion no coincide.
- error de funciones legacy: faltan las dos variables de escritura semantica; el modo
  `--semantic` las establece automaticamente.

## 4. Backfill completo

Reutiliza las dependencias ya instaladas:

```python
!python scripts/colab_initial_load_places.py \
  --semantic \
  --skip-install \
  --page-limit 50 \
  --batch-size 16
```

No agregues `--max-pages`. Una primera carga correcta de una tabla vacia termina
con valores equivalentes a:

```text
Finished place sync processed=N skipped=0 upserted=N errors=0
```

El job trabaja por lotes y es idempotente. Si Colab se desconecta, ejecuta el
mismo comando otra vez; no trunques la tabla. Una segunda corrida sin cambios
debe mostrar aproximadamente `processed=N`, `skipped=N`, `upserted=0`,
`errors=0`.

## 5. Verificar PostgreSQL

Ejecuta `sql/verify_places_semantic_v1.sql` desde pgAdmin con un rol de lectura
sobre la tabla. El archivo falla si la tabla esta vacia y muestra conteos,
normas, modelo, version y version del documento.

Confirma:

- `rows > 0` y `rows = unique_ids`.
- `rows` coincide con `processed=N` del snapshot completo.
- `min_norm` y `max_norm` estan cerca de `1.0`.
- aparece un solo `embedding_model`, igual a `MODEL_ID`.
- aparece una sola `embedding_version`, igual a
  `places-e5-retriever-v1@MODEL_SHA`.
- la version del documento es la esperada por el codigo actual.
- el indice HNSW y las funciones semanticas existen.

Con pocas filas PostgreSQL puede elegir un escaneo secuencial aunque HNSW exista;
eso no significa por si solo que el indice este roto.

## 6. Activar el Space

En **Settings > Variables and secrets > Variables** cambia el perfil completo
en una sola ventana:

```text
PLACES_EMBEDDING_PROVIDER=sentence_transformer
PLACES_EMBEDDING_DIMENSION=768
PLACES_EMBEDDING_MODEL=tu-usuario/places-e5-retriever-v1
PLACES_EMBEDDING_MODEL_REVISION=SHA_COMPLETO_DE_40_CARACTERES
PLACES_EMBEDDING_FIX_MISTRAL_REGEX=true
PLACES_EMBEDDING_VERSION=places-e5-retriever-v1@SHA_COMPLETO_DE_40_CARACTERES
PLACES_EMBEDDING_QUERY_PREFIX=query:
PLACES_EMBEDDING_PASSAGE_PREFIX=passage:
PLACES_EMBEDDING_BATCH_SIZE=8
PLACES_EMBEDDING_DEVICE=cpu
PLACES_PGVECTOR_MATCH_FUNCTION=match_places_semantic_v1
PLACES_PGVECTOR_HYBRID_FUNCTION=search_places_semantic_v1
PLACES_PGVECTOR_UPSERT_FUNCTION=upsert_place_embedding_semantic_v1
PLACES_PGVECTOR_HASH_FUNCTION=get_place_content_hashes_semantic_v1
```

La aplicacion agrega el espacio separador de `query:` y `passage:` aunque la UI
de Hugging Face recorte espacios finales.

`PLACES_EMBEDDING_FIX_MISTRAL_REGEX=true` aplica al tokenizer del retriever el
ajuste indicado por Transformers. Debe permanecer igual en evaluacion,
backfill y produccion para que los mismos textos produzcan los mismos tokens.

Mantiene en **Secrets**:

- `HF_TOKEN`, si cualquiera de los modelos es privado.
- passwords de PostgreSQL.
- `MAIN_API_INTERNAL_TOKEN`, `NLP_SERVICE_TOKEN`, `GROQ_API_KEY` y cualquier
  otra credencial.

Reinicia el Space y espera a que termine el build.

Hugging Face documenta que las conexiones salientes de Spaces se permiten por
los puertos 80, 443 y 8080. Si `PGVECTOR_PORT=5432`, confirma primero que la
ruta actual realmente funciona con `/ready`. Si el puerto esta bloqueado, el
cutover no puede completarse con una conexion directa: se necesita una ruta TCP
segura por un puerto permitido o ejecutar la API en infraestructura con acceso
a RDS.

## 7. Smoke tests obligatorios

1. `GET /health` debe responder HTTP 200.
2. `GET /ready` debe responder HTTP 200 y reportar ejecutables
   `match_places_semantic_v1` y `search_places_semantic_v1`.
3. Haz una consulta real para forzar la descarga y carga diferida del modelo.
4. Prueba `/places/chat` con lenguaje no literal, por ejemplo:
   `unas donitas chidas para llevar` y
   `algo tranqui pa platicar, sin musica fuerte`.
5. Revisa los logs del Space: no debe haber errores de autenticacion de HF,
   dimension, memoria ni funciones SQL.

`/ready` por si solo no prueba el modelo porque la carga es diferida.

Ejemplo de busqueda publica, sustituyendo la URL:

```bash
curl -X POST "https://TU-SPACE.hf.space/places/search" \
  -H "Content-Type: application/json" \
  -d '{"query":"unas donitas chidas para llevar","limit":5}'
```

Si `PLACES_CHAT_V2_ENABLED=true`, fuerza el flujo semantico del chat enviando un
`conversation_id`:

```bash
curl -X POST "https://TU-SPACE.hf.space/places/chat" \
  -H "Content-Type: application/json" \
  -d '{"message":"algo tranqui pa platicar, sin musica fuerte","conversation_id":"11111111-1111-4111-8111-111111111111","limit":5}'
```

## 8. Rollback

No borres `place_embeddings_semantic_v1`. Para volver atras, restaura de forma
coordinada:

```text
PLACES_EMBEDDING_PROVIDER=fasttext
PLACES_EMBEDDING_DIMENSION=300
PLACES_EMBEDDING_MODEL=facebook/fasttext-es-vectors
PLACES_EMBEDDING_MODEL_REVISION=
PLACES_EMBEDDING_FIX_MISTRAL_REGEX=true
PLACES_EMBEDDING_VERSION=common-crawl-300-v1
PLACES_EMBEDDING_QUERY_PREFIX=
PLACES_EMBEDDING_PASSAGE_PREFIX=
PLACES_PGVECTOR_MATCH_FUNCTION=match_places
PLACES_PGVECTOR_HYBRID_FUNCTION=
PLACES_PGVECTOR_UPSERT_FUNCTION=upsert_place_embedding
PLACES_PGVECTOR_HASH_FUNCTION=get_place_content_hashes
```

Reinicia y repite los smoke tests. Como la migracion es aditiva, la tabla
semantica queda disponible para diagnostico o un nuevo intento.

## Referencias oficiales

- SentenceTransformers, parametro `revision` y prefijos de retrieval:
  <https://www.sbert.net/docs/package_reference/sentence_transformer/model.html>
- Hugging Face Spaces, variables, secrets y puertos de red permitidos:
  <https://huggingface.co/docs/hub/main/spaces-overview>
