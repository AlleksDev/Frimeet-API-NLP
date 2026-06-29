# Migracion Y Despliegue De FastText

Esta migracion cambia los vectores derivados de 16 dimensiones mock a embeddings
FastText reales de 300 dimensiones. No cambia ningun contrato HTTP.

## 1. Preparar Y Validar Localmente

```powershell
cd C:\Users\aleja\Desktop\Uni\9o\MID\C2\Frimeet-API-NLP
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m app.shared.nlp.embeddings.download_fasttext_model `
  --repo-id facebook/fasttext-es-vectors `
  --filename model.bin `
  --destination .models/fasttext-es/model.bin
```

Configura el `.env` local con:

```env
EMBEDDING_PROVIDER=fasttext
EMBEDDING_DIMENSION=300
EMBEDDING_MODEL=facebook/fasttext-es-vectors
EMBEDDING_VERSION=common-crawl-300-v1
FASTTEXT_MODEL_PATH=.models/fasttext-es/model.bin
FASTTEXT_MODEL_REPO_ID=facebook/fasttext-es-vectors
FASTTEXT_MODEL_FILENAME=model.bin
FASTTEXT_AUTO_DOWNLOAD=true
SEMANTIC_NO_MATCH_THRESHOLD=0.30
SEMANTIC_RELEVANCE_THRESHOLD=0.50
```

Antes de tocar la base, valida una pagina real sin escribir datos:

```powershell
.\.venv\Scripts\python.exe -m app.jobs.sync_place_embeddings `
  --dry-run --max-pages 1 --page-limit 5
.\.venv\Scripts\python.exe -m app.jobs.sync_post_embeddings `
  --dry-run --max-pages 1 --page-limit 5
```

Ambos comandos deben terminar con `errors=0`. Este ensayo comprueba la descarga y
carga de FastText, la lectura de la API principal y las credenciales de lectura del
contrato pgvector, pero no modifica RDS.

## 2. Publicar Codigo En GitHub

```powershell
git add .
git commit -m "Use Spanish FastText embeddings with pgvector"
git push origin hf-deploy
```

## 3. Pausar El Space Y Migrar RDS

Pausa el Space antes de cambiar la dimension; la API anterior genera vectores de 16
dimensiones y no puede consultar una columna `VECTOR(300)`.

Haz un snapshot de RDS y ejecuta con un usuario administrador o `nlp_owner`:

```powershell
psql "host=<host> port=5432 dbname=nlp_vectors user=<admin> sslmode=require" `
  -f sql/migrate_fasttext_300.sql
psql "host=<host> port=5432 dbname=nlp_vectors user=<admin> sslmode=require" `
  -f sql/aws_pgvector_contract.sql
```

La migracion trunca `place_embeddings` y `post_embeddings` porque son indices
derivados incompatibles. No toca la base transaccional de la API principal.

Para hacerlo desde pgAdmin y ejecutar la carga pesada desde Google Colab consulta
`docs/pgadmin_colab_fasttext.md`. Incluye un SQL unico para Query Tool y dos scripts
independientes de carga.

## 4. Repoblar PGVector

Con las credenciales writer en `.env`:

```powershell
.\.venv\Scripts\python.exe -m app.jobs.initial_load_place_embeddings
.\.venv\Scripts\python.exe -m app.jobs.initial_load_post_embeddings
psql "host=<host> port=5432 dbname=nlp_vectors user=<admin> sslmode=require" `
  -f sql/verify_fasttext_embeddings.sql
```

La verificacion debe mostrar filas, dimension 300, el modelo FastText configurado y
normas cercanas a 1.

## 5. Actualizar Hugging Face

En Settings del Space conserva `VECTOR_STORE_PROVIDER=aws_pgvector`, las credenciales
actuales de RDS/Groq y configura estas variables de embedding:

```env
EMBEDDING_PROVIDER=fasttext
EMBEDDING_DIMENSION=300
EMBEDDING_MODEL=facebook/fasttext-es-vectors
EMBEDDING_VERSION=common-crawl-300-v1
FASTTEXT_MODEL_PATH=/opt/models/fasttext-es/model.bin
FASTTEXT_MODEL_REPO_ID=facebook/fasttext-es-vectors
FASTTEXT_MODEL_FILENAME=model.bin
FASTTEXT_AUTO_DOWNLOAD=false
SEMANTIC_NO_MATCH_THRESHOLD=0.30
SEMANTIC_RELEVANCE_THRESHOLD=0.50
```

El Dockerfile descarga el modelo durante el build. Publica la rama local en `main`
del Space:

```powershell
git push hf hf-deploy:main
```

Cuando el build termine, reinicia el Space y revisa `/ready` y una consulta real a
`POST /places/recommendations`.

## Rollback

No intentes insertar vectores de 16 dimensiones en las columnas nuevas. Para volver
al motor anterior se requiere restaurar el snapshot o ejecutar una migracion inversa,
repoblar los indices de 16 dimensiones y desplegar el commit anterior.
