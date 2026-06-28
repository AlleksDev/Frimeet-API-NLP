# Migracion FastText Con pgAdmin Y Google Colab

## 1. Antes De Modificar RDS

1. Publica esta version en la rama `hf-deploy` de GitHub.
2. Crea un snapshot de RDS.
3. Pausa el Space de Hugging Face y cualquier job de sincronizacion.
4. No ejecutes todavia los cargadores de Colab.

## 2. Migrar Desde pgAdmin

En pgAdmin selecciona la base `nlp_vectors`, abre **Tools > Query Tool**, carga el
archivo `sql/pgadmin_migrate_fasttext_300.sql` y ejecutalo completo.

El script hace en una sola transaccion:

- valida que ambas columnas sigan siendo `VECTOR(16)`;
- elimina las funciones e indices incompatibles;
- trunca solamente `place_embeddings` y `post_embeddings`;
- cambia ambas columnas a `VECTOR(300)`;
- reconstruye HNSW, funciones y permisos;
- confirma al final la dimension y que ambas tablas quedaron vacias.

No lo ejecutes una segunda vez: esta protegido y abortara si ya encuentra
`VECTOR(300)`.

## 3. Preparar Secrets En Colab

En el panel **Secrets** de Colab agrega y habilita **Notebook access** para:

| Nombre | Requerido | Uso |
|---|---:|---|
| `PGVECTOR_WRITER_PASSWORD` | Si | Password del rol `nlp_writer`. |
| `MAIN_API_INTERNAL_TOKEN` | Solo si aplica | Token para leer lugares/posts de la API principal. |
| `HF_TOKEN` | No | Puede ayudar con la descarga, pero el modelo es publico. |

Los scripts incluyen los valores no secretos actuales del host, base, usuario y URL
principal. Tambien puedes sobrescribirlos con Secrets llamados
`PGVECTOR_HOST`, `PGVECTOR_DATABASE`, `PGVECTOR_WRITER_USER` y
`MAIN_API_BASE_URL`.

La URL base actual es `http://3.212.166.108`. No agregues `/api/v1` al valor de
`MAIN_API_BASE_URL`, porque los paths de lugares y publicaciones ya incluyen ese
segmento. Si existe un Secret `MAIN_API_BASE_URL` en Colab, actualizalo o eliminalo
para evitar que reemplace este valor.

## 4. Permitir Temporalmente La IP De Colab

RDS debe ser accesible desde el runtime. Obten la IP publica actual en una celda:

```python
!curl -s https://api.ipify.org
```

En el Security Group de RDS agrega temporalmente una regla de entrada TCP 5432 con
origen `<IP_OBTENIDA>/32`. No uses `0.0.0.0/0`. Elimina la regla cuando terminen
ambas cargas.

Si RDS es privado y no tiene una ruta publica, Colab no podra conectarse directamente;
en ese caso ejecuta los jobs desde una instancia dentro de la VPC.

## 5. Ejecutar Las Cargas

En un runtime de Colab con memoria suficiente:

```python
!git clone --branch hf-deploy https://github.com/AlleksDev/Frimeet-API-NLP.git
%cd Frimeet-API-NLP
```

Primero lugares:

```python
!python scripts/colab_initial_load_places.py
```

Se recomienda ejecutar los archivos con `!python` como arriba. Tambien puedes copiar
el contenido completo de cada script en una celda vacia y ejecutarlo directamente:
si no existe el repositorio, el propio script clonara `hf-deploy`. En ese modo ignora
los argumentos internos de Jupyter. Si no configuraste `PGVECTOR_WRITER_PASSWORD` en
Secrets, mostrara una entrada oculta para solicitarlo.

Despues publicaciones, reutilizando dependencias y el modelo ya descargado:

```python
!python scripts/colab_initial_load_posts.py --skip-install --skip-download
```

Para una prueba sin escrituras se pueden agregar `--dry-run --max-pages 1`. Para una
carga pequena real usa solamente `--max-pages 1`; despues puedes ejecutar otra vez
sin esa opcion y el job completara los registros faltantes. La ejecucion completa no
debe usar `--dry-run` ni `--max-pages`. El resultado correcto termina con `errors=0`
y el mensaje de carga terminada.

## 6. Verificar Desde pgAdmin

Despues de ambas cargas abre nuevamente Query Tool y ejecuta el contenido de
`sql/verify_fasttext_embeddings.sql`. Debe mostrar:

- filas mayores que cero;
- dimensiones minima y maxima iguales a `300`;
- modelo `facebook/fasttext-es-vectors`;
- version `common-crawl-300-v1`;
- normas cercanas a `1`.

Finalmente elimina la regla temporal del Security Group, actualiza las variables del
Space y despliega la nueva imagen.
