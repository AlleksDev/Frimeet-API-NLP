# Busqueda global

## Responsabilidades

- Los modulos `users`, `clubs`, `groups` y `events` construyen su documento semantico y
  adaptan el contrato de la API principal.
- `modules/search` genera un embedding por consulta, ejecuta proveedores en paralelo y
  compone `top_results` y resultados por seccion.
- La API principal conserva los recursos originales y la identidad autenticada.
- PostgreSQL + pgvector contiene exclusivamente indices derivados reconstruibles.

## Pesos semanticos

| Recurso | Campos |
| --- | --- |
| users | username x10, full_name x8, bio x2 |
| clubs | name x8, category x5, description x3 |
| groups | name x8, description x3 |
| events | title x8, nombres de tags x5, description x3 |

## Despliegue

1. Ejecutar `sql/aws_pgvector_contract.sql` con el rol propietario.
2. Conceder las funciones nuevas a `nlp_reader` y `nlp_writer` usando las sentencias
   comentadas al final del archivo SQL.
3. Configurar los cuatro paths `MAIN_API_*_SEARCH_PATH`.
4. Probar una pagina sin escribir:
   `python -m app.jobs.sync_search_embeddings --resource all --dry-run --max-pages 1`.
5. Ejecutar la carga: `python -m app.jobs.sync_search_embeddings --resource all`.
6. Verificar dimensiones, normas y funciones con
   `psql <cadena-de-conexion> -f sql/verify_global_search_embeddings.sql`.

El contrato SQL agrega `textsearch` a lugares y posts sin regenerar sus vectores. Las
tablas nuevas usan HNSW para FastText y GIN para texto/metadatos.
