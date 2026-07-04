# Busqueda global

## Responsabilidades

- Los modulos `users`, `clubs`, `groups` y `events` construyen su documento semantico y
  adaptan el contrato de la API principal.
- `modules/search` genera un embedding por consulta, ejecuta proveedores en paralelo y
  compone `top_results`, resultados paginados y cursores por seccion.
- La API principal conserva los recursos originales y la identidad autenticada.
- La API principal tambien conserva la verdad geografica. El servicio NLP obtiene un
  allowlist de `place_ids` desde `/api/v1/places/nearby`; no duplica coordenadas.
- PostgreSQL + pgvector contiene exclusivamente indices derivados reconstruibles.

## Ubicacion y filtros

- `location.mode=prioritize` agrega una senal de ranking separada para lugares cercanos.
  El score semantico original no se modifica.
- `location.mode=strict` limita places, clubs y events a los `place_ids` resueltos por la
  API principal.
- Los filtros son tipados y se aplican por recurso: ciudad/estado, categorias, precios,
  tags, fechas de posts, fechas de eventos, modalidad del club y roles de usuarios.
- Los filtros duros se ejecutan en PostgreSQL antes del ranking y se vuelven a validar en
  la aplicacion como defensa del contrato.
- Cada recurso entrega su propio cursor. El cursor queda vinculado a consulta, filtros,
  ubicacion, identidad y tamano de pagina.

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
7. Volver a ejecutar `python -m app.jobs.initial_load_post_embeddings` para incorporar
   `published_at` a la metadata de posts ya indexados.
8. Volver a ejecutar `python -m app.jobs.sync_search_embeddings --resource all` para
   refrescar filtros de recursos; `user_roles` requiere que el snapshot interno de
   usuarios incluya `role`.

El contrato SQL agrega `textsearch` a lugares y posts sin regenerar sus vectores. Las
tablas nuevas usan HNSW para FastText y GIN para texto/metadatos.
