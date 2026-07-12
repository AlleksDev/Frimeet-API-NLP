# Busqueda global

## Fuente de verdad y flujo

```text
App movil + JWT
  -> API principal Go: POST /api/v1/search
  -> NLP: POST /internal/search/candidates + NLP_SERVICE_TOKEN
  -> pgvector/FTS: IDs y scores sobre umbral
  -> Go: permisos, reglas de negocio e hidratacion PostgreSQL
  -> App movil: recursos completos
```

- La API principal conserva identidad, permisos y objetos originales.
- NLP solo calcula embeddings, recuperacion, umbrales, ranking y cursores internos.
- PostgreSQL pgvector contiene indices derivados reconstruibles.
- La respuesta interna excluye metadata privada y campos para presentacion.
- `POST /search` es compatibilidad temporal y puede apagarse con
  `PUBLIC_GLOBAL_SEARCH_ENABLED=false`.

## Relevancia sin relleno

Cada recurso posee `semantic_min` y `lexical_min`. Un candidato se acepta cuando cumple
al menos una de estas condiciones:

```text
semantic_score >= semantic_min
OR lexical_score >= lexical_min
```

RRF ordena candidatos aceptados, pero no decide por si solo si existe coincidencia. El
limite solicitado es un maximo: cero coincidencias devuelve `[]`; una coincidencia
devuelve una. La diversificacion nunca recupera elementos bajo el umbral.

Variables:

```dotenv
GLOBAL_SEARCH_MIN_SEMANTIC_SCORE=0.30
GLOBAL_SEARCH_MIN_LEXICAL_SCORE=0.05
GLOBAL_SEARCH_RESOURCE_THRESHOLDS_JSON={}
GLOBAL_SEARCH_THRESHOLD_POLICY_VERSION=global-search-relevance-v1
```

Ejemplo de override completo:

```dotenv
GLOBAL_SEARCH_RESOURCE_THRESHOLDS_JSON={"posts":{"semantic_min":0.35,"lexical_min":0.05},"events":{"semantic_min":0.38,"lexical_min":0.06}}
```

Los valores iniciales deben recalibrarse con consultas positivas y negativas reales. El
umbral lexical corresponde a `ts_rank_cd`; no se debe copiar
`BM25_RELEVANCE_THRESHOLD`.

El repositorio incluye un calibrador reproducible que solo consume etiquetas y scores,
sin texto ni metadata privada. Cada linea JSONL debe usar este contrato:

```json
{"resource_type":"events","relevant":true,"semantic_score":0.62,"lexical_score":0.08}
```

Ejecucion:

```powershell
python scripts/calibrate_global_search_thresholds.py .\global_search_labels.jsonl `
  --policy-version global-search-relevance-v2 `
  --target-precision 0.95 `
  --output .\global_search_calibration.json
```

El resultado contiene `resource_thresholds` listo para copiar a
`GLOBAL_SEARCH_RESOURCE_THRESHOLDS_JSON` y metricas por recurso. La activacion sigue
bloqueada hasta alimentar el script con resultados reales de los seis recursos.

## Eventos

El endpoint interno recibe `as_of` desde Go y pgvector aplica en el CTE `eligible`, antes
del ranking:

```text
start_time + duration_minutes > as_of
```

Esto conserva eventos futuros y en curso, y elimina eventos terminados o con metadata
temporal invalida. Go repite la validacion con la tabla principal; NLP solo optimiza.

## Ubicacion, filtros y cursores

- `prioritize` agrega una senal separada de cercania.
- `strict` limita places, clubs y events a los `place_ids` resueltos por la API principal.
- Los filtros duros se ejecutan en SQL y se validan nuevamente en Python.
- Cada recurso tiene cursor independiente ligado a query, filtros, ubicacion, `as_of`,
  limite y version de la politica.
- Go encapsula el cursor NLP dentro de su cursor HMAC; la app nunca recibe el cursor
  interno sin esa proteccion.

## Migracion de una BD existente

Ejecutar con el propietario de `search_resource_embeddings(...)`:

1. `sql/migrations/20260712_01_global_search_candidate_filters.sql`.
2. `sql/verify_global_search_candidate_filters.sql`.

El primer archivo es el unico incremental de esta funcionalidad. No crea tablas,
columnas ni indices; usa `CREATE OR REPLACE FUNCTION`, conserva la firma, agrega helpers
de parseo seguro y vuelve a otorgar `EXECUTE` a `nlp_reader`. Sus postchecks funcionales
ocurren antes del `COMMIT`. El verificador posterior abre una transaccion de solo lectura
y aborta si encuentra contrato viejo, consultas bajo umbral con resultados, eventos
terminados o eventos activos con metadata temporal invalida.

No ejecutar `sql/new_pgvector_schema.sql` en produccion: es el esquema convergente para
inicializar desde cero. `sql/aws_pgvector_contract.sql` contiene el contrato completo de
referencia.

## Sincronizacion

Antes de activar la busqueda:

1. Ejecutar dry-run de `python -m app.jobs.sync_search_embeddings --resource all`.
2. Sincronizar todos los recursos.
3. Confirmar que events incluye `start_time` y `duration_minutes` validos.
4. Ejecutar `sql/verify_global_search_embeddings.sql`.
5. Ejecutar el verificador V2 indicado arriba.
6. Desplegar NLP y probar `/health`, `/ready` e `/internal/search/candidates`.
7. Activar `GLOBAL_SEARCH_ENABLED=true` en Go.

`/ready` devuelve `503` si la firma existe pero la definicion instalada no contiene el
contrato V2. Search interno tambien posee timeout total y las consultas pgvector usan el
mismo presupuesto. El rate limit local es una ultima defensa por proceso; en despliegues
con multiples replicas debe complementarse con el gateway.
