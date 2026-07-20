# Cambios requeridos en la API principal para el chat de lugares

> Estado: pendiente fuera de este repositorio. Este documento describe cambios que deben implementarse en la API principal. No se modificó código Go como parte del ajuste del servicio NLP.

## Objetivo

La API principal debe conservar el ranking y las decisiones producidas por `POST /internal/places/chat`, hidratar únicamente recursos vigentes y presentar aclaraciones localizadas sin alterar los identificadores técnicos que protegen el flujo conversacional.

## 1. No volver a aplicar la categoría como filtro duro

NLP trata la categoría como evidencia de ranking, no como una restricción absoluta. Después de recibir los candidatos de NLP, la API principal no debe descartarlos mediante un `categoryMatches` estricto ni exigir que la categoría almacenada coincida literalmente con `target_category`.

La hidratación posterior a NLP puede aplicar estas restricciones duras:

- existencia y vigencia del lugar;
- permisos y reglas de visibilidad;
- alcance geográfico explícito y distancia final con PostGIS;
- cualquier otra regla de negocio que sea independiente de una coincidencia textual de categoría.

Una diferencia de categoría no debe eliminar por sí sola un candidato. Por ejemplo, una intención `cafe` puede recuperar un lugar registrado con una categoría compatible o más específica. NLP ya incorpora categoría, compatibilidad, contenido semántico, evidencia lexical y exclusiones en su puntuación.

Si `categoryMatches` no puede retirarse inmediatamente, debe ejecutarse temporalmente en modo de observación: registrar qué candidatos habría descartado, pero no modificar la respuesta. Su eliminación definitiva debe ocurrir después de comparar esos registros con el ranking de NLP.

## 2. Localizar las opciones de aclaración sin cambiar su identidad

La localización aplica exclusivamente cuando:

```text
clarification.kind = target_category | intent_category
```

No debe aplicarse a `location_scope`, `location_anchor`, `reference_entity` ni `reference_location_anchor`, porque sus etiquetas ya describen lugares o acciones concretas.

### Fuente de traducciones

La API principal ya expone:

```http
GET /api/v1/places/categories?lang=es
```

El catálogo devuelto contiene pares equivalentes a:

```json
{
  "value": "religious_organization",
  "label": "Organización religiosa"
}
```

La API principal debe reutilizar ese catálogo, preferentemente mediante la misma capa de aplicación o repositorio que atiende el endpoint, y puede almacenarlo en caché por idioma. Un error al obtener traducciones no debe hacer fallar el chat.

### Algoritmo de correspondencia

Para cada elemento de `clarification.options`:

1. Buscar por `id` la opción correspondiente en `state_patch.pending_clarification.options`.
2. Tomar su `value` técnico.
3. Buscar una entrada del catálogo cuyo `value` coincida exactamente con ese valor.
4. En el DTO dirigido a la app, reemplazar únicamente `label` y `message` por el `label` localizado.
5. Conservar el orden original de NLP.

Ejemplo de salida para la app:

```json
{
  "id": "religious_organization",
  "label": "Organización religiosa",
  "message": "Organización religiosa"
}
```

### Invariantes obligatorios

La localización nunca debe modificar:

- `clarification.id`;
- `clarification.options[].id`;
- `state_patch.pending_clarification.options[].id`;
- `state_patch.pending_clarification.options[].value`;
- el orden de las opciones;
- la allowlist que se persiste para validar el turno siguiente.

El estado técnico recibido desde NLP debe persistirse y reenviarse sin sustituir IDs o valores por textos traducidos. La traducción pertenece únicamente al DTO de presentación.

Si no existe una traducción, el fallback debe ser el `label` entregado por NLP. Como último recurso puede humanizarse el valor reemplazando guiones bajos por espacios; nunca debe usarse ese texto como `option_id`.

## 3. Selección estructurada

Cuando la app seleccione una opción, la API principal debe reenviar a NLP tanto el estado pendiente original como la selección estructurada:

```json
{
  "clarification_choice": {
    "clarification_id": "22222222-2222-4222-8222-222222222222",
    "option_id": "religious_organization"
  }
}
```

No debe intentar resolver la selección comparando el texto localizado, la posición del botón o el mensaje visible. Una selección inexistente u obsoleta debe conservar el manejo de conflicto del contrato interno.

## 4. Hidratación y observabilidad

La API principal es responsable de hidratar los IDs técnicos de NLP y de aplicar
vigencia, permisos y distancia.

El mensaje actual `No encontré lugares vigentes que cumplan con los criterios` se
construye después de que NLP ya devolvió candidatos y la API principal terminó con una
lista hidratada vacía. Por tanto, verlo no demuestra que `/nearby` haya encontrado cero
lugares: también puede significar que `categoryMatches`, vigencia, permisos o distancia
descartaron todos los IDs. Esta distinción debe conservarse en métricas y logs.

Para poder explicar una respuesta vacía debe emitir un registro estructurado,
correlacionado por `trace_id`, que incluya al menos:

- `conversation_id` y número de turno;
- `trace_id` de NLP;
- cantidad de candidatos recibidos desde NLP;
- cantidad de IDs encontrados durante la hidratación;
- cantidad final enviada a la app;
- conteos de descarte por causa: inexistente, inactivo o eliminado, sin permisos, fuera del radio y error de hidratación;
- cantidad que el antiguo `categoryMatches` habría descartado durante su periodo de observación.

No deben registrarse coordenadas exactas ni datos privados innecesarios. Los IDs pueden registrarse solo cuando la política operativa lo permita; los conteos y el `trace_id` son obligatorios.

Si NLP devuelve `action="recommendations"` pero ningún candidato sobrevive a las reglas autorizadas, la API principal debe producir un estado de `no_match` coherente para la app y registrar una razón interna como `empty_after_hydration`. No debe atribuir automáticamente el vacío a la categoría ni ocultar la causa operativa.

## 5. Ciclo de vida de lugares

La fuente utilizada para sincronizar Places con NLP y los endpoints de hidratación/nearby deben compartir la misma definición de vigencia.

El contrato de ciclo de vida debe cubrir:

- **snapshot completo:** debe tener un límite consistente, paginación estable y una señal inequívoca de finalización exitosa;
- **cierre o desactivación:** debe exponer `is_active=false` o un evento equivalente para que NLP deje de recomendar el lugar;
- **eliminación:** debe producir una baja o tombstone identificable, no limitarse a omitir silenciosamente el registro;
- **reconciliación:** los lugares ausentes solo pueden marcarse inactivos después de completar correctamente un snapshot total;
- **fallo o snapshot parcial:** nunca debe provocar una desactivación masiva ni considerarse una reconciliación válida;
- **consistencia:** `/api/v1/places/nearby`, la hidratación por IDs y el snapshot deben usar el mismo tipo de ID y las mismas reglas de vigencia.

Estos requisitos permiten que el filtro `is_active=true` de NLP represente el estado real y reducen los casos en que Go recibe IDs que ya no puede hidratar.

## Criterios de aceptación

- Un candidato relevante no se elimina solo porque su categoría almacenada no coincide literalmente con `target_category`.
- `categoryMatches` deja de modificar la lista final; mientras exista en observación, solo genera métricas.
- Para una opción con `value="religious_organization"`, la app recibe `label="Organización religiosa"` y el mismo `id` técnico.
- La selección posterior reenvía el `clarification_id` y `option_id` originales, aunque el usuario haya visto un texto traducido.
- La localización conserva orden, valores y allowlist, y no modifica aclaraciones geográficas o de referencias.
- Si el catálogo de traducciones no está disponible, el chat continúa con un fallback legible.
- Cada respuesta vacía después de hidratación puede investigarse mediante `trace_id`, conteos y razones de descarte.
- Un lugar cerrado o eliminado deja de aparecer después de una sincronización completa exitosa.
- Un snapshot parcial o fallido no desactiva lugares que siguen vigentes.
