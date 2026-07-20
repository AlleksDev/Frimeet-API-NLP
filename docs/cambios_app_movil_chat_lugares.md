# Cambios requeridos en la app móvil para el chat de lugares

> Estado: pendiente fuera de este repositorio. Este documento describe cambios que deben implementarse en la app móvil. No se modificó código móvil como parte del ajuste del servicio NLP.

## Objetivo

La app debe presentar aclaraciones con sus textos localizados y conservar la identidad estructurada de cada opción al continuar la conversación. La app consume únicamente el contrato público de la API principal; nunca debe llamar directamente al servicio NLP.

## 1. Renderizado de aclaraciones

Cuando la respuesta incluya `action="clarification"`, la interfaz debe:

- mostrar `clarification.prompt` o el mensaje conversacional definido por la API principal;
- conservar el orden de `clarification.options`;
- usar `option.label` como texto visible de cada botón;
- mantener `option.id` únicamente como dato interno asociado al botón.

La app nunca debe mostrar `option.id` ni `value`, ni construir etiquetas reemplazando guiones bajos. Tampoco debe traducir categorías por su cuenta si la API principal ya entrega `label` y `message` localizados.

Ejemplo esperado:

```json
{
  "id": "religious_organization",
  "label": "Organización religiosa",
  "message": "Organización religiosa"
}
```

El botón debe mostrar `Organización religiosa`, no `religious_organization`.

## 2. Envío de la selección

Al tocar un botón, la app debe conservar la opción seleccionada y enviar a la API principal una elección estructurada:

```json
{
  "message": "Organización religiosa",
  "clarification_choice": {
    "clarification_id": "22222222-2222-4222-8222-222222222222",
    "option_id": "religious_organization"
  }
}
```

El texto de la burbuja del usuario puede tomarse de `option.message`, pero la selección se identifica exclusivamente con:

- `clarification.id` como `clarification_id`;
- `option.id` como `option_id`.

La app no debe:

- enviar el `label` o `message` traducido como `option_id`;
- resolver una opción por su posición en la lista;
- inferir la selección comparando texto libre;
- reconstruir IDs a partir de una etiqueta;
- reutilizar una opción perteneciente a una aclaración anterior.

La API principal conserva el estado conversacional y la allowlist. La app solo debe mantener los identificadores necesarios para enviar la elección del turno visible.

## 3. Estados de carga y selección

Después de tocar una opción, el botón debe quedar temporalmente deshabilitado para evitar envíos duplicados. Las opciones anteriores deben dejar de ser interactivas cuando llegue el siguiente turno o cuando la API informe que la aclaración es obsoleta.

Si la API principal responde con un conflicto por una selección vencida, la app debe descartar esos botones y mostrar el estado conversacional más reciente; no debe reintentar con el texto o con otro índice.

## 4. Manejo de `no_match`

Cuando la API principal devuelva `action="no_match"`, la app debe:

- mostrar el mensaje recibido;
- retirar las opciones de aclaración del turno anterior;
- limpiar cards o recomendaciones anteriores que pudieran confundirse con la respuesta actual;
- mantener disponible el campo de texto para que el usuario reformule su búsqueda;
- no fabricar categorías, cards ni mensajes alternativos a partir de IDs previos.

Este comportamiento también permite presentar respuestas amistosas ante saludos o mensajes que todavía no contienen una intención de búsqueda, sin mostrar opciones arbitrarias.

## 5. Responsabilidades que no pertenecen a móvil

La app no debe:

- consumir `POST /internal/places/chat` directamente;
- consultar `GET /api/v1/places/categories?lang=es` para corregir una respuesta que la API principal ya debe localizar;
- hidratar IDs de lugares;
- decidir vigencia, permisos o distancia;
- volver a filtrar resultados por categoría.

Estas responsabilidades pertenecen a la API principal y a NLP según el contrato de integración.

## Criterios de aceptación

- Ningún botón muestra valores como `ropa_barata` o `religious_organization` cuando existe un `label` localizado.
- Todos los botones renderizan exactamente `option.label` y conservan el orden recibido.
- Al seleccionar `Organización religiosa`, la petición envía `option_id="religious_organization"` y el `clarification_id` vigente.
- Dos opciones con textos iguales siguen siendo distinguibles porque la selección usa su ID, no texto ni posición.
- Una traducción o cambio de copy no modifica el identificador enviado al backend.
- Una selección no puede enviarse dos veces mientras el turno está en proceso.
- Ante una aclaración obsoleta, la app elimina sus botones y no intenta resolverla por texto.
- Una respuesta `no_match` elimina cards y opciones anteriores y muestra el mensaje actual.
- La app no llama directamente a NLP ni duplica la lógica de traducción, hidratación o filtrado de la API principal.

