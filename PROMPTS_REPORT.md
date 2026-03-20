# Prompt Report

Este documento resume las prompts del experimento actual, cómo se usan dentro del protocolo few-shot, qué diferencias reales hay entre ellas y qué convendría mejorar antes de escalar experimentos o comparar modelos por API.

## Contexto Compartido

Las cinco variantes comparten la misma estructura general:

1. Un `system prompt` que define el rol y las restricciones.
2. `K` ejemplos por clase en formato few-shot.
3. Cada ejemplo few-shot usa:
   - `user`: imagen + `"What is the class of this image?"`
   - `assistant`: `<response>label</response>`
4. La query final usa:
   - una imagen nueva
   - una instrucción textual que fuerza una de las `N` clases vistas
5. La evaluación solo mira el contenido dentro de `<response>...</response>`.

Eso significa que la diferencia experimental real no está en los datos ni en el protocolo de evaluación, sino en la forma de pedirle al modelo que razone antes de clasificar.

## Plantilla Few-Shot Compartida

### Ejemplos de soporte

```text
User:
- [image]
- What is the class of this image?

Assistant:
<response>{label}</response>
```

### Query final

Siempre se le pide elegir exactamente una clase de una lista cerrada:

```text
You MUST choose exactly one from [{options_str}]
```

Eso es bueno porque reduce bastante la ambigüedad del output y hace más robusta la evaluación automática.

## 1. Classification

### System Prompt

```text
You are a high-precision image classifier agent. Your task consists of performing a few-shot classification. You will see:
N different classes of entities to classify,
K samples of each class,
Q query images to be classified according to the seen classes.

1. Analyze the provided input images and their ground truth labels for each category. Pay close attention to distinct visual features (form, shape, texture, color).
2. Examine the Target Image and compare it strictly against the provided input examples.
3. Reason step-by-step to determine which category the Target Image belongs to.

Constraints:
- Use ONLY the labels provided in the final options list.
- Do not make assumptions outside the visual evidence.
- Do not use your prior knowledge to classify entities (breeds, species, or objects). Rely entirely on the visual features of the examples.

Output Format:
The label of this new image in format XML <response>output_class</response>
```

### Query Template

```text
Based on the examples seen, what is the class of this image? You MUST choose exactly one from [{options_str}]
```

### Qué intenta medir

Es la baseline más limpia: clasificación few-shot con el mínimo extra de estructura.

### Ventajas

- Es la variante más simple y más fácil de interpretar.
- Tiene menos riesgo de que el modelo “se pierda” generando explicaciones largas.
- Es la mejor referencia para comparar si las explicaciones ayudan o perjudican.

### Desventajas

- Pide “reason step-by-step” pero no obliga a mostrar ese razonamiento.
- Esa instrucción puede inducir razonamiento latente, pero no deja trazabilidad.
- Si el modelo usa conocimiento previo igual, después es difícil auditarlo.

### Mejoras posibles

- Si querés baseline realmente estricta, sacar `Reason step-by-step`.
- Si querés baseline explicable, pedir una justificación mínima estructurada de una línea.
- Agregar una cláusula más fuerte tipo: `If uncertain, still choose one of the provided labels and do not invent labels.`

## 2. NLE

### System Prompt

```text
You are a high-precision Explainable Vision-Language Agent. Your task consists of performing a few-shot classification and explaining your reasoning in natural language.
Task:
1. Analyze the provided input images and their ground truth labels for each category. Pay close attention to distinct visual features (form, shape, texture, color).
2. Examine the Target Image and compare it strictly against the provided input examples.
3. Reason step-by-step to determine which category the Target Image belongs to. Write a clear, concise explanation in natural language detailing why the visual features of the target image match the selected class.
Constraints:
- Use ONLY the labels provided in the final options list.
- Do not make assumptions outside the visual evidence.
- Do not use your prior knowledge to classify entities; rely entirely on the visual features of the examples.
Output Format:
Your natural language explanation first, and then the label of this new image in format XML <response>output_class</response>
```

### Query Template

```text
Based on the examples seen, what is the class of this image? You MUST choose exactly one from [{options_str}].
WARNING: For this final target image, remember to provide your step-by-step Natural Language Explanation FIRST, and then wrap your final classification in <response>output_class</response> tags.
```

### Qué intenta medir

Si obligar al modelo a verbalizar una explicación en lenguaje natural mejora o empeora la clasificación.

### Ventajas

- Es legible y fácil de inspeccionar manualmente.
- Sirve bien para detectar si el modelo está atendiendo a rasgos razonables.
- Es una forma natural de auditar errores.

### Desventajas

- “Natural language explanation” es muy abierta.
- Puede producir explicaciones plausibles pero post-hoc.
- No fuerza una estructura estable entre corridas y modelos.
- Mezcla explicación libre con clasificación, lo que puede aumentar longitud y costo sin ganar precisión.

### Mejoras posibles

- Forzar una plantilla más corta:
  - `Observed cues: ...`
  - `Contrast with alternatives: ...`
  - `<response>...</response>`
- Limitar la explicación a 2-3 frases.
- Pedir evidencia comparativa explícita entre clases candidatas, no solo descripción de la elegida.

## 3. Features

### System Prompt

```text
You are a high-precision image classifier and feature extractor agent.
Task:
1. Analyze the provided input images and their ground truth labels for each category. Pay close attention to distinct visual features (form, shape, texture, color).
2. Examine the Target Image and compare it strictly against the provided input examples.
3. Extract and list the minimum number of critical, concrete, and observable visual features that distinguish the Target Image from previously seen classes.
- "Features" must refer only to visible physical components or structural elements (e.g., "wheels", "handle", "screen", "wings"), not attributes such as color, size, length, texture, or shape.
- Use short noun phrases only.
- Include only features necessary for differentiation.
- Do NOT use abstract explanations.
4. Based strictly on those extracted features, determine the category of the Target Image.
Constraints:
- Use ONLY the labels provided in the final options list.
- Do not use prior knowledge outside the visual evidence.
Output Format Instructions:
While the few-shot examples only provide the final class, your response for the new target image must be fully expanded.
Structure your output strictly as follows:
Features: List the concrete observable features as bullet points.
Classification: You MUST wrap your final chosen class label in XML tags exactly like this: <response>output_class</response>
```

### Query Template

```text
Based on the examples seen, what is the class of this image? You MUST choose exactly one from [{options_str}].
WARNING: For this final target image, remember to extract the critical visual features FIRST. Structure your response EXACTLY like this:
Features:
[Concrete observable feature 1]
[Concrete observable feature 2]
<response>output_class</response>
```

### Qué intenta medir

Si obligar al modelo a pasar por una etapa explícita de extracción de rasgos visuales mejora la decisión.

### Ventajas

- Introduce una estructura más concreta que NLE.
- Empuja al modelo a usar evidencia visual observable.
- Es más fácil de comparar entre outputs que una explicación libre.

### Desventajas

- Hay una inconsistencia interna importante:
  - al principio dice que mire color, textura y shape;
  - después prohíbe usar color, tamaño, longitud, textura y shape como “features”.
- Los ejemplos incluidos en la query (`e.g., Specific color/texture`) contradicen el system prompt.
- “Componentes físicos” funciona bien para objetos, pero peor para flores, texturas o razas donde muchas diferencias son de patrón, forma o color.

### Mejoras posibles

- Separar claramente:
  - `parts/components`
  - `surface patterns`
  - `global shape cues`
  - `color cues`
- Para datasets como `flowers` y `dtd`, permitir explícitamente patrón, simetría, textura y distribución cromática.
- Reemplazar la regla actual por algo más general:
  - `Use only directly observable visual cues; do not use taxonomy or world knowledge.`

## 4. Rulebased

### System Prompt

```text
You are an Explainable Vision-Language agent specialized in feature-based Classification.

Task:
1. Analyze the provided input images and their ground truth labels for each category. Pay close attention to distinct visual features (form, shape, texture, color).
2. Examine the Target Image and compare it strictly against the provided input examples.
3. Feature Extraction Protocol: Extract the minimum number of critical, concrete, and observable visual features that clearly distinguish the Target Image from all previously seen classes. Do not use abstract words.
4. Knowledge Base: Based on the few-shot examples, formulate at least one IF-THEN rule per class learned mapping specific visual features that explain that class.
5. Decision Logic (Rule Check): Evaluate the Target Image's extracted features against your Knowledge Base rules step-by-step to determine which rule matches the visual features best.
6. Final Classification: Based strictly on the rule check, determine the class and rule activated to produce the corresponding feature-based explanation.

Constraints:
- Use ONLY the labels provided in the final options list.
- Do not make assumptions outside the visual evidence.
- Do not use your prior knowledge to classify entities. Rely entirely on the examples.

Output Format Instructions:
While the few-shot examples only provide the final class, your response for the new target image must be fully expanded. Structure your output strictly as follows:
Features: List the extracted concrete visual features as bullet points.
KB (Knowledge Base): Formulate the logical IF-THEN rules derived from the examples.
Rule Check: Explain step-by-step how the extracted features match the rules in your KB.
Classification: You MUST wrap your final chosen class label in XML tags exactly like this: <response>output_class</response>
```

### Query Template

```text
Based on the examples seen, what is the class of this image? You MUST choose exactly one from [{options_str}].
WARNING: For this final target image, remember to provide the full logical explanation FIRST. Structure your response EXACTLY like this:
Features:
- [Feature 1]
- [Feature 2]
KB:
- IF [Features] THEN [Class]
Rule Check:
- [Rules from the KB that matches the visual features best]
<response>output_class</response>
```

### Qué intenta medir

Si estructurar el razonamiento como reglas explícitas tipo `IF-THEN` ayuda al modelo a discriminar mejor.

### Ventajas

- Hace visible una forma de razonamiento intermedia.
- Es más auditable que NLE.
- La parte `Rule Check` puede revelar si el modelo está comparando clases o solo justificando una decisión ya tomada.

### Desventajas

- “At least one IF-THEN rule per class” con muy pocos shots puede inducir sobreajuste narrativo.
- Puede inventar reglas demasiado fuertes a partir de muy poca evidencia.
- Mezcla tres tareas difíciles:
  - extraer features,
  - inducir reglas,
  - clasificar.
- El costo en tokens crece bastante.

### Mejoras posibles

- Pedir reglas tentativas y no definicionales:
  - `Propose provisional visual rules supported by the examples.`
- Forzar contraste:
  - `For each candidate class, state one supporting cue and one conflicting cue.`
- Limitar cantidad de reglas:
  - `At most one short rule per class.`
- Pedir un score de match por clase en vez de explicación larga.

## 5. Axioms / Ontology

### System Prompt

Versión resumida de lo que hoy se le pide:

```text
You are an expert Explainable Vision-Language Agent ... explain classification using Description Logics (DL) axioms according to a knowledge base (KB) built from the few-shot examples.

It explains:
- Necessary Condition (C ⊑ P)
- Sufficient Condition (P ⊑ C)
- Necessary & Sufficient (C ≡ P)
- TBox
- ABox
- OWL object and data properties

Final response should include:
1. Class in <response>{output_class}</response>
2. TBox axioms
3. ABox assertions
4. Logical explanation matching ABox against TBox
```

### Query Template

```text
Analyze this new image using Description Logics (DL) axioms according to the knowledge base (KB). You MUST choose exactly one class from this list: [{options_str}].
WARNING: DO NOT use generic placeholders. Extract REAL visible features acceptable in your internal KB’s TBox, and express them accordingly in the ABox.
Structure your response EXACTLY like this:
Entity Identification:
<response>{output_class}</response>
TBox (axioms in a Knowledge Base):
- class ⊑ hasVisualFeature [feature] (Necessary condition)
- hasVisualFeature [feature] ⊑ class (Sufficient condition)
- class ≡ hasVisualFeature [feature] (Necessary & Sufficient condition)
ABox (Observed Properties & Assertions):
hasVisualFeature [specific shape/texture observed in this image]
Description Logics’ Ontological Axioms
The ABox assertion presents [Properties/Assertions], which satisfies the TBox [necessary/sufficient/necessary & sufficient] condition [axiom] to classify it as [class].
```

### Qué intenta medir

Si una estructura de razonamiento ontológico, mucho más formal, cambia la performance o la calidad explicativa.

### Ventajas

- Es la condición más fuerte en términos de explicitud estructural.
- Puede ser interesante como experimento de interpretabilidad.
- Hace muy visible cuándo el modelo está “rellenando” formalismo sin anclaje visual.

### Desventajas

- Es probablemente la prompt con más carga cognitiva y más riesgo de alucinación formal.
- Pide DL/OWL/TBox/ABox aunque la evidencia few-shot es mínima.
- Los ejemplos conceptuales tipo pizza/triangle/centaur están muy lejos del dominio real del experimento.
- Incentiva fuertemente la invención de pseudo-ontologías que suenan bien pero no están sustentadas.
- Es la menos comparable con una tarea natural de clasificación visual.

### Mejoras posibles

- Bajar mucho la ambición formal.
- Reemplazar OWL/DL duro por una versión “soft schema”:
  - `Class hypothesis`
  - `Observed cues`
  - `Necessary-like cues`
  - `Most diagnostic cue`
- Si querés seguir con ontologías, usar un mini lenguaje controlado más simple y consistente con visión:
  - `Class -> diagnostic visual cues`
  - `Observed instance -> cues present`
  - `Decision -> best-matching class`

## Comparación General

## Complejidad

- `classification`: baja
- `nle`: baja-media
- `features`: media
- `rulebased`: media-alta
- `axioms_ontology_v2`: muy alta

## Riesgo de alucinación explicativa

- `classification`: bajo
- `nle`: medio
- `features`: medio
- `rulebased`: medio-alto
- `axioms_ontology_v2`: alto

## Facilidad de auditar errores

- `classification`: baja
- `nle`: media
- `features`: alta
- `rulebased`: alta
- `axioms_ontology_v2`: media

## Costo esperado en tokens

- `classification`: bajo
- `nle`: medio
- `features`: medio
- `rulebased`: alto
- `axioms_ontology_v2`: muy alto

## Calidad experimental como comparación limpia

- `classification`: muy buena
- `nle`: buena
- `features`: buena si se corrige la inconsistencia
- `rulebased`: aceptable, pero ya mezcla varias operaciones
- `axioms_ontology_v2`: débil como benchmark limpio, interesante como condición exploratoria

## Problemas Transversales

### 1. Inconsistencia entre prompts

No todas están pidiendo exactamente el mismo tipo de evidencia visual. Algunas privilegian color/textura/shape, otras los restringen, y otras los convierten en pseudo-propiedades ontológicas.

### 2. Demasiada carga en el system prompt

Especialmente en `rulebased` y `axioms`, el system prompt es muy largo. Eso puede:

- aumentar costo,
- meter ruido,
- hacer más frágil la adherencia al formato,
- perjudicar modelos más chicos.

### 3. Falta de calibración por dataset

No todos los datasets se benefician del mismo tipo de “feature language”.

- `cifar10`: objetos globales, funciona bien con partes y forma.
- `pets`: rasgos finos, pelaje, morfología, cara, orejas.
- `flowers`: color, forma de pétalos, disposición.
- `dtd`: textura y patrón, donde “componentes físicos” no aplica bien.

### 4. Falta de una variante “compact explain”

Hay salto grande entre:

- baseline casi muda,
- explicación libre,
- esquema de features,
- reglas,
- ontología pesada.

Falta una condición intermedia, corta y estable.

## Recomendación Práctica

Si el objetivo es comparar prompts de forma seria y después migrar a OpenRouter con costo razonable, yo priorizaría esta jerarquía:

1. Mantener `classification` como baseline principal.
2. Refinar `features` para que sea consistente con todos los datasets.
3. Mantener `nle` como condición explicativa simple.
4. Dejar `rulebased` como condición secundaria de razonamiento estructurado.
5. Tratar `axioms_ontology_v2` como experimento exploratorio, no como condición principal.

## Versión Mejorada Recomendada

Si después querés limpiar el benchmark, yo propondría una familia así:

- `classification_strict`
  - solo clase final
- `classification_brief_justification`
  - 1-2 frases breves
- `classification_visual_cues`
  - 2-4 cues observables, sin world knowledge
- `classification_comparative`
  - por qué esta clase y no las otras
- `classification_rulecheck`
  - versión compacta de reglas, sin ontología

## Conclusión

Hoy las prompts forman una buena progresión de “más estructura explicativa”, pero no todas están igualmente bien calibradas.

La más sólida como baseline es `classification`.
La más prometedora para interpretabilidad útil es `features`, si se corrige la inconsistencia sobre qué cuenta como feature.
`rulebased` puede servir si se compacta.
`axioms_ontology_v2` es interesante como stress test de formalismo, pero demasiado pesada y propensa a pseudo-explicaciones para usarla como condición central.
