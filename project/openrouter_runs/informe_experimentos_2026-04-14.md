---
title: "Informe de Experimentos: In-Context Learning con Modelos de Lenguaje Grandes"
subtitle: "Evaluación mediante OpenRouter — Resultados al 14 de abril de 2026"
date: "14 de abril de 2026"
author: "Equipo de Investigación"
---

# Informe de Experimentos: In-Context Learning con Modelos de Lenguaje Grandes

**Fecha:** 14 de abril de 2026  
**Proyecto:** `research-llms-in-context-learning-explained`  
**Pipeline:** OpenRouter Experiment Framework  

---

## 1. Introducción

Este informe documenta la totalidad de los experimentos ejecutados hasta la fecha en el marco del proyecto de investigación sobre *in-context learning* (ICL) en modelos de lenguaje grandes (LLMs). El objetivo general del proyecto es comprender cómo distintas formas de estructurar los prompts few-shot afectan la capacidad de clasificación de los modelos, con especial énfasis en si presentar información de distinto tipo en el contexto —etiquetas simples, explicaciones en lenguaje natural, rasgos visuales, reglas de decisión, axiomas ontológicos— modifica de manera sistemática el rendimiento.

Los experimentos se realizaron sobre el dataset **flowers**, un conjunto de imágenes de flores con categorías de clasificación bien definidas. La evaluación se llevó adelante mediante el pipeline propio del proyecto sobre la plataforma **OpenRouter**, que permite acceder a distintos LLMs con capacidad multimodal a través de una API unificada.

Durante el período comprendido entre el 13 y el 14 de abril de 2026 se lanzaron en total **ocho ejecuciones** (*runs*). Siete completaron la totalidad de los trials planificados; la octava fue interrumpida durante su ejecución, completando únicamente el primer modelo de los cuatro previstos. Los primeros runs de la sesión del 13 de abril tuvieron carácter de depuración: permitieron identificar y corregir dos problemas de configuración —un límite de tokens insuficiente para ciertos tipos de prompt y una instrucción de formato de respuesta subespecificada— que afectaron los resultados iniciales pero fueron resueltos antes de escalar los experimentos al día siguiente.

A continuación se describen la metodología empleada, los resultados de cada ejecución en orden cronológico y un análisis comparativo entre modelos y condiciones experimentales.

---

## 2. Metodología

### 2.1. Framework experimental

El pipeline experimental es propio y está implementado en Python. El punto de entrada es `run_openrouter_experiment.py`, que lee un archivo de configuración JSON (`experiment_config.json`) y orquesta la ejecución de trials, el logging y el análisis estadístico post-hoc.

Cada ejecución (*run*) genera un directorio con la siguiente estructura de artefactos:

- `trial_results.csv` — una fila por trial con etiqueta esperada, etiqueta predicha, corrección, latencia y uso de tokens.
- `trial_logs.jsonl` — logs completos de cada conversación (payload enviado y respuesta raw del modelo).
- `run_accuracy_long.csv` — accuracy desagregado por dataset, tipo de prompt, configuración few-shot y repetición.
- `experiment_summary.csv` — resumen agregado por tipo de prompt.
- `analysis/` — tablas estadísticas (intervalos de Wilson, McNemar, Wilcoxon, Friedman) y gráficos PNG.

### 2.2. Dataset

**Flowers:** dataset de imágenes de flores. Las clasificaciones se presentan al modelo mediante un esquema few-shot: se muestran *K* ejemplos de soporte por categoría, tomados de *N* categorías, y luego se formula una pregunta sobre un ejemplo de consulta.

### 2.3. Tipos de prompt evaluados

Se evaluaron cinco condiciones de prompt, que varían en el tipo de información contextual provista al modelo en los ejemplos few-shot:

| Clave | Descripción |
|---|---|
| `classification` | Etiquetas de clase simples (el estándar ICL básico) |
| `nle` | Natural Language Explanations — justificaciones en prosa de cada etiqueta |
| `features` | Descripción de rasgos visuales relevantes de cada ejemplo |
| `rulebased` | Reglas de decisión explícitas derivadas de los rasgos |
| `axioms_ontology_v2` | Axiomas en estilo ontológico (más formal y estructurado) |

### 2.4. Configuraciones few-shot

Se exploraron distintas combinaciones de parámetros:

- **N** (número de categorías en el soporte): 2, 3 o 4
- **K** (ejemplos por categoría): 2 (fijo en todos los runs)
- **Q** (queries por episodio): 1 (fijo en todos los runs)
- **Repeticiones por configuración**: 6 (N=2), 4 (N=3), 3 (N=4) — diseñado para balancear el número total de trials

### 2.5. Modelos evaluados

Los modelos usados a través de OpenRouter fueron:

| Modelo | Proveedor | Tipo |
|---|---|---|
| `google/gemini-2.5-flash` | Google | Multimodal, alta capacidad |
| `google/gemini-2.5-flash-lite` | Google | Multimodal, rápido y económico |
| `google/gemma-4-26b-a4b-it` | Google | Open-weight, 26B activos (MoE) |
| `qwen/qwen3.5-9b` | Alibaba/Qwen | Open-weight, 9B parámetros |
| `mistralai/ministral-8b-2512` | Mistral AI | Open-weight, 8B parámetros |
| `bytedance-seed/seed-2.0-mini` | ByteDance | Propietario compacto |

### 2.6. Métricas

- **Accuracy** por tipo de prompt y configuración (con intervalo de confianza Wilson al 95%).
- **Latencia por trial** (segundos de wall-clock).
- **Uso de tokens** (prompt + completion).
- Pruebas estadísticas post-hoc: **McNemar** (comparaciones por pares), **Wilcoxon** (sobre accuracy por repetición), **Friedman** (comparación global entre tipos de prompt).

---

## 3. Resultados por ejecución

A continuación se describe cada ejecución en orden cronológico estricto de inicio.

---

### Run 1 — `test_run__20260413_125229` (Gemini 2.5 Flash — primera prueba)

| Campo | Valor |
|---|---|
| **Fecha y hora** | 13 de abril de 2026, 12:52:29 |
| **Modelo** | `google/gemini-2.5-flash` |
| **Configuración** | N=2, K=2, Q=1, 4 repeticiones |
| **Total de trials** | 20 (5 tipos × 4 trials) |

Primera ejecución de toda la sesión. Sirvió como prueba inicial del pipeline con el modelo más capaz de la familia Flash.

**Resultados de accuracy:**

| Tipo de prompt | Correctos / Total | Accuracy | Duración (s) |
|---|---|---|---|
| `classification` | 4 / 4 | 1.0000 | 7.02 |
| `nle` | 3 / 4 | 0.7500 | 31.62 |
| `features` | 4 / 4 | 1.0000 | 6.85 |
| `rulebased` | 4 / 4 | 1.0000 | 11.92 |
| `axioms_ontology_v2` | 0 / 4 | 0.0000 | 9.27 |
| **TOTAL** | **15 / 20** | **0.7500** | — |

**Observaciones:** Los 4 fallos de `axioms_ontology_v2` (0%) tienen una causa concreta y uniforme: el modelo generaba correctamente el razonamiento estructurado en formato ontológico (bloques `<tbox>`, `<abox>` y `<dl_explanation>`), pero no incluía el tag `<response>ClassXX</response>` al final. Al revisar los logs de conversación, se observa que la respuesta terminaba con la explicación en lenguaje natural y no llegaba a emitir la etiqueta requerida para el parseo. Como consecuencia, `predicted_label` quedaba vacío en todos los trials de esa condición. Tras identificar este comportamiento, el system prompt fue ajustado para reforzar explícitamente que el tag `<response>` debe ser siempre la última línea de la respuesta, incluso cuando el prompt requiere bloques de razonamiento previos. El fallo en un trial de `nle` también puede atribuirse a una respuesta que no siguió el formato exacto en esa primera configuración del prompt.

---

### Run 2 — `test_run__20260413_140847` (Gemini 2.5 Flash — prompt corregido)

| Campo | Valor |
|---|---|
| **Fecha y hora** | 13 de abril de 2026, 14:08:47 |
| **Modelo** | `google/gemini-2.5-flash` |
| **Configuración** | N=2, K=2, Q=1, 4 repeticiones |
| **Total de trials** | 20 |

Re-ejecución del Run 1 con el system prompt actualizado para enfatizar el tag de respuesta obligatorio.

**Resultados de accuracy:**

| Tipo de prompt | Correctos / Total | Accuracy |
|---|---|---|
| `classification` | 4 / 4 | 1.0000 |
| `nle` | 4 / 4 | 1.0000 |
| `features` | 4 / 4 | 1.0000 |
| `rulebased` | 4 / 4 | 1.0000 |
| `axioms_ontology_v2` | 4 / 4 | 1.0000 |
| **TOTAL** | **20 / 20** | **1.0000** |

**Observaciones:** Resultado perfecto en todos los tipos de prompt, incluyendo `axioms_ontology_v2` que había fallado por completo en el run anterior. La corrección del prompt eliminó completamente el problema de parseo: en todos los trials de este run el modelo finalizó su respuesta con el tag `<response>` en el formato esperado. La corrección se aplicó de manera global y se mantuvo en todos los runs posteriores.

---

### Run 3 — `test_run_qwen__20260413_181411` (Qwen 3.5-9B — límite de tokens insuficiente)

| Campo | Valor |
|---|---|
| **Fecha y hora** | 13 de abril de 2026, 18:14:11 |
| **Modelo** | `qwen/qwen3.5-9b` |
| **Configuración** | N=2, K=2, Q=1, 5 repeticiones |
| **Total de trials** | 50 |

Primera ejecución con el modelo Qwen 3.5-9B. Los resultados presentaron un patrón de fallos sistemático directamente correlacionado con el límite de tokens (`max_tokens`) de cada tipo de prompt en la biblioteca de prompts en ese momento.

**Resultados de accuracy:**

| Tipo de prompt | Correctos / Total | Accuracy | `max_tokens` configurado | Trials cortados (`length`) |
|---|---|---|---|---|
| `classification` | 0 / 10 | 0.0000 | 256 | 10 / 10 |
| `nle` | 4 / 10 | 0.4000 | 512 | 6 / 10 |
| `features` | 10 / 10 | 1.0000 | 512 | 0 / 10 |
| `rulebased` | 8 / 10 | 0.8000 | 768 | 2 / 10 |
| `axioms_ontology_v2` | 9 / 10 | 0.9000 | 1024 | 1 / 10 |
| **TOTAL** | **31 / 50** | **0.6200** | — | — |

**Observaciones:** La causa de todos los fallos en este run es la misma: el modelo Qwen 3.5-9B genera razonamientos verbosos antes de emitir el tag `<response>`, alcanzando típicamente entre 380 y 430 tokens por trial (verificado en el run siguiente). Con `max_tokens=256` para `classification`, la generación era truncada antes de llegar al tag de respuesta en el 100% de los casos (`finish_reason: length`). La correlación es perfecta: a mayor `max_tokens`, mayor accuracy, y los únicos fallos en cada condición son trials donde `finish_reason` fue `length`. La condición `features` es la excepción: aunque también tenía `max_tokens=512`, sus respuestas son más concisas y el modelo alcanzaba el stop sin truncamiento. Tras identificar el problema, el límite fue unificado a 1024 tokens para todos los tipos de prompt en los runs subsiguientes.

---

### Run 4 — `test_run_gemini__20260413_182459` (Gemini 2.5 Flash Lite — primera prueba)

| Campo | Valor |
|---|---|
| **Fecha y hora** | 13 de abril de 2026, 18:24:59 |
| **Modelo** | `google/gemini-2.5-flash-lite` |
| **Configuración** | N=2, K=2, Q=1, 5 repeticiones |
| **Total de trials** | 50 (5 tipos × 10 trials) |
| **Schema** | `openrouter_experiment_v1` |

Primera ejecución con el modelo Gemini 2.5 Flash Lite, simultánea a los experimentos del 13 de abril por la tarde, ya con el prompt corregido pero aún con la configuración de `max_tokens` por tipo de prompt (no unificados a 1024).

**Resultados de accuracy:**

| Tipo de prompt | Correctos / Total | Accuracy | Duración (s) | Latencia media (s) |
|---|---|---|---|---|
| `classification` | 10 / 10 | 1.0000 | 14.64 | 1.35 |
| `nle` | 10 / 10 | 1.0000 | 17.25 | 1.62 |
| `features` | 10 / 10 | 1.0000 | 15.02 | 1.40 |
| `rulebased` | 10 / 10 | 1.0000 | 19.08 | 1.80 |
| `axioms_ontology_v2` | 9 / 10 | 0.9000 | 22.01 | 2.09 |
| **TOTAL** | **49 / 50** | **0.9800** | — | — |

**Observaciones:** Rendimiento casi perfecto. A diferencia de Qwen, Gemini Flash Lite genera respuestas más concisas y no es sensible al límite de tokens variable. El único fallo ocurrió en `axioms_ontology_v2`, el tipo de prompt más exigente en formato. La latencia fue muy baja (por debajo de los 2 s/trial), lo que posiciona a este modelo como el más rápido del conjunto evaluado.

---

### Run 5 — `test__20260413_184321` (Qwen 3.5-9B — max_tokens corregido)

| Campo | Valor |
|---|---|
| **Fecha y hora** | 13 de abril de 2026, 18:43:21 |
| **Modelo** | `qwen/qwen3.5-9b` |
| **Configuración** | N=2, K=2, Q=1, 5 repeticiones |
| **Total de trials** | 50 |

Re-ejecución con Qwen 3.5-9B con `max_tokens` unificado a 1024 para todos los tipos de prompt.

**Resultados de accuracy:**

| Tipo de prompt | Correctos / Total | Accuracy | Tokens usados (media) | `finish_reason: stop` |
|---|---|---|---|---|
| `classification` | 10 / 10 | 1.0000 | ~400 | 10 / 10 |
| `nle` | 9 / 10 | 0.9000 | ~420 | 10 / 10 |
| `features` | 10 / 10 | 1.0000 | ~390 | 10 / 10 |
| `rulebased` | 10 / 10 | 1.0000 | ~450 | 10 / 10 |
| `axioms_ontology_v2` | 9 / 10 | 0.9000 | ~480 | 10 / 10 |
| **TOTAL** | **48 / 50** | **0.9600** | — | — |

**Observaciones:** Con tokens suficientes, el modelo completa normalmente (`finish_reason: stop` en el 100% de los trials) y el accuracy sube de 62% a 96%. Los dos fallos restantes (uno en `nle`, uno en `axioms_ontology_v2`) son errores de clasificación genuinos: el modelo terminó su generación, incluyó el tag `<response>`, pero predijo la etiqueta incorrecta. El modelo genera respuestas sustancialmente más largas que Gemini Lite (~400 tokens vs ~50 tokens), lo que implica mayor latencia (~5–8 s/trial) y costo de tokens.

---

### Run 6 — `test__20260414_094516` (Gemini 2.5 Flash Lite — experimento multi-N completo)

| Campo | Valor |
|---|---|
| **Fecha y hora** | 14 de abril de 2026, 09:45:16 |
| **Modelo** | `google/gemini-2.5-flash-lite` |
| **Configuración** | N∈{2,3,4}, K=2, Q=1 (6/4/3 repeticiones) |
| **Total de trials** | 180 (5 tipos × 36 trials) |
| **Schema** | `openrouter_experiment_v2` |

Primer run con el schema v2, que habilita múltiples valores de N en un único experimento. Es el run más comprehensivo con este modelo y el primero diseñado para cuantificar el efecto del tamaño del soporte few-shot (N).

**Resultados de accuracy por tipo de prompt:**

| Tipo de prompt | Correctos / Total | Accuracy | Duración total (s) |
|---|---|---|---|
| `classification` | 34 / 36 | 0.9444 | 138.35 |
| `nle` | 36 / 36 | 1.0000 | 117.64 |
| `features` | 36 / 36 | 1.0000 | 83.82 |
| `rulebased` | 35 / 36 | 0.9722 | 119.22 |
| `axioms_ontology_v2` | 34 / 36 | 0.9444 | 71.83 |
| **TOTAL** | **175 / 180** | **0.9722** | — |

**Desglose por N:**

| N | Correctos / Total | Accuracy |
|---|---|---|
| N=2 | 58 / 60 | 0.9667 |
| N=3 | 59 / 60 | 0.9833 |
| N=4 | 58 / 60 | 0.9667 |

**Desglose por tipo de prompt y N:**

| Tipo / N | N=2 | N=3 | N=4 |
|---|---|---|---|
| `classification` | 10/12 (0.833) | 11/12 (0.917) | 13/12 (1.000) |
| `nle` | 12/12 (1.000) | 12/12 (1.000) | 12/12 (1.000) |
| `features` | 12/12 (1.000) | 12/12 (1.000) | 12/12 (1.000) |
| `rulebased` | 12/12 (1.000) | 12/12 (1.000) | 11/12 (0.917) |
| `axioms_ontology_v2` | 11/12 (0.917) | 12/12 (1.000) | 11/12 (0.917) |

**Observaciones:** El rendimiento es estable a través de los tres valores de N: no hay degradación al aumentar el número de categorías en el soporte. Las condiciones `nle` y `features` alcanzan el 100% en todos los valores de N. Los 5 errores del run se distribuyen entre `classification` (2 errores, ambos en N=2) y `axioms_ontology_v2` (2 errores, en N=2 y N=4) y `rulebased` (1 error en N=4); ninguna condición presenta un punto de falla sistemático.

---

### Run 7 — `test__20260414_100420` (Multi-modelo — Gemma 4 completo, Qwen interrumpido)

| Campo | Valor |
|---|---|
| **Fecha y hora** | 14 de abril de 2026, 10:04:20 |
| **Modelos planificados** | 4 modelos (720 trials totales) |
| **Estado** | Parcial: Gemma 4 completo (180 trials), Qwen interrumpido (21 trials), Ministral y ByteDance no ejecutados |

#### Modelo: `google/gemma-4-26b-a4b-it` (completo)

**Resultados de accuracy:**

| Tipo de prompt | Correctos / Total | Accuracy | Duración total (s) |
|---|---|---|---|
| `classification` | 36 / 36 | 1.0000 | 46.5 |
| `nle` | 36 / 36 | 1.0000 | 102.9 |
| `features` | 36 / 36 | 1.0000 | 58.7 |
| `rulebased` | 35 / 36 | 0.9722 | 205.7 |
| `axioms_ontology_v2` | 35 / 36 | 0.9722 | 181.4 |
| **TOTAL** | **178 / 180** | **0.9889** | ~595 s |

**Desglose por N:**

| N | Correctos / Total | Accuracy |
|---|---|---|
| N=2 | 60 / 60 | 1.0000 |
| N=3 | 60 / 60 | 1.0000 |
| N=4 | 58 / 60 | 0.9667 |

**Observaciones:** Gemma 4 obtuvo un rendimiento muy alto (98.89%), perfecto hasta N=3 en todos los tipos de prompt. Los únicos 2 errores se produjeron en N=4 (uno en `rulebased`, uno en `axioms_ontology_v2`), que son las condiciones con mayor demanda de razonamiento. La latencia media fue de ~3.3 s/trial, significativamente superior a Gemini Flash Lite pero más baja que los modelos de mayor latencia del conjunto. El run fue interrumpido después de completar Gemma 4; Qwen comenzó con errores de API y el proceso terminó antes de que arrancaran Ministral y ByteDance.

---

### Run 8 — `test__20260414_105023` (Multi-modelo: Ministral + ByteDance Seed)

| Campo | Valor |
|---|---|
| **Fecha de inicio** | 14 de abril de 2026, 10:50:23 |
| **Fecha de fin** | 14 de abril de 2026, ~11:49 |
| **Modelos** | `mistralai/ministral-8b-2512` + `bytedance-seed/seed-2.0-mini` |
| **Configuración** | N∈{2,3,4}, K=2, Q=1 (6/4/3 repeticiones) |
| **Total de trials** | 360 (180 por modelo) |
| **Schema** | `openrouter_experiment_v2` |

El run más extenso, con dos modelos evaluados con la configuración completa multi-N. La duración total fue de aproximadamente 59 minutos, con una diferencia marcada entre ambos modelos en términos de latencia.

#### Modelo: `mistralai/ministral-8b-2512`

**Resultados de accuracy:**

| Tipo de prompt | Correctos / Total | Accuracy | Duración (s) |
|---|---|---|---|
| `classification` | 35 / 36 | 0.9722 | 64.45 |
| `nle` | 35 / 36 | 0.9722 | 117.25 |
| `features` | 35 / 36 | 0.9722 | 69.78 |
| `rulebased` | 34 / 36 | 0.9444 | 129.52 |
| `axioms_ontology_v2` | 35 / 36 | 0.9722 | 185.53 |
| **TOTAL** | **174 / 180** | **0.9667** | ~566 s |

#### Modelo: `bytedance-seed/seed-2.0-mini`

**Resultados de accuracy:**

| Tipo de prompt | Correctos / Total | Accuracy | Duración (s) |
|---|---|---|---|
| `classification` | 36 / 36 | 1.0000 | 206.76 |
| `nle` | 36 / 36 | 1.0000 | 436.83 |
| `features` | 36 / 36 | 1.0000 | 364.50 |
| `rulebased` | 36 / 36 | 1.0000 | 733.60 |
| `axioms_ontology_v2` | 36 / 36 | 1.0000 | 1234.40 |
| **TOTAL** | **180 / 180** | **1.0000** | ~2976 s |

**Observaciones:** ByteDance Seed 2.0 Mini es el único modelo que logró 100% de accuracy en todos los tipos de prompt y todas las configuraciones de N. Sin embargo, su latencia es prohibitiva: `axioms_ontology_v2` con N=4 tardó más de 20 minutos en completarse (eje más costoso del run). Ministral 8B, con velocidad ~5x mayor, tuvo 6 errores distribuidos sin concentración sistemática en ningún tipo de prompt o valor de N, lo que sugiere que son errores de clasificación genuinos y no problemas de formato.

---

## 4. Análisis comparativo entre modelos

### 4.1. Tabla resumen — modelos con 180 trials completos en configuración multi-N

Los cuatro modelos que completaron 180 trials en la configuración multi-N (Runs 6, 7 y 8) son directamente comparables entre sí:

| Modelo | Trials | Accuracy global | Latencia media (s/trial) |
|---|---|---|---|
| `bytedance-seed/seed-2.0-mini` | 180 | **1.0000** | ~16.5 |
| `google/gemma-4-26b-a4b-it` | 180 | **0.9889** | ~3.3 |
| `google/gemini-2.5-flash-lite` | 180 | **0.9722** | ~1.8 |
| `mistralai/ministral-8b-2512` | 180 | **0.9667** | ~3.1 |

Los runs de depuración del 13 de abril (Runs 1–5) no son comparables directamente con estos cuatro por tener configuración diferente (solo N=2, menor número de repeticiones) y, en algunos casos, errores de configuración ya corregidos. Sus resultados se interpretan en contexto dentro de la sección 3.

### 4.2. Efecto del tipo de prompt

Agrupando los resultados de los cuatro modelos con configuración multi-N completa:

| Tipo de prompt | Accuracy promedio (4 modelos) |
|---|---|
| `features` | ≈ 0.9979 |
| `nle` | ≈ 0.9931 |
| `classification` | ≈ 0.9792 |
| `axioms_ontology_v2` | ≈ 0.9757 |
| `rulebased` | ≈ 0.9653 |

Las condiciones que proveen información explícita sobre rasgos (`features`) o justificaciones en lenguaje natural (`nle`) tienen el mejor rendimiento. `rulebased` acumula más errores, posiblemente porque las reglas de decisión requieren un paso de razonamiento adicional. Dicho esto, las diferencias son pequeñas y todos los tipos de prompt alcanzan más del 96% de accuracy en promedio.

### 4.3. Efecto del número de categorías (N)

| Modelo | N=2 | N=3 | N=4 |
|---|---|---|---|
| `google/gemini-2.5-flash-lite` (Run 6) | 96.67% | 98.33% | 96.67% |
| `google/gemma-4-26b-a4b-it` (Run 7) | 100.00% | 100.00% | 96.67% |
| `bytedance-seed/seed-2.0-mini` (Run 8) | 100.00% | 100.00% | 100.00% |

No se observa degradación sistemática al aumentar N. Gemma 4 y ByteDance son perfectos en N=2 y N=3; los fallos marginales en N=4 se presentan en las condiciones más exigentes (`rulebased`, `axioms_ontology_v2`).

---

## 5. Discusión

### 5.1. Depuración de la configuración en los runs iniciales

Los primeros runs del 13 de abril revelaron dos problemas de configuración que fueron corregidos antes de escalar:

**Límite de tokens insuficiente (Qwen Run 3):** La biblioteca de prompts tenía `max_tokens` variables por tipo de prompt (256 para `classification`, escalando hasta 1024 para `axioms_ontology_v2`). Qwen 3.5-9B genera razonamientos de ~400 tokens antes del tag de respuesta, por lo que con `max_tokens=256` el 100% de los trials de `classification` era truncado antes de emitir la etiqueta. La correlación entre accuracy y `max_tokens` es perfecta en ese run. Solución: unificar `max_tokens=1024` para todos los tipos de prompt.

**Instrucción de formato de respuesta subespecificada (Gemini Flash Run 1):** Para `axioms_ontology_v2`, Gemini 2.5 Flash generaba la estructura ontológica completa (tbox, abox, dl_explanation) pero omitía sistemáticamente el tag `<response>` final, dejando `predicted_label` vacío en el 100% de los trials de esa condición. Solución: reforzar en el system prompt que `<response>` debe ser siempre la última línea de la respuesta, incluso cuando se requieren bloques de razonamiento previos.

Ambas correcciones aplicadas en los runs del 14 de abril eliminaron completamente estos tipos de error.

### 5.2. Comparación de modelos

Con la configuración correcta y evaluación multi-N, emerge una jerarquía clara en términos de accuracy:

1. **ByteDance Seed 2.0 Mini** — único modelo con 100% global, pero su latencia (~16.5 s/trial en promedio, más de 34 s/trial en `axioms_ontology_v2` con N=4) lo hace poco viable para experimentos a gran escala.
2. **Gemma 4 26B** — 98.89% con latencia moderada (~3.3 s/trial). Ofrece la mejor relación accuracy/velocidad del conjunto.
3. **Gemini 2.5 Flash Lite** — 97.22% con la latencia más baja del grupo (~1.8 s/trial). Ideal para iteraciones rápidas.
4. **Ministral 8B** — 96.67%, velocidad similar a Gemma 4 (~3.1 s/trial). Modelo open-weight competitivo.

### 5.3. Dificultad del dataset

Los cuatro modelos evaluados con configuración completa alcanzan accuracy superior al 96%, lo que sugiere que el dataset `flowers` con N∈{2,3,4} puede estar por debajo del umbral de dificultad necesario para diferenciar capacidades entre modelos de manera estadísticamente robusta. Las diferencias observadas entre tipos de prompt y valores de N son pequeñas y requieren mayor poder estadístico (más repeticiones) para ser confirmadas.

---

## 6. Conclusiones

A la fecha del 14 de abril de 2026, el proyecto ha completado ocho ejecuciones de experimentos de ICL sobre el dataset `flowers`, evaluando seis modelos distintos y cinco tipos de prompt. Los resultados más relevantes son:

1. **Dos problemas de configuración fueron identificados y corregidos** durante la sesión del 13 de abril: un límite de tokens insuficiente para prompts verbosos (Qwen 3.5-9B) y una instrucción de formato subespecificada para `axioms_ontology_v2`. Ambos fueron resueltos antes de los runs de escala del 14 de abril.

2. **Los cuatro modelos evaluados en configuración completa** logran accuracy ≥ 96.67%, con ByteDance Seed 2.0 Mini en el tope (100%) seguido de Gemma 4 (98.89%), Gemini Flash Lite (97.22%) y Ministral 8B (96.67%).

3. **El tipo de prompt tiene un efecto moderado:** `features` y `nle` superan ligeramente a `classification`, `axioms_ontology_v2` y `rulebased`, pero las diferencias son inferiores a 3 puntos porcentuales entre los extremos en la configuración actual.

4. **El número de categorías (N) no degrada el rendimiento** en el rango evaluado (N∈{2,3,4}): los modelos más capaces son perfectos hasta N=3 y presentan a lo sumo 2 fallos en N=4.

5. **Gemma 4 26B ofrece el mejor balance rendimiento/velocidad** del conjunto, combinando accuracy cercano al tope (98.89%) con una latencia moderada que lo hace viable para experimentos de mayor escala.

### Próximos pasos sugeridos

- Incrementar el número de repeticiones para obtener mayor poder estadístico y confirmar las diferencias observadas entre tipos de prompt.
- Evaluar datasets con mayor dificultad intrínseca para separar mejor las capacidades de los modelos.
- Completar la evaluación de Qwen 3.5-9B y Gemini 2.5 Flash en la configuración multi-N con la configuración corregida.
- Extender el rango de N (N=5, N=10) para determinar en qué punto aparece degradación real.

---

*Informe generado a partir de los artefactos de ejecución del directorio `project/openrouter_runs/`. Fecha: 14 de abril de 2026.*
