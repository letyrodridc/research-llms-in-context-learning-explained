# HITL Validation — Instrucciones Completas

Este documento contiene todo lo que cada investigador necesita para completar la validación humana del experimento ICL.

---

## Por qué 192 muestras y cómo se seleccionaron

### El experimento completo

El experimento evaluó 4,608 trials con explicación: 4 modelos × 4 datasets × 4 condiciones de explicación (E2-E5) × 72 episodios por combinación. Cada episodio es una clasificación independiente con un support set distinto. El juez LLM puntuó los 4,608 trials automáticamente en 9 métricas (escala 1-5).

### Qué es un stratum

Un stratum es una combinación única de (condición × modelo × dataset). Por ejemplo: "E3 + Gemini 2.5 Flash + Oxford Flowers" es un stratum. Hay 4 × 4 × 4 = 64 strata en total. Cada uno contiene 72 episodios.

### Por qué 192 muestras

Queremos que la validación humana cubra TODAS las combinaciones del experimento de forma equilibrada, no solo las más frecuentes o más fáciles. La solución es coger el mismo número de muestras de cada stratum: 3 muestras × 64 strata = 192 muestras totales.

192 / 4,608 = 4.2% del total — porcentaje estándar para validación de LLM-as-a-judge en la literatura (rango típico: 3-10% estratificado).

### Por qué 3 muestras por stratum y no más

- Con 3 por stratum, cada anotador evalúa 128 items (manejable en 3-4 días sin degradación de calidad por fatiga)
- Con 5 por stratum serían 320 muestras, 160 items por persona
- 3 es el mínimo para tener variabilidad dentro del stratum y poder detectar si el juez es consistente o errático en esa combinación

### Por qué forzamos al menos 1 predicción incorrecta por stratum

Si solo evaluáramos predicciones correctas, las métricas estarían sesgadas: los modelos tienden a generar mejores explicaciones cuando aciertan. Forzar al menos 1 incorrecto por stratum garantiza que evaluamos también la calidad de las explicaciones en casos de error, que es precisamente donde la validación humana es más informativa.

14 de los 64 strata tienen 0 incorrectos disponibles (concentrados en flowers y pets, donde los modelos alcanzan casi 100% de accuracy) — en esos casos se seleccionan 3 correctos y queda documentado en human_validation_sample.csv.

### Por qué seed=42

Para reproducibilidad: cualquier investigador con los mismos datos puede regenerar exactamente las mismas 192 muestras ejecutando `python sample_for_validation.py`. El seed está documentado en la columna `selection_seed` de human_validation_sample.csv.

### Cómo se reparten entre anotadores

Para calcular acuerdo inter-anotador (Cohen's κ), cada item debe ser evaluado por 2 personas distintas:

- Items 1-64:   Carmen + Leticia
- Items 65-128: Leticia + Nico
- Items 129-192: Carmen + Nico

Cada anotador evalúa 128 items. Ningún item se repite dentro del CSV de una misma persona.

---

## Para Carmen — Paso 0: Generación de datos (solo una vez, en el servidor)

> **Solo lo hace Carmen, una vez, en el servidor donde están los datos del experimento.**

```bash
# 1. Activar entorno
conda activate /mnt/homeGPU/cquiles/ICL/ICL/research-explain

# 2. Ir al directorio del repo
cd /mnt/homeGPU/cquiles/ICL/ICL

# 3. Ejecutar el script de muestreo
python pipeline/hitl_evaluator/sample_for_validation.py
```

El script genera:
- `pipeline/hitl_evaluator/human_validation_sample.csv` — las 192 muestras seleccionadas
- `pipeline/hitl_evaluator/annotations_carmen.csv` — tus 128 items
- `pipeline/hitl_evaluator/annotations_leticia.csv` — los 128 items de Leticia
- `pipeline/hitl_evaluator/annotations_nico.csv` — los 128 items de Nico

También imprime un **audit report** con las estadísticas del muestreo y las excepciones (strata con pocos incorrectos). Léelo antes de continuar.

```bash
# 4. Subir los CSVs al repo
git add pipeline/hitl_evaluator/annotations_*.csv pipeline/hitl_evaluator/human_validation_sample.csv
git commit -m "hitl: generate 192-item validation sample and annotator CSVs"
git push
```

Leticia y Nico harán `git pull` para obtener sus CSVs.

---

## Para Carmen — Anotación

```bash
# 1. Activar entorno
conda activate /mnt/homeGPU/cquiles/ICL/ICL/research-explain

# 2. Ir al directorio de la app
cd /mnt/homeGPU/cquiles/ICL/ICL/pipeline/hitl_evaluator

# 3. Asegúrate de tener los datasets en data/
#    (ya los tienes en el servidor, no hace falta descargar nada)

# 4. Lanzar la app
python annotation_app/app.py --annotator carmen --data-dir ../../data

# 5. Abrir en el navegador:
#    http://127.0.0.1:8766
```

**Qué verás en la app:**
- Columna izquierda: la imagen query del trial, con la clase predicha (verde si correcto, rojo si incorrecto)
- Columna central: el texto de la explicación generada por el modelo, y las support images en un desplegable
- Columna derecha: las 9 métricas para puntuar (botones 1-5)

**Cómo anotar:**
1. Lee la explicación del modelo y observa la imagen
2. Puntúa cada una de las 9 métricas del 1 al 5 (haz clic en el ℹ para ver la rúbrica completa)
3. Pulsa **"Guardar y Siguiente"** → la anotación se guarda inmediatamente en disco
4. Repite para los 128 items

**Si cierras el navegador y quieres retomar:** vuelve a ejecutar el mismo comando y la app retomará automáticamente en el primer item sin completar.

**Al terminar los 128 items:**
1. Pulsa el botón **"Exportar CSV"** → descarga `annotations_carmen_final.csv`
2. Guárdalo en `pipeline/hitl_evaluator/`
3. Sube al repo:
   ```bash
   git add pipeline/hitl_evaluator/annotations_carmen_final.csv
   git commit -m "hitl: carmen annotations complete"
   git push
   ```

---

## Para Leticia — Instalación y anotación

### Paso 1: Obtener el código y los datos

```bash
# Clonar el repo (o hacer pull si ya lo tienes)
git clone <URL_DEL_REPO>
cd <nombre_del_repo>

# O si ya tienes el repo:
git pull
```

### Paso 2: Instalar el entorno

```bash
# Instalar miniconda si no lo tienes:
# https://docs.conda.io/en/latest/miniconda.html

# Crear el entorno (solo la primera vez):
conda env create -f environment.yml
# O si hay un requirements.txt:
conda create -n research-explain python=3.10
conda activate research-explain
pip install -r requirements.txt
```

### Paso 3: Descargar los datasets

Los datasets de imágenes no están en el repo (son demasiado grandes). Descárgalos automáticamente con:

```bash
conda activate research-explain
python -c "
import sys; sys.path.insert(0, '.')
from pipeline.utils.setup_utils import load_datasets
load_datasets('data/')
print('Datasets descargados correctamente.')
"
```

Esto descarga automáticamente Flowers-102, Oxford Pets, CIFAR-10 y DTD en la carpeta `data/`. Tardará unos minutos la primera vez.

### Paso 4: Lanzar la app

```bash
conda activate research-explain
cd pipeline/hitl_evaluator
python annotation_app/app.py --annotator leticia --data-dir ../../data
```

Abre **http://127.0.0.1:8767** en tu navegador.

### Paso 5: Anotar

Sigue las mismas instrucciones que Carmen (ver sección anterior). Tienes 128 items.

### Paso 6: Exportar y subir

Al terminar, pulsa "Exportar CSV" → guarda `annotations_leticia_final.csv` en `pipeline/hitl_evaluator/`, luego:

```bash
git add pipeline/hitl_evaluator/annotations_leticia_final.csv
git commit -m "hitl: leticia annotations complete"
git push
```

---

## Para Nico — Instalación y anotación

Sigue exactamente los mismos pasos que Leticia (Paso 1 al 6), pero en el Paso 4 usa `--annotator nico`:

```bash
conda activate research-explain
cd pipeline/hitl_evaluator
python annotation_app/app.py --annotator nico --data-dir ../../data
```

Abre **http://127.0.0.1:8768** en tu navegador.

Al exportar, el archivo se llamará `annotations_nico_final.csv`. Súbelo al repo:

```bash
git add pipeline/hitl_evaluator/annotations_nico_final.csv
git commit -m "hitl: nico annotations complete"
git push
```

---

## Para Carmen — Análisis final

> **Hacer esto solo cuando los 3 hayáis hecho push de vuestros CSVs finales.**

```bash
# 1. Obtener los CSVs de los 3 anotadores
git pull

# 2. Verificar que están los 3 archivos:
ls pipeline/hitl_evaluator/annotations_*_final.csv
# Debe mostrar: annotations_carmen_final.csv  annotations_leticia_final.csv  annotations_nico_final.csv

# 3. Ejecutar el análisis
conda activate /mnt/homeGPU/cquiles/ICL/ICL/research-explain
cd /mnt/homeGPU/cquiles/ICL/ICL
python pipeline/hitl_evaluator/analysis_human_validation.py
```

### Qué calcula el análisis

**Cohen's κ ponderado (pesos lineales):**
Mide el acuerdo entre los 2 anotadores de cada item, para cada una de las 9 métricas.
- κ > 0.6: acuerdo sustancial → los anotadores ven las mismas cosas
- κ 0.4-0.6: acuerdo moderado → aceptable para métricas subjetivas
- κ < 0.4: acuerdo débil → la métrica puede ser difícil de calibrar

Se calcula tanto globalmente (los 192 items agregados) como por par de anotadores (carmen_leticia, carmen_nico, leticia_nico) para detectar si alguna combinación tiene calibraciones muy distintas.

**Calibración por anotador:**
Comprueba si los anotadores usan el rango numérico de la escala de forma similar. Diferencias sistemáticas de >0.5 puntos en la media global entre anotadores indican falta de calibración y deben mencionarse como limitación en el paper. No implica que un anotador esté equivocado, sino que usan el extremo alto o bajo de la escala de forma distinta.

**Spearman ρ y MAE (juez vs. humanos):**
Mide si el juez LLM ordena y puntúa los items de forma similar a como lo hacen los humanos.
- ρ > 0.5: el juez captura bien el ordenamiento humano
- MAE < 1.0: el juez se desvía menos de 1 punto en promedio de los humanos

**Sesgo sistemático (bias):**
Para cada métrica se calcula la diferencia media (juez − media_humana), su desviación típica y los porcentajes de veces que el juez puntúa más alto, más bajo o igual. Un sesgo negativo significa que el juez es más estricto que los humanos; positivo, más generoso. Siempre interpretar el sesgo junto con la tabla de calibración: un sesgo aparente puede deberse a que la media humana está inflada por un anotador sistemáticamente más generoso.

Para el paper: usa κ como argumento de fiabilidad inter-anotador, ρ/MAE como argumento de validez del juez LLM, y el bias chart para documentar y justificar cualquier desviación sistemática del juez.

### Resultados generados

Los resultados se guardan en `pipeline/hitl_evaluator/results/`:

#### Tablas CSV

| Archivo | Contenido |
|---------|-----------|
| `alignment_results.csv` | Tabla base: κ, ρ, MAE por métrica |
| `full_summary.csv` | Tabla maestra: κ, ρ, MAE, sesgo, std, % higher/lower, human_mean, judge_mean — todo en una fila por métrica |
| `items_with_scores.csv` | Tabla completa item a item: scores de ann1, ann2, media humana y score del juez para los 192 items |
| `calibration_by_annotator.csv` | Media por anotador × métrica + media global y nº de items — revela diferencias de calibración |
| `bias_judge_vs_human.csv` | Sesgo del juez por métrica: mean, std, mediana, min, max y % veces que el juez puntúa más alto/más bajo/igual |
| `score_comparison_human_vs_judge.csv` | Medias y medianas de human_mean vs judge por métrica, con std y diferencia |
| `kappa_by_pair_and_metric.csv` | κ para cada par de anotadores (carmen_leticia, carmen_nico, leticia_nico) × métrica, más medias y diferencia absoluta media |
| `bias_by_condition_and_metric.csv` | Sesgo del juez desglosado por condición (E2-E5) × métrica — detecta si el juez es más o menos fiable según el tipo de prompt |

#### Tabla LaTeX

| Archivo | Contenido |
|---------|-----------|
| `table_alignment.tex` | Tabla con κ, ρ, MAE por métrica lista para incluir en el paper |

#### Gráficas

| Archivo | Contenido |
|---------|-----------|
| `scatter_judge_vs_human.png` | 9 scatter plots (uno por métrica): juez en eje X, media humana en eje Y, diagonal perfecta en negro. Muestra ρ y n en cada panel |
| `heatmap_kappa_by_annotator_pair.png` | Heatmap de κ por par de anotadores × métrica. Permite ver qué pares discrepan más y en qué métricas |
| `heatmap_annotator_calibration.png` | Heatmap de puntuación media por anotador × métrica. La gráfica más útil para detectar diferencias de calibración |
| `violin_score_distributions.png` | Violines por métrica mostrando la distribución de scores de cada anotador y del juez lado a lado. Confirma visualmente las diferencias de calibración y el rango que usa cada uno |
| `bias_barchart_judge_vs_human.png` | Barras horizontales con media(juez−humano) ± 1 SD por métrica. Rojo = juez infravalora, verde = juez sobrevalora |
| `boxplot_diff_by_condition.png` | Boxplot de (juez−humano) por condición E2-E5 para cada métrica. Detecta si el sesgo del juez varía según el tipo de prompt |

---

## Referencia rápida — Rúbricas

| Métrica | Pregunta clave |
|---------|---------------|
| LD — Local Discriminativeness | ¿Destaca features que distinguen esta clase de las otras? |
| TG — Textual Groundedness | ¿Captura todos los conceptos relevantes de la imagen? |
| HF — Hallucination-Free | ¿Sin alucinaciones (claims que no están en la imagen)? |
| IF — Instruction Following | ¿Siguió el formato/estructura pedido en el prompt? |
| CC — Concept Counting | ¿Cuenta correctamente los atributos? |
| CP — Comprehensibility | ¿Es clara y fácil de leer? |
| Cn — Conciseness | ¿Concisa, sin redundancias? |
| S — Specificity | ¿Específica, con detalles concretos? |
| LC — Logical Coherence | ¿Los razonamientos/axiomas son coherentes entre sí? |

Todas en escala **1 (muy malo) → 5 (perfecto)**. Usa el botón ℹ en la app para ver la rúbrica completa de cada métrica.

---

## Comparison Analysis (Judge vs. Humans) - Detail and Scoring

This additional step allows detailed comparison of the LLM judge's evaluations against human annotators, calculating an agreement "score" and generating interactive visualizations.

### 1. Generate the comparison CSV

Run the script to combine the results from the three annotators and calculate the absolute distances per dimension:

```bash
python process_csv.py
```

This will generate the file `comparison_results.csv` with `dist_` columns for each of the 9 dimensions.

### 2. Analysis and Visualization in Jupyter

Open the `analysis.ipynb` notebook to explore the results:

```bash
jupyter notebook analysis.ipynb
```

**Scoring Logic (Score):**
The notebook calculates a proximity metric for each dimension and trial:
- **1.0**: If the distance between the judge and human is **0** (exact match).
- **0.75**: If the distance is **1** (very close).
- **0.0**: In any other case.

**Included Visualizations:**
The notebook automatically generates bar charts and heatmaps broken down by:
- **Annotator Name** (`annotator`)
- **Dimension** (TG, HF, CC, etc.)
- **Condition** (E2-E5)
- **Dataset** (CIFAR-10, Flowers, etc.)