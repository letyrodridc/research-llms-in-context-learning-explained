# EXPERIMENT_GUIDE

# Guía de ejecución del experimento completo

## Diseño del experimento

| Dimensión | Valores |
| --- | --- |
| N (clases/episodio) | {2, 3, 4} |
| K (ejemplos/clase) | {1, 5} |
| Q (queries/clase) | 1 |
| Datasets | flowers, pets, cifar10, dtd |
| Prompt types | classification, nle, features, rulebased, axioms_ontology_v2 |
| Modelos | Gemini 2.5 Flash · Gemma 4 26B · Qwen3.5-9B · Llama 3.2 11B Vision |
| Reps | N=2 → 6, N=3 → 4, N=4 → 3 |
| Seed | 42 |

El diseño está **balanceado**: Reps × N = 12 para todos los valores de N, lo que garantiza
que cada nivel de dificultad contribuye el mismo número total de presentaciones de clase
y los resultados son comparables entre configuraciones.

Niveles de azar: N=2 → 50 %, N=3 → 33.3 %, N=4 → 25 %

Cada (prompt_type, N, K, dataset) produce los siguientes trials por modelo:
- N=2 → 6 reps × Q=1 = **6 trials**
- N=3 → 4 reps × Q=1 = **4 trials**
- N=4 → 3 reps × Q=1 = **3 trials**
- Total por (prompt_type, dataset): 6+6+4+4+3+3 = **26 trials/modelo**

---

## Cambios necesarios antes de ejecutar

### `generate_episodes.py` — actualizar el grid de tests

Localizar el bloque:

```python
tests = [
    (2, 2, 9), (2, 1, 9), (3, 1, 9), (4, 1, 9)
]
```

Reemplazar por:

```python
tests = [
    (2, 1, 1),  # N=2, K=1 
    (2, 5, 1),  # N=2, K=5 
    (3, 1, 1),  # N=3, K=1 
    (3, 5, 1),  # N=3, K=5 
    (4, 1, 1),  # N=4, K=1 
    (4, 5, 1),  # N=4, K=5 
]
```

---

## Tablas de resultados para el paper

### TABLA 2 — Efecto del tipo de prompt por modelo *(tabla principal)*

**Unidad**: agregado sobre los 4 datasets y las 6 configuraciones (N,K) → **104 obs/celda**
(4 datasets × 26 trials por prompt y modelo)

**Qué enseña**: si las condiciones de prompt (NLE, features, rulebased, axioms) mejoran o
empeoran la accuracy respecto al baseline de clasificación pura, y si el efecto es
consistente entre los 4 modelos.

**Cómo leer**: cada celda = accuracy media (%) de ese modelo bajo ese prompt. La fila
"Classification" es el baseline. Diferencias positivas en otras filas indican mejora.
Negritas = mejor por columna; subrayado = segundo mejor.

```
Prompt type           | Gemini 2.5 F | Gemma 4 26B | Qwen3.5-9B | Llama 3.2 11B | Avg
Classification (†)    |    xx.x      |    xx.x     |    xx.x    |     xx.x      | xx.x
NLE                   |    xx.x      |    xx.x     |    xx.x    |     xx.x      | xx.x
Features              |    xx.x      |    xx.x     |    xx.x    |     xx.x      | xx.x
Rule-based            |    xx.x      |    xx.x     |    xx.x    |     xx.x      | xx.x
Axioms                |    xx.x      |    xx.x     |    xx.x    |     xx.x      | xx.x
```

Caption: *"Mean accuracy (%) averaged over 4 datasets and 6 few-shot configurations
(N∈{2,3,4}, K∈{1,5}, Q=1): 104 observations per cell. Chance level ranges from 25%
(4-way) to 50% (2-way). (†) Classification-only baseline. Best per column in bold."*

CSV generado: `analysis/tables/accuracy_by_prompt_and_model.csv`

---

### TABLA 3 — Efecto del tipo de prompt por dataset *(generalización)*

**Unidad**: agregado sobre los 4 modelos y las 6 configuraciones (N,K) → **104 obs/celda**
(4 modelos × 26 trials por prompt y dataset)

**Qué enseña**: si el efecto del tipo de prompt se generaliza entre datasets o si hay
interacciones dataset×prompt (p. ej., DTD —texturas— puede reaccionar diferente que
CIFAR-10 a una descripción de rasgos visuales).

**Cómo leer**: comparar filas para ver qué prompt es mejor en cada dataset; comparar
columnas para ver qué dataset es más sensible a los prompts.

```
Prompt type           | Oxford Flowers | Oxford Pets | CIFAR-10 | DTD  | Avg
Classification (†)    |    xx.x        |    xx.x     |   xx.x   | xx.x | xx.x
NLE                   |    xx.x        |    xx.x     |   xx.x   | xx.x | xx.x
Features              |    xx.x        |    xx.x     |   xx.x   | xx.x | xx.x
Rule-based            |    xx.x        |    xx.x     |   xx.x   | xx.x | xx.x
Axioms                |    xx.x        |    xx.x     |   xx.x   | xx.x | xx.x
```

Caption: *"Mean accuracy (%) per dataset, averaged over 4 models and 6 few-shot
configurations: 104 observations per cell."*

CSV generado: `analysis/tables/accuracy_by_prompt_and_dataset.csv`

---

### TABLA 4 — Efecto de la configuración few-shot por modelo

**Unidad**: agregado sobre los 4 datasets y los 5 prompts. Obs/celda **varía por N**:
- N=2 → 6 reps × 4 datasets × 5 prompts = **120 obs/celda**
- N=3 → 4 reps × 4 datasets × 5 prompts = **80 obs/celda**
- N=4 → 3 reps × 4 datasets × 5 prompts = **60 obs/celda**

**Qué enseña**: cómo varía la accuracy con la dificultad del episodio (más clases N →
más difícil; más ejemplos de soporte K → más fácil).

```
Config    | Gemini 2.5 F | Gemma 4 26B | Qwen3.5-9B | Llama 3.2 11B | obs/celda
N=2, K=1  |    xx.x      |    xx.x     |    xx.x    |     xx.x      |   120
N=2, K=5  |    xx.x      |    xx.x     |    xx.x    |     xx.x      |   120
N=3, K=1  |    xx.x      |    xx.x     |    xx.x    |     xx.x      |    80
N=3, K=5  |    xx.x      |    xx.x     |    xx.x    |     xx.x      |    80
N=4, K=1  |    xx.x      |    xx.x     |    xx.x    |     xx.x      |    60
N=4, K=5  |    xx.x      |    xx.x     |    xx.x    |     xx.x      |    60
Chance    |  50/33/25 %  |  50/33/25 % |  50/33/25 %|   50/33/25 %  |    —
```

Caption: *"Mean accuracy (%) per few-shot configuration, averaged over 4 datasets and
5 prompt types. Observations per cell: 120 (N=2), 80 (N=3), 60 (N=4). Chance levels:
50% (N=2), 33.3% (N=3), 25% (N=4)."*

CSV generado: `analysis/tables/accuracy_by_config_and_model.csv`

---

### TABLA 5 — Accuracy por modelo y dataset

**Unidad**: agregado sobre los 5 prompts y las 6 configuraciones (N,K) → **130 obs/celda**
(5 prompts × 26 trials por modelo y dataset)

```
Model          | Flowers | Oxford Pets | CIFAR-10 | DTD  | Avg
Gemini 2.5 F   |  xx.x   |    xx.x     |   xx.x   | xx.x | xx.x
Gemma 4 26B    |  xx.x   |    xx.x     |   xx.x   | xx.x | xx.x
Qwen3.5-9B     |  xx.x   |    xx.x     |   xx.x   | xx.x | xx.x
Llama 3.2 11B  |  xx.x   |    xx.x     |   xx.x   | xx.x | xx.x
```

Caption: *"Mean accuracy (%) per model and dataset, averaged over 5 prompt conditions
and 6 few-shot configurations: 130 observations per cell."*

CSV generado: `analysis/tables/accuracy_by_model_and_dataset.csv`

---

### TABLA 6 — Mejor condición de prompt por modelo y dataset *(landscape, resumen)*

Para cada (modelo, dataset), el prompt con mayor accuracy media sobre las 6
configuraciones (N,K) — 26 trials del prompt ganador.

Caption: *"Best-performing prompt condition per model and dataset, averaged over
N∈{2,3,4} and K∈{1,5} (26 trials per prompt)."*

---

### TABLAS DE APÉNDICE A1–A4 — Desglose completo por modelo *(landscape)*

Una tabla por modelo. Filas = 5 prompt conditions. Columnas agrupadas en 3 niveles:
**Dataset → N-way → K-shot** (4 datasets × 3 N × 2 K = 24 columnas de datos).

Obs/celda varía por N: N=2 → 6, N=3 → 4, N=4 → 3.

CSV generado: `analysis/tables/accuracy_full_breakdown.csv`

---

## Cambios necesarios en `analysis.py`

Fichero: `project/openrouter_mode/analysis.py`

### Qué añadir JUSTO DESPUÉS del bloque existente (~líneas 654-671)

```python
# ── Tablas para el paper ──────────────────────────────────────────────────────

prompt_model   = _aggregate_accuracy(trial_rows, ["prompt_type", "model"])
prompt_dset    = _aggregate_accuracy(trial_rows, ["prompt_type", "dataset"])
config_model   = _aggregate_accuracy(trial_rows, ["config_n", "config_k", "model"])
model_dset     = _aggregate_accuracy(trial_rows, ["model", "dataset"])
full_breakdown = _aggregate_accuracy(trial_rows, ["prompt_type", "model", "dataset"])
full_nk        = _aggregate_accuracy(trial_rows, ["prompt_type", "model", "dataset",
                                                   "config_n", "config_k"])

_write_csv(
    tables_dir / "accuracy_by_prompt_and_model.csv",          # Tabla 2
    ["prompt_type", "model", "correct", "total", "accuracy",
     "standard_error", "ci95_low", "ci95_high"],
    prompt_model,
)
_write_csv(
    tables_dir / "accuracy_by_prompt_and_dataset.csv",         # Tabla 3
    ["prompt_type", "dataset", "correct", "total", "accuracy",
     "standard_error", "ci95_low", "ci95_high"],
    prompt_dset,
)
_write_csv(
    tables_dir / "accuracy_by_config_and_model.csv",           # Tabla 4
    ["config_n", "config_k", "model", "correct", "total", "accuracy",
     "standard_error", "ci95_low", "ci95_high"],
    config_model,
)
_write_csv(
    tables_dir / "accuracy_by_model_and_dataset.csv",          # Tabla 5
    ["model", "dataset", "correct", "total", "accuracy",
     "standard_error", "ci95_low", "ci95_high"],
    model_dset,
)
_write_csv(
    tables_dir / "accuracy_by_prompt_model_dataset.csv",       # Tabla 6 / Apéndice
    ["prompt_type", "model", "dataset", "correct", "total", "accuracy",
     "standard_error", "ci95_low", "ci95_high"],
    full_breakdown,
)
_write_csv(
    tables_dir / "accuracy_full_breakdown.csv",                # Apéndice A1-A4 con N,K
    ["prompt_type", "model", "dataset", "config_n", "config_k",
     "correct", "total", "accuracy", "standard_error", "ci95_low", "ci95_high"],
    full_nk,
)
```

### Qué añadir en el bloque README (~líneas 738-740)

```python
"- `tables/accuracy_by_prompt_and_model.csv`",        # Tabla 2 del paper
"- `tables/accuracy_by_prompt_and_dataset.csv`",       # Tabla 3 del paper
"- `tables/accuracy_by_config_and_model.csv`",         # Tabla 4 del paper
"- `tables/accuracy_by_model_and_dataset.csv`",        # Tabla 5 del paper
"- `tables/accuracy_by_prompt_model_dataset.csv`",     # Tabla 6 / Apéndice A1-A4
"- `tables/accuracy_full_breakdown.csv`",              # Apéndice A1-A4 con N,K desglosado
```

> **Nota**: `_aggregate_accuracy` y `_write_csv` ya existen y no requieren cambios.
>

---

## Ejecución distribuida entre dos ordenadores

### Prerequisito: generar episodios con la misma semilla

En **ambos** ordenadores, antes de empezar actualizar `generate_episodes.py` (ver
sección anterior) y ejecutar:

```bash
conda activate research-explain
python project/generate_episodes.py  # usa seed=42 por defecto
```

> **Alternativa más rápida**: generar los episodios en un ordenador, subirlos al
repositorio de GitHub compartido y obtenerlos en el otro con `git pull`.
>

---

### Fusión de resultados vía GitHub

Dado que ambos ordenadores comparten el mismo repositorio de GitHub, no hace falta
`rsync` ni transferencias manuales. Cada ordenador sube sus resultados al repo y el
otro los obtiene con `git pull`.

**Flujo recomendado:**

```bash
# Al terminar, en el Ordenador A:
git add project/openrouter_runs/full_experiment_pc_a__<timestamp>/
git commit -m "Add results: Gemini 2.5 Flash + Qwen3.5-9B"
git push

# Al terminar, en el Ordenador B:
git add project/openrouter_runs/full_experiment_pc_b__<timestamp>/
git commit -m "Add results: Gemma 4 26B + Llama 3.2 11B Vision"
git push

# En cualquiera de los dos, una vez ambos han subido:
git pull  # obtiene los resultados del otro ordenador
```

Una vez que ambas carpetas están presentes localmente, fusionar los modelos en un
único directorio y lanzar el análisis:

```bash
# Crear directorio fusionado
mkdir -p project/openrouter_runs/full_experiment_merged/models/

cp -r project/openrouter_runs/full_experiment_pc_a__<timestamp>/models/* \
      project/openrouter_runs/full_experiment_merged/models/
cp -r project/openrouter_runs/full_experiment_pc_b__<timestamp>/models/* \
      project/openrouter_runs/full_experiment_merged/models/

# Verificar que los 4 modelos están presentes
ls project/openrouter_runs/full_experiment_merged/models/

# Lanzar el análisis sobre el directorio fusionado
python project/run_openrouter_dashboard.py \
  --run-dir project/openrouter_runs/full_experiment_merged
```

> **Nota sobre `.gitignore`**: asegurarse de que `openrouter_runs/` no está excluido.
Si lo está, añadir una excepción o usar una rama dedicada para los resultados.
>

---

### Configuración Ordenador A

Crear `project/configs/openrouter_experiment.full.pc_a.json`:

```json
{
  "schema_version": "openrouter_experiment_v2",
  "experiment_name": "full_experiment_pc_a",
  "description": "Modelos A: Gemini 2.5 Flash + Qwen3.5-9B",
  "env_file": "../../.env",
  "output_root": "../openrouter_runs",
  "prompt_library_path": "openrouter_prompt_library.default.json",
  "model": {
    "model_name": [
      "google/gemini-2.5-flash",
      "qwen/qwen3.5-9b"
    ],
    "model_params": {
      "validate_image_input": true,
      "generation": { "temperature": 0.0 }
    }
  },
  "datasets": ["flowers", "pets", "cifar10", "dtd"],
  "prompt_types": ["classification", "nle", "features", "rulebased", "axioms_ontology_v2"],
  "few_shot_configs": [
    {"n": 2, "k": 1, "q": 1, "runs": 6},
    {"n": 2, "k": 5, "q": 1, "runs": 6},
    {"n": 3, "k": 1, "q": 1, "runs": 4},
    {"n": 3, "k": 5, "q": 1, "runs": 4},
    {"n": 4, "k": 1, "q": 1, "runs": 3},
    {"n": 4, "k": 5, "q": 1, "runs": 3}
  ],
  "seed": 42,
  "analysis": { "enabled": false },
  "logging": {
    "write_full_conversations": true,
    "write_request_payloads": false,
    "write_sharded_logs": true
  }
}
```

### Configuración Ordenador B

Crear `project/configs/openrouter_experiment.full.pc_b.json` — idéntico al anterior
salvo `experiment_name` y `model_name`:

```json
  "experiment_name": "full_experiment_pc_b",
  "description": "Modelos B: Gemma 4 26B + Llama 3.2 11B Vision",
  "model": {
    "model_name": [
      "google/gemma-4-26b-a4b-it",
      "meta-llama/llama-3.2-11b-vision-instruct"
    ],
    ...
  }
```

> `"analysis": { "enabled": false }` en ambos para no hacer análisis parciales.
El análisis definitivo se hace sobre los resultados fusionados.
>

---

### Comandos de ejecución

**Ordenador A:**

```bash
conda activate research-explain
cd /ruta/al/repo
python project/run_openrouter_experiment.py \
  --config project/configs/openrouter_experiment.full.pc_a.json
```

**Ordenador B:**

```bash
conda activate research-explain
cd /ruta/al/repo
python project/run_openrouter_experiment.py \
  --config project/configs/openrouter_experiment.full.pc_b.json
```

---

## Volumen de llamadas API estimado

| Config | Reps | Trials / modelo | Total (4 modelos) |
| --- | --- | --- | --- |
| N=2, K=1 | 6 | 6 × 4 datasets × 5 prompts = 120 | 480 |
| N=2, K=5 | 6 | 120 | 480 |
| N=3, K=1 | 4 | 4 × 4 × 5 = 80 | 320 |
| N=3, K=5 | 4 | 80 | 320 |
| N=4, K=1 | 3 | 3 × 4 × 5 = 60 | 240 |
| N=4, K=5 | 3 | 60 | 240 |
| **Total** | | **520 / modelo** | **2 080 API calls** |

Cada ordenador maneja **1 040 llamadas API** (2 modelos × 520).