# Guía de ejecución distribuida — Simulación paso a paso

> **Resumen**: Tú (PC A, Linux) corres Gemini 2.5 Flash + Qwen3.5-9B.
> Tu compañero (PC B, Windows) corre Gemma 4 26B + Llama 3.2 11B Vision.
> Cada uno lanza su config, sube los resultados a GitHub, y luego uno fusiona
> y ejecuta el análisis final.

---

## 0. Lo que tiene que haber antes de empezar (ambos ordenadores)

### 0.1 Archivo `.env` en la raíz del repo

El runner necesita tu clave de OpenRouter. Crea/verifica el fichero `.env`
**en la raíz del repositorio** (junto a `project/`, `episodes/`, etc.):

```
OPENROUTER_API_KEY=sk-or-v1-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

> **Importante**: `.env` está en `.gitignore` — nunca se sube al repo.
> Cada ordenador tiene el suyo propio.

### 0.2 Verificar que el repo está actualizado

**PC A (tú, Linux):**
```bash
conda activate research-explain
cd /ruta/al/repo          # donde esté clonado el repo
git pull origin nico      # rama de trabajo actual
```

**PC B (compañero, Windows) — en Anaconda Prompt o PowerShell:**
```powershell
conda activate research-explain
cd C:\ruta\al\repo
git pull origin nico
```

Tras el pull, el compañero tiene los episodios ya generados en
`episodes/seed_42/` — **no necesita ejecutar `generate_episodes.py`**.

### 0.3 Verificar episodios (ambos)

Comprueba que existen los ficheros de episodios:

**Linux:**
```bash
ls episodes/seed_42/pets/ | wc -l   # debe dar 26
ls episodes/seed_42/flowers/ | wc -l
ls episodes/seed_42/cifar10/ | wc -l
ls episodes/seed_42/dtd/ | wc -l
```

**Windows (PowerShell):**
```powershell
(Get-ChildItem episodes\seed_42\pets\).Count    # debe dar 26
(Get-ChildItem episodes\seed_42\flowers\).Count
(Get-ChildItem episodes\seed_42\cifar10\).Count
(Get-ChildItem episodes\seed_42\dtd\).Count
```

Los 4 datasets deben tener exactamente **26 ficheros** cada uno:
- `episode_N2_K1_Q1_run{0..5}.npy` (6)
- `episode_N2_K5_Q1_run{0..5}.npy` (6)
- `episode_N3_K1_Q1_run{0..3}.npy` (4)
- `episode_N3_K5_Q1_run{0..3}.npy` (4)
- `episode_N4_K1_Q1_run{0..2}.npy` (3)
- `episode_N4_K5_Q1_run{0..2}.npy` (3)

Si falta alguno → el runner lo regenera automáticamente con seed=42, pero
es mejor tenerlos ya para que ambos evaluéis exactamente las mismas imágenes.

---

## 1. Smoke test — verificar conectividad antes del experimento real

> **¿Por qué?** Para confirmar que la API key funciona y que el pipeline
> produce output correcto, gastando solo ~9 llamadas en vez de 1040.

Esto lo hace **solo PC A** (aunque PC B puede hacerlo con su propia key).

```bash
# Desde la raíz del repo
python project/run_openrouter_experiment.py \
  --config project/configs/openrouter_experiment.smoke.json
```

### Qué esperar durante el smoke test

```
[*] Loading experiment config...
[*] Loading datasets...
[*] Loading prompt library...
[*] Checking episodes...
[*] Running experiment: smoke_pets_nle
    Dataset: pets | Prompt: nle | N=2 K=1 Q=9 | Run 1/1
    Trial 1/9 ... OK (correct)
    Trial 2/9 ... OK (incorrect)
    ...
    Trial 9/9 ... OK (correct)
[+] Run complete. Accuracy: 6/9 (66.7%)
[*] Writing results to project/openrouter_runs/smoke_pets_nle__<timestamp>/
[*] Running analysis...
[+] Done.
```

### Qué comprobar al terminar

```bash
ls project/openrouter_runs/smoke_pets_nle__*/
```

Debe existir:
- `models/google--gemini-2-5-flash/trial_results.csv` — filas con `correct=0` o `correct=1`
- `models/google--gemini-2-5-flash/datasets/pets/nle/` — subcarpeta con resultados

Si el smoke test falla con `OPENROUTER_API_KEY is missing` → revisa el `.env`.
Si falla con error HTTP 401/403 → la key es incorrecta o no tiene saldo.
Si falla con error HTTP 429 → rate limit, espera unos segundos y reintenta.

---

## 2. Ejecución del experimento completo

### 2.1 PC A (tú, Linux) — Gemini 2.5 Flash + Qwen3.5-9B

```bash
# Desde la raíz del repo
python project/run_openrouter_experiment.py \
  --config project/configs/openrouter_experiment.full.pc_a.json
```

#### Qué hace el runner internamente (paso a paso)

1. **Lee el config** `openrouter_experiment.full.pc_a.json`
2. **Carga los 4 datasets** (flowers, pets, cifar10, dtd) desde `data/`
3. **Verifica los 26×4 = 104 ficheros de episodios** en `episodes/seed_42/`
4. **Por cada modelo** en la lista `["google/gemini-2.5-flash", "qwen/qwen3.5-9b"]`:
   - Lanza un subproceso independiente con ese modelo
   - El subproceso itera sobre todas las combinaciones:
     - 6 configuraciones N,K × 4 datasets × 5 prompt types = **120 runs**
     - Cada run tiene entre 3 y 6 trials (reps × Q=1)
     - **Total: 520 trials por modelo**
   - Por cada trial:
     - Carga el episodio `.npy` → obtiene índices de imágenes de soporte y query
     - Codifica las imágenes en base64
     - Construye el mensaje con el prompt correspondiente
     - Llama a OpenRouter API (1 llamada HTTP)
     - Parsea la respuesta buscando `<response>LABEL</response>`
     - Registra si es correcto o incorrecto
   - Escribe resultados en CSV y JSONL
5. **Agrega** los resultados de ambos modelos en ficheros combinados

#### Salida en terminal durante la ejecución

```
[*] Schema v2 — running 2 models sequentially
[*] Model 1/2: google/gemini-2.5-flash
    [*] Run 1/120: pets | classification | N=2 K=1 Q=1 | rep 0
        Trial 1/1 ... correct
    [*] Run 2/120: pets | classification | N=2 K=1 Q=1 | rep 1
        Trial 1/1 ... incorrect
    ...
    [+] Model complete: 520 trials, accuracy 72.3%
[*] Model 2/2: qwen/qwen3.5-9b
    [*] Run 1/120: pets | classification | N=2 K=1 Q=1 | rep 0
    ...
    [+] Model complete: 520 trials, accuracy 58.1%
[+] Experiment complete. Results in:
    project/openrouter_runs/full_experiment_pc_a__20260419_143022/
```

#### Estructura de output generada

```
project/openrouter_runs/full_experiment_pc_a__<timestamp>/
  models/
    google--gemini-2-5-flash/
      trial_results.csv           ← una fila por trial (520 filas)
      run_accuracy_long.csv       ← accuracy por run
      experiment_summary.csv      ← accuracy por (dataset, prompt)
      datasets/
        flowers/
          classification/
            N2_K1_Q1/
              run_0/
                trial_results.csv
                conversations.jsonl   ← mensajes completos enviados/recibidos
                run_summary.json
              run_1/ ...
          nle/ ...
          features/ ...
          rulebased/ ...
          axioms_ontology_v2/ ...
        pets/ ...
        cifar10/ ...
        dtd/ ...
    qwen--qwen3-5-9b/
      [misma estructura]
```

#### Tiempo estimado

| Modelo | Llamadas | Tiempo aprox. |
|--------|----------|---------------|
| Gemini 2.5 Flash | 520 | ~30-60 min |
| Qwen3.5-9B | 520 | ~30-60 min |
| **Total PC A** | **1040** | **~1-2 horas** |

(Depende del rate limit de OpenRouter y latencia de red.)

#### Si el experimento se interrumpe a mitad

El runner **no tiene checkpoint automático** por run, pero las carpetas de
runs completados ya estarán escritas. Para saber qué se ha ejecutado:

```bash
# Cuenta cuántas carpetas run_* existen bajo el modelo
find project/openrouter_runs/full_experiment_pc_a__*/models/google--gemini-2-5-flash/datasets/ \
  -name "run_summary.json" | wc -l
# Debe llegar a 120 (6 configs × 4 datasets × 5 prompts)
```

Si se interrumpe, actualmente hay que reiniciar el experimento completo para
ese modelo — los runs ya completos se sobreescribirán (mismo resultado al ser
deterministas con temperature=0 y mismos episodios).

---

### 2.2 PC B (compañero, Windows) — Gemma 4 26B + Llama 3.2 11B Vision

**En Anaconda Prompt o PowerShell:**

```powershell
conda activate research-explain
cd C:\ruta\al\repo
python project/run_openrouter_experiment.py `
  --config project/configs/openrouter_experiment.full.pc_b.json
```

> **Nota Windows**: el carácter `` ` `` (backtick) es la continuación de línea
> en PowerShell. En cmd usa `^`. O escríbelo todo en una línea.

El flujo interno es idéntico al de PC A, pero con los modelos B:
- `google/gemma-4-26b-a4b-it` → 520 trials
- `meta-llama/llama-3.2-11b-vision-instruct` → 520 trials

Output en:
```
project\openrouter_runs\full_experiment_pc_b__<timestamp>\
```

#### Tiempo estimado PC B

| Modelo | Llamadas | Tiempo aprox. |
|--------|----------|---------------|
| Gemma 4 26B | 520 | ~60-90 min |
| Llama 3.2 11B | 520 | ~30-60 min |
| **Total PC B** | **1040** | **~1.5-2.5 horas** |

---

## 3. Subir resultados a GitHub

### PC A — al terminar su ejecución

```bash
# Desde la raíz del repo
git add project/openrouter_runs/full_experiment_pc_a__<timestamp>/
git commit -m "Add results: Gemini 2.5 Flash + Qwen3.5-9B"
git push origin nico
```

> Reemplaza `<timestamp>` por el nombre real de la carpeta generada.
> Puedes verlo con: `ls project/openrouter_runs/`

### PC B — al terminar su ejecución

```powershell
git add project\openrouter_runs\full_experiment_pc_b__<timestamp>\
git commit -m "Add results: Gemma 4 26B + Llama 3.2 11B Vision"
git push origin nico
```

> **Nota sobre `.gitignore`**: `openrouter_runs/` no está en `.gitignore`,
> así que las carpetas se añaden normalmente.

---

## 4. Fusión y análisis final

Una vez que **ambos** han hecho push, cualquiera de los dos hace:

```bash
git pull origin nico   # obtener los resultados del otro
```

Verificar que están las dos carpetas:
```bash
ls project/openrouter_runs/
# full_experiment_pc_a__<timestamp>/
# full_experiment_pc_b__<timestamp>/
```

### 4.1 Crear directorio fusionado

**Linux:**
```bash
mkdir -p project/openrouter_runs/full_experiment_merged/models/

cp -r project/openrouter_runs/full_experiment_pc_a__<timestamp>/models/* \
      project/openrouter_runs/full_experiment_merged/models/

cp -r project/openrouter_runs/full_experiment_pc_b__<timestamp>/models/* \
      project/openrouter_runs/full_experiment_merged/models/
```

**Windows:**
```powershell
New-Item -ItemType Directory -Force project\openrouter_runs\full_experiment_merged\models\

Copy-Item -Recurse `
  project\openrouter_runs\full_experiment_pc_a__<timestamp>\models\* `
  project\openrouter_runs\full_experiment_merged\models\

Copy-Item -Recurse `
  project\openrouter_runs\full_experiment_pc_b__<timestamp>\models\* `
  project\openrouter_runs\full_experiment_merged\models\
```

### 4.2 Verificar que están los 4 modelos

```bash
ls project/openrouter_runs/full_experiment_merged/models/
```

Debe mostrar exactamente 4 carpetas:
```
google--gemini-2-5-flash/
qwen--qwen3-5-9b/
google--gemma-4-26b-a4b-it/
meta-llama--llama-3-2-11b-vision-instruct/
```

Si falta alguna → el push/pull no se hizo bien, o hay un error en el cp.

### 4.3 Lanzar el análisis

```bash
python project/run_openrouter_dashboard.py \
  --run-dir project/openrouter_runs/full_experiment_merged
```

O bien lanzar el análisis sin dashboard:

```bash
python -c "
from pathlib import Path
from project.openrouter_mode.analysis import analyze_run_directory
analyze_run_directory(Path('project/openrouter_runs/full_experiment_merged'))
"
```

### 4.4 Output del análisis

```
project/openrouter_runs/full_experiment_merged/
  analysis/
    tables/
      overall_accuracy_by_prompt.csv
      accuracy_by_dataset_and_prompt.csv
      accuracy_by_config_and_prompt.csv
      accuracy_by_prompt_and_model.csv        ← Tabla 2 del paper
      accuracy_by_prompt_and_dataset.csv      ← Tabla 3 del paper
      accuracy_by_config_and_model.csv        ← Tabla 4 del paper
      accuracy_by_model_and_dataset.csv       ← Tabla 5 del paper
      accuracy_by_prompt_model_dataset.csv    ← Tabla 6 / Apéndice
      accuracy_full_breakdown.csv             ← Apéndice A1-A4 con N,K
    plots/
      overall_accuracy_by_prompt.png
      accuracy_by_dataset_and_prompt.png
      ...
    report.md
```

---

## 5. Resumen del flujo completo

```
AMBOS
  └─ git pull origin nico
  └─ verificar 26 episodios × 4 datasets en episodes/seed_42/
  └─ verificar .env con OPENROUTER_API_KEY

PC A (Linux)
  └─ [opcional] smoke test con openrouter_experiment.smoke.json
  └─ python project/run_openrouter_experiment.py \
         --config project/configs/openrouter_experiment.full.pc_a.json
  └─ [esperar ~1-2 horas]
  └─ git add + commit + push

PC B (Windows) — puede empezar en paralelo con PC A
  └─ python project/run_openrouter_experiment.py \
         --config project/configs/openrouter_experiment.full.pc_b.json
  └─ [esperar ~1.5-2.5 horas]
  └─ git add + commit + push

CUALQUIERA DE LOS DOS (cuando ambos hayan hecho push)
  └─ git pull
  └─ mkdir full_experiment_merged/models/
  └─ cp modelos de pc_a y pc_b al directorio fusionado
  └─ verificar que hay 4 carpetas de modelos
  └─ lanzar análisis → tablas CSV + plots PNG
```

---

## 6. Errores comunes y cómo resolverlos

| Error | Causa | Solución |
|-------|-------|----------|
| `OPENROUTER_API_KEY is missing` | `.env` no existe o está en otra carpeta | Crea `.env` en la raíz del repo |
| `HTTP 401 Unauthorized` | Key incorrecta o expirada | Verifica la key en el dashboard de OpenRouter |
| `HTTP 429 Too Many Requests` | Rate limit alcanzado | El runner tiene retry automático; si persiste, espera 1 min |
| `FileNotFoundError: episode_N*.npy` | Episodios no generados | Ejecuta `git pull` para obtenerlos, o corre `generate_episodes.py` |
| `KeyError: model` en analysis | El merged/ no tiene `trial_results.csv` unificado | El análisis necesita que la fusión esté bien hecha |
| `No module named 'setup_utils'` | No estás en la raíz del repo | Ejecuta siempre desde la raíz, no desde `project/` |

---

## 7. Checklist antes de lanzar el experimento real

- [ ] `.env` existe en la raíz del repo con `OPENROUTER_API_KEY` válida
- [ ] `git pull` hecho — estás en la última versión de la rama `nico`
- [ ] `ls episodes/seed_42/pets/ | wc -l` → 26
- [ ] Smoke test completado sin errores
- [ ] Tienes saldo suficiente en OpenRouter (~1040 llamadas por PC)
- [ ] El PC va a estar encendido sin interrupciones durante el experimento
