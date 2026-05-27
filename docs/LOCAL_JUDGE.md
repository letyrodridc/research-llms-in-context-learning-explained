# Judge local (offline) — Qwen3-VL-32B-Thinking en MareNostrum5

Reemplazo *offline* del judge original (`gpt-5-thinking-mini` vía OpenRouter)
para poder correr la evaluación dentro de MareNostrum5, donde los nodos de
cómputo no tienen acceso a internet.

> **Caveat de scope** — el judge local **no es** el judge original. La
> comparación honesta para el paper es: "los scores absolutos provienen de un
> judge open-weight (Qwen3-VL-32B-Thinking), no del juez API usado en la
> versión previa; los rankings relativos entre condiciones siguen siendo
> útiles, pero los valores absolutos no son directamente comparables con los
> obtenidos con `gpt-5-thinking-mini`."

## 1. Por qué Qwen3-VL-32B-Thinking

Comparativa rápida de candidatos open-weight que entran en un H100-80GB:

| Modelo | VRAM BF16 | Quant en H100 | MMMU | MathVista | Eval como judge | Licencia |
|---|---|---|---|---|---|---|
| **Qwen3-VL-32B-Thinking** | ~64 GB | FP8 ~38 GB | **78.1** | **85.9** | SOTA open-weight en M-JudgeBench, −4.5 pts vs GPT-5-Nano | Apache 2.0 |
| InternVL3.5-38B | ~76 GB | FP8 ~40 GB | ~73 | n/d | Buen judge pero por debajo de Qwen3-VL en M-JudgeBench | MIT |
| Llama-4 Scout (109B MoE) | >200 GB | No entra | 69.4 | 70.7 | Sin evals publicadas como judge multimodal | Llama 4 Community |
| Pixtral Large 124B | ~250 GB | INT4 ~70 GB (degradado) | n/d | 69.4 | Sin evals como judge | MRL no comercial |

**Razones de elección:**

- SOTA open-weight en M-JudgeBench multimodal entre los modelos que entran en
  una sola H100-80GB.
- Reusa el loader `Qwen3VLForConditionalGeneration` que ya está cableado en
  `pipeline/utils/setup_utils.py` durante el refactor de `phase2`. No hay que
  introducir nuevas familias de modelos.
- Apache 2.0 → uso académico sin fricciones.
- La variante *Thinking* emite una traza de razonamiento previa al veredicto,
  útil cuando inspeccionemos manualmente los casos límite del judge.

## 2. Descarga de pesos (con internet, **antes** de subir a MN5)

MN5 no tiene salida a internet en los nodos de cómputo. La descarga se hace
desde una máquina con internet y luego se sube/copia al GPFS del proyecto.

```bash
huggingface-cli download Qwen/Qwen3-VL-32B-Thinking \
    --local-dir /gpfs/projects/ugr92/ICL/hf_cache/Qwen3-VL-32B-Thinking \
    --local-dir-use-symlinks False
```

`MODEL_IDS` ya apunta exactamente a esa ruta
(`pipeline/utils/setup_utils.py`), así que no hace falta tocar nada más.

## 3. Cómo correr el judge en MN5

### Smoke test (recomendado antes del run completo)

Antes del run pleno corré **un smoke test con ~10 episodios** para verificar
que el modelo se carga, que las imágenes se construyen bien y que el parseo
recupera los nueve scores XML del veredicto:

```bash
sbatch slurm/mn5_local_judge.sh \
    "pipeline/openrouter_runs/full_experiment_single__20260427_213833" \
    "query_only" \
    "10"
```

### Run completo

Para correr todo el experimento (omitir el tercer argumento):

```bash
sbatch slurm/mn5_local_judge.sh \
    "pipeline/openrouter_runs/full_experiment_single__20260427_213833" \
    "query_only"
```

Para el modo enriquecido con imágenes de support:

```bash
sbatch slurm/mn5_local_judge.sh \
    "pipeline/openrouter_runs/full_experiment_single__20260427_213833" \
    "query_and_support"
```

El script SLURM exporta `TRANSFORMERS_OFFLINE=1`, `HF_HUB_OFFLINE=1` y
`HF_HOME=/gpfs/projects/ugr92/ICL/hf_cache`, pide 1× H100, 20 CPUs, 180 GB de
RAM y 8 h de wallclock. Ajustá los recursos si el run completo es más largo.

### Llamada directa (sin SLURM, por ejemplo en un nodo interactivo)

```bash
python execute_local_judge.py \
    --run-dir pipeline/openrouter_runs/full_experiment_single__20260427_213833 \
    --judge-mode query_only \
    --limit 10 \
    --debug
```

## 4. Modos del judge

| Modo | Qué ve el judge | Para qué sirve |
|---|---|---|
| `query_only` *(default)* | Sólo la imagen del query, los labels candidatos y el output del clasificador. | **Comparabilidad con el paper** — reproduce el setup que usaba `gpt-5-thinking-mini` en la versión OpenRouter. |
| `query_and_support` | Lo anterior + las K imágenes de support (con su etiqueta GT) que vio el modelo evaluado. | Da al judge más grounding visual para evaluar *discriminativeness* y *specificity* (la mejora propuesta en `phase2`). No es directamente comparable con los scores anteriores. |

Default razonable: empezar con `query_only` para mantener la línea base, y
luego (si hay tiempo) repetir con `query_and_support` para reportar la
diferencia delta como ablación.

## 5. Footprint VRAM esperado

- **BF16** (default, `--quantization auto`): ~64 GB VRAM. Cabe en H100-80GB
  pero deja poco headroom para KV cache cuando hay muchas imágenes en
  contexto (`query_and_support` con K alto). Si ves OOM, bajá
  `--max-new-tokens` o pasá a FP8.
- **FP8** (recomendado en producción): ~38–40 GB VRAM. Deja headroom para
  varias imágenes en contexto. Requiere un build de Transformers/Qwen-VL con
  soporte FP8 (vía `torch_dtype=torch.float8_e4m3fn` o vía
  `quantization_config` con un quantizer compatible). El loader actual
  expone una opción de quantización (`--quantization {auto,bf16,nf4}`); si el
  env `research-explain/` tiene `torchao` o `bitsandbytes` con kernels FP8,
  la mejor forma de habilitarlos es extender `load_model_globally` en
  `pipeline/utils/setup_utils.py` con una rama `fp8`. Mientras tanto, el
  default `auto` → BF16 es el camino seguro.
- **NF4** (`--quantization nf4`): ~22 GB VRAM. Degrada significativamente la
  calidad del judge thinking — **no usar para resultados de paper**, sólo
  para debug rápido si la H100 está saturada.

## 6. Salida y compatibilidad con el judge anterior

El judge local escribe en
`<run_dir>/judge_outputs/qwen3-vl-32b-thinking-local-<modo>/`:

- `judge_results.csv` — mismo conjunto de columnas que el judge OpenRouter
  (incluye las nueve dimensiones, `overall_score`, `judge_parse_error`,
  `judge_raw_response_text`) **más** una columna extra `judge_mode` y
  `num_support_images_shown`.
- `judge_logs.jsonl` — payload por trial con el preview de mensajes (sin
  imágenes binarias, sólo metadata) y la reasoning opcional.
- `config.json` y `judge_prompt_library_snapshot.json` — auditoría de la
  corrida (modo, quantización, max tokens, hash de prompts).
- Si no se pasa `--skip-analysis`, también se generan automáticamente las
  tablas/plots de `pipeline/evaluation/judge_analysis.py`.

El parser del judge (`extract_scores_from_judge_response` en
`pipeline/evaluation/local_judge.py`) es **tolerante a la traza de
razonamiento de Qwen3-VL-Thinking**: busca primero un bloque
`<evaluation>…</evaluation>` y, si no lo encuentra, escanea el texto completo
en busca de los nueve tags `<dimension>N</dimension>`. Cuando algún score
falta, lo reporta en `judge_parse_error` sin abortar la corrida.

## 7. Recordatorio de deadline

Dado el deadline del domingo, el orden recomendado de ejecución es:

1. Descargar pesos a `/gpfs/projects/ugr92/ICL/hf_cache/Qwen3-VL-32B-Thinking/`
   (ver §2).
2. Subir / git pull la rama `phase2` en MN5.
3. **Smoke test 10 trials** con `query_only` (~15–25 min, según latencia).
4. Si el smoke pasa: lanzar el run completo en `query_only`.
5. (Opcional, si hay tiempo) re-correr con `query_and_support` para la
   ablación.
