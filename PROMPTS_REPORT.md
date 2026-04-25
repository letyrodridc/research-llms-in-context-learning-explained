# Prompt Report

This document summarizes the prompts used in the current experiment, their role within the few-shot protocol, the actual differences between them, and recommended improvements before scaling experiments or comparing models via API.

## Shared Context

All five variants share the same general structure:

1. A `system prompt` defining the role and constraints.
2. `K` examples per class in a few-shot format.
3. Each few-shot example uses:
   - `user`: image + `"What is the class of this image?"`
   - `assistant`: `<response>label</response>`
4. The final query uses:
   - A new image.
   - A textual instruction that forces the choice of one of the `N` seen classes.
5. Evaluation only considers the content within the `<response>...</response>` tags.

This means that the actual experimental difference lies not in the data or the evaluation protocol, but in how the model is asked to reason before classifying.

## Shared Few-Shot Template

### Support Examples

```text
User:
- [image]
- What is the class of this image?

Assistant:
<response>{label}</response>
```

### Final Query

The model is always asked to choose exactly one class from a closed list:

```text
You MUST choose exactly one from [{options_str}]
```

This is beneficial as it significantly reduces output ambiguity and makes automatic evaluation more robust.

---

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

### Purpose
Acts as the cleanest baseline: few-shot classification with minimal extra structure.

### Advantages
- Simplest variant and easiest to interpret.
- Lower risk of the model "getting lost" generating long explanations.
- Best reference for comparing whether explanations help or hinder performance.

### Disadvantages
- Asks to "reason step-by-step" but does not force the model to show that reasoning.
- This instruction may induce latent reasoning but leaves no traceability.
- If the model uses prior knowledge, it is difficult to audit later.

### Potential Improvements
- For a strictly mute baseline, remove "Reason step-by-step".
- For an explainable baseline, request a minimal one-line structured justification.
- Add a stronger clause such as: "If uncertain, still choose one of the provided labels and do not invent labels."

---

## 2. NLE (Natural Language Explanation)

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

### Purpose
To measure if forcing the model to verbalize an explanation in natural language improves or hinders classification accuracy.

### Advantages
- Readable and easy to inspect manually.
- Useful for detecting if the model attends to reasonable features.
- A natural way to audit errors.

### Disadvantages
- "Natural language explanation" is very open-ended.
- May produce plausible but post-hoc explanations.
- Does not enforce a stable structure across runs and models.
- Mixing free-form explanation with classification may increase length and cost without gaining precision.

### Potential Improvements
- Enforce a shorter template:
  - `Observed cues: ...`
  - `Contrast with alternatives: ...`
  - `<response>...</response>`
- Limit the explanation to 2-3 sentences.
- Request explicit comparative evidence between candidate classes, not just a description of the chosen one.

---

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

### Purpose
To measure if forcing the model through an explicit visual feature extraction stage improves decision-making.

### Advantages
- Introduces a more concrete structure than NLE.
- Pushes the model to use observable visual evidence.
- Easier to compare across outputs than free-form explanations.

### Disadvantages
- Significant internal inconsistency:
  - Initially asks to look at color, texture, and shape.
  - Subsequently prohibits using color, size, length, texture, and shape as "features."
- Examples in the query (`e.g., Specific color/texture`) contradict the system prompt.
- "Physical components" works well for objects but less so for flowers, textures, or breeds where differences often lie in pattern, shape, or color.

### Potential Improvements
- Clearly separate:
  - `parts/components`
  - `surface patterns`
  - `global shape cues`
  - `color cues`
- For datasets like `flowers` and `dtd`, explicitly allow pattern, symmetry, texture, and color distribution.
- Replace the current rule with something more general: "Use only directly observable visual cues; do not use taxonomy or world knowledge."

---

## 4. Rule-based

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

### Purpose
To measure if structuring reasoning as explicit `IF-THEN` rules helps the model discriminate better.

### Advantages
- Makes an intermediate reasoning path visible.
- More auditable than NLE.
- The `Rule Check` part can reveal if the model is comparing classes or just justifying a pre-made decision.

### Disadvantages
- "At least one IF-THEN rule per class" with very few shots may induce narrative overfitting.
- May invent rules that are too strong based on very little evidence.
- Mixes three difficult tasks:
  - Feature extraction.
  - Rule induction.
  - Classification.
- Significant increase in token cost.

### Potential Improvements
- Request tentative rather than definitional rules: "Propose provisional visual rules supported by the examples."
- Force contrast: "For each candidate class, state one supporting cue and one conflicting cue."
- Limit the number of rules: "At most one short rule per class."
- Request a match score per class instead of a long explanation.

---

## 5. Axioms / Ontology

### System Prompt (Summary)

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

### Purpose
To measure if a much more formal ontological reasoning structure changes performance or explanatory quality.

### Advantages
- Strongest condition in terms of structural explicitness.
- Interesting as an interpretability experiment.
- Clearly reveals when the model is "filling in" formalism without visual anchoring.

### Disadvantages
- Likely the prompt with the highest cognitive load and risk of formal hallucination.
- Requests DL/OWL/TBox/ABox despite minimal few-shot evidence.
- Conceptual examples (e.g., pizza/triangle/centaur) are far from the actual experimental domain.
- Strongly incentivizes the invention of pseudo-ontologies that sound correct but lack support.
- Least comparable to a natural visual classification task.

### Potential Improvements
- Significantly lower formal ambition.
- Replace strict OWL/DL with a "soft schema":
  - `Class hypothesis`
  - `Observed cues`
  - `Necessary-like cues`
  - `Most diagnostic cue`
- If continuing with ontologies, use a simpler, vision-consistent controlled language:
  - `Class -> diagnostic visual cues`
  - `Observed instance -> cues present`
  - `Decision -> best-matching class`

---

## General Comparison

### Complexity
- `classification`: Low
- `nle`: Low-Medium
- `features`: Medium
- `rulebased`: Medium-High
- `axioms_ontology_v2`: Very High

### Risk of Explanatory Hallucination
- `classification`: Low
- `nle`: Medium
- `features`: Medium
- `rulebased`: Medium-High
- `axioms_ontology_v2`: High

### Ease of Error Auditing
- `classification`: Low
- `nle`: Medium
- `features`: High
- `rulebased`: High
- `axioms_ontology_v2`: Medium

### Expected Token Cost
- `classification`: Low
- `nle`: Medium
- `features`: Medium
- `rulebased`: High
- `axioms_ontology_v2`: Very High

### Experimental Quality (Clean Comparison)
- `classification`: Very Good
- `nle`: Good
- `features`: Good if inconsistency is corrected.
- `rulebased`: Acceptable, but already mixes multiple operations.
- `axioms_ontology_v2`: Weak as a clean benchmark; interesting as an exploratory condition.

---

## Transversal Issues

### 1. Inconsistency Between Prompts
Not all prompts request the exact same type of visual evidence. Some prioritize color/texture/shape, others restrict them, and others convert them into ontological pseudo-properties.

### 2. High Cognitive Load in System Prompts
Especially in `rulebased` and `axioms`, the system prompt is very long. This can:
- Increase cost.
- Introduce noise.
- Make format adherence more fragile.
- Negatively affect smaller models.

### 3. Lack of Dataset Calibration
Not all datasets benefit from the same "feature language."
- `cifar10`: Global objects; works well with parts and shape.
- `pets`: Fine features, fur, morphology, face, ears.
- `flowers`: Color, petal shape, arrangement.
- `dtd`: Texture and pattern, where "physical components" do not apply well.

### 4. Absence of a "Compact Explain" Variant
There is a significant gap between:
- Near-mute baseline.
- Free-form explanation.
- Feature schema.
- Rules.
- Heavy ontology.

A short, stable intermediate condition is missing.

---

## Practical Recommendations

To compare prompts seriously and migrate to OpenRouter with reasonable costs, I would prioritize this hierarchy:

1. Maintain `classification` as the primary baseline.
2. Refine `features` to be consistent across all datasets.
3. Maintain `nle` as a simple explanatory condition.
4. Keep `rulebased` as a secondary structured reasoning condition.
5. Treat `axioms_ontology_v2` as an exploratory experiment rather than a core condition.

---

## Recommended Improved Versions

To clean up the benchmark, I would propose the following family:

- `classification_strict`: Final class only.
- `classification_brief_justification`: 1-2 brief sentences.
- `classification_visual_cues`: 2-4 observable cues, no world knowledge.
- `classification_comparative`: Why this class and not others.
- `classification_rulecheck`: Compact version of rules, without ontology.

---

## Conclusion

The current prompts form a good progression of "increasing explanatory structure," but they are not all equally well-calibrated.

- `classification` is the most solid baseline.
- `features` is the most promising for useful interpretability, provided the feature definition inconsistency is corrected.
- `rulebased` can be useful if compacted.
- `axioms_ontology_v2` is an interesting stress test of formalism but is too heavy and prone to pseudo-explanations for use as a central condition.
