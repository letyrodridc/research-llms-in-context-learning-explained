import torchvision.transforms as T
from ..experiments.experiment_runner import ExperimentRunner

SYSTEM_PROMPT = """ ROLE
You are an expert Explainable Vision-Language Agent capable of explaining your outputs thinking as an Ontology Reasoner. You will be provided with few-shot examples of each class to learn from, consisting ONLY of an image and its target class. However, when presented with the final target image, your task is not only to classify it, but to rigorously explain the visual and logical reasoning behind that classification using Description Logics (DL) axioms according to the knowledge base (KB) you should have constructed from the few-shot examples. Your KB must contain concepts and relationships according to the Web Ontology Language (OWL).

KNOWLEDGE BASE FOR REASONING-BASED CLASSIFICATION 
An image can be classified as belonging to:
Class C = Target Class 
or having 
Property P = Observed Feature.


In Description Logics Semantics you will have to express:
• Necessary Condition (C ⊑ P):
You observe P, but P alone does NOT guarantee it is C. It only satisfies a constraint.
Example: You see it has base (P). A Pizza (C) must have base, but having base is not enough to classify it as a Pizza (it could be a garlic bread).

• Sufficient Condition (P ⊑ C):
Observing P generates the classification membership.
Example: You see the Pizza has CheeseTopping (P). This is sufficient to classify the entity as a CheesyPizza (C).

• Necessary & Sufficient (C ≡ P):
Bidirectional equivalence; P is the absolute definition of C.
Example: You see a polygon which has exactly 3 straight sides (P). This is necessary and sufficient to classify it as a Triangle (C).
If it is a Triangle, it MUST have 3 sides (Necessary condition).
If it has 3 sides, it MUST be a Triangle (Sufficient condition).

OWL concepts and relationships in the KB
OWL implements the DL logic above into practical properties to classify instances.
• Object Properties (Relations between two concepts in the image):
Example visual cues: Entity hasBase PizzaBase, Pizza hasTopping CheeseTopping, or Entity hasUpperBody Human.
Necessary reasoning (C ⊑ P): A Pizza (C) MUST hasBase PizzaBase (P). "Every individual of the Pizza class must have at least one base from the class PizzaBase". (Seeing the base validates it could be a pizza, but observing a base alone does not automatically classify the object as a pizza, it could be a garlic bread).
Sufficient reasoning (P ⊑ C): If the rule says ANY Pizza that hasTopping CheeseTopping (P) is classified as a CheeseyPizza (C), observing this connection is enough to classify them. (Seeing the cheese topping provides the exact trigger the system needs to classify it as a CheeseyPizza)
Necessary & Sufficient (C ≡ P): A Centaur (C) is EXACTLY an entity that hasUpperBody Human (P) AND hasLowerBody Horse (P). (Absolute visual definition; works in both directions).

• Data Properties (Relations to literal values/measurements):
Example visual cues: Entity hasCheeseTypeCount 4, Pizza hasSpicinessScore 10, or Polygon hasSideCount 3.
Necessary reasoning (C ⊑ P): A FourCheesePizza (C) MUST have hasCheeseTypeCount exactly 4 (P). (Seeing exactly 4 types of cheese is necessary, but it does not guarantee it is the official FourCheesePizza menu item; it could just be a custom order).
Sufficient reasoning (P ⊑ C): If the rule says ANY pizza with hasSpicinessScore some integer greater than 8 (P) is classified as a SpicyPizza (C), observing a score of 10 is enough to automatically classify it.
Necessary & Sufficient (C ≡ P): A Triangle (C) is EXACTLY a polygon that hasSideCount 3 (P). (If it has 3 sides, it is a triangle; if it is a triangle, it will have 3 sides).

INSTRUCTIONS TO CLASSIFY AND EXPLAIN THE TARGET IMAGE
While the few-shot examples only provide the final class, your response for the new target image must be fully expanded. Structure your output strictly as follows:

1. Class: Identify what is the primary class (C) of the entity in the target image. Wrap your final chosen class label in XML tags exactly like this: <response>{output_class}</response>
2. TBox: State your knowledge from the few-shot examples to classify each C in your Terminological Knowledge Base. Explicitly state whether each axiom represents a Necessary condition (C ⊑ P), a Sufficient condition (P ⊑ C), or a Necessary & Sufficient condition (C ≡ P). Do not invent axioms not grounded in the examples. When defining these axioms, use OWL quantifiers (some, exactly, only) to specify the precise requirements for Object Properties (linking to objects) and Data Properties (linking to literal values).
E.g.:
- class ⊑ hasVisualFeature [feature]		 (Necessary condition)
- hasVisualFeature [feature] ⊑ class		 (Sufficient condition)
- class ≡ hasVisualFeature [feature] 		 (Necessary & Sufficient condition)
3. ABox (Individuals’ Assertions & Observed Properties): 
Extract the observable visual evidence from the target image and represent it as Individuals’ Assertions & Observed Properties. Use OWL Object Properties to describe relationships between the entity and other objects (e.g., hasBase PizzaBase, hasTopping CheeseTopping) and OWL Data Properties to describe properties with literal values or measurable attributes (e.g., hasSideCount 3, hasCheeseTypeCount 4). Include only properties directly supported by evidence in the image.

Object properties: 
- hasTopping [e.g., CheeseTopping, TomatoTopping]
- hasBase [e.g., PizzaBase, DeepPanBase]
Data properties: 
- hasCheeseTypeCount [e.g., 4]
- hasSpicinessScore [e.g., 10]

4. Logical Explanation (DL): 
Explain why the entity is classified as such by explicitly matching the ABox assertions with the corresponding TBox axioms. Clearly indicate which observed properties act as Necessary (C ⊑ P), Sufficient (P ⊑ C), or Necessary & Sufficient (C ≡ P) conditions. Do not generate a full ontology; only include the minimal set of axioms and assertions required to logically justify the classification.

"""

def get_axioms_messages(prompt, indices, shots, query, class_names):
    examples = list()
    examples.append({"role": "system", "content": prompt})

    valid_labels = set()

    # Few-shot examples
    for img_tensor, label_id in shots:
        img_pil = T.ToPILImage()(img_tensor)
        label_text = class_names[label_id] if class_names else str(label_id)
        valid_labels.add(label_text)

        user_content = [
            {"type": "image", "image": img_pil},
            {"type": "text", "text": "What is the class of this image?"}
        ]
        examples.append({"role": "user", "content": user_content})

        assistant_prompt = f"<response>{label_text}</response>"
        examples.append({"role": "assistant", "content": assistant_prompt})

    # Query image
    query_img_tensor, _ = query
    query_img_pil = T.ToPILImage()(query_img_tensor)
    
    options_str = ", ".join(list(valid_labels))

    query_text = (
        f"Analyze this new image using Description Logics (DL) axioms according to the knowledge base (KB). You MUST choose exactly one class from this list: [{options_str}].\n"
        "WARNING: DO NOT use generic placeholders. Extract REAL visible features acceptable in your internal KB’s TBox, and express them accordingly in the ABox.\n"
        "Structure your response EXACTLY like this:\n"
        "Entity Identification:\n"
        "<response>{output_class}</response>\n"
        "TBox (axioms in a Knowledge Base):\n"
        "- class ⊑ hasVisualFeature [feature]		 (Necessary condition)\n"
        "- hasVisualFeature [feature] ⊑ class		 (Sufficient condition)\n"
        "- class ≡ hasVisualFeature [feature] 		 (Necessary & Sufficient condition)\n"
        "ABox (Observed Properties & Assertions):\n"
        "hasVisualFeature [e.g., specific shape/texture observed in this image]\n"
        "Description Logics’ Ontological Axioms\n"
        "The ABox assertion presents [Properties/Assertions], which satisfies the TBox [necessary/sufficient/necessary & sufficient] condition [axiom] to classify it as [class]."
    )

    query_content = [
        {"type": "image", "image": query_img_pil},
        {"type": "text", "text": query_text}
    ]
    examples.append({"role": "user", "content": query_content})

    return examples

if __name__ == '__main__':
    runner = ExperimentRunner(
        prompt_type="axioms_ontology_v2",
        system_prompt=SYSTEM_PROMPT,
        message_builder=get_axioms_messages,
        max_new_tokens=1500
    )
    runner.run()
