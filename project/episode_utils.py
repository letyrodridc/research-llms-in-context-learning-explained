import numpy as np
import os
import random

def select_w_images_from_class(class_indices_map, class_id, w_count, exclude_indices=None):
    """
    Selects W images from a specific class, avoiding specific indices.
    """
    if exclude_indices is None:
        exclude_indices = set()
        
    candidates = [
        idx for idx in class_indices_map[class_id] 
        if idx not in exclude_indices
    ]
    
    if len(candidates) < w_count:
        raise ValueError(f"Not enough images in class {class_id}. Requested {w_count}, found {len(candidates)}.")
    
    selected_indices = random.sample(candidates, w_count)
    return selected_indices

def create_and_save_episode_indices(class_indices_map, num_classes, num_shots, num_queries, save_dir='episodes', fixed_classes=None, run_id=0):
    """
    Selects indices for an N-way K-shot episode and saves them to a .npy file.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if fixed_classes is not None:
        selected_class_ids = fixed_classes
    else:
        available_classes = list(class_indices_map.keys())
        selected_class_ids = random.sample(available_classes, num_classes)
    
    support_indices = []
    query_indices = []
    used_indices = set() 
    
    for class_id in selected_class_ids:
        shots = select_w_images_from_class(
            class_indices_map, 
            class_id, 
            num_shots, 
            exclude_indices=used_indices
        )
        support_indices.extend(shots)
        used_indices.update(shots)

    for class_id in selected_class_ids:
        queries = select_w_images_from_class(
            class_indices_map, 
            class_id, 
            num_queries, 
            exclude_indices=used_indices
        )
        query_indices.extend(queries)
        used_indices.update(queries)

    episode_data = {
        "support_indices": np.array(support_indices),
        "query_indices": np.array(query_indices),
        "selected_class_ids": np.array(selected_class_ids)
    }

    filename = f"episode_N{num_classes}_K{num_shots}_Q{num_queries}_run{run_id}.npy"
    save_path = os.path.join(save_dir, filename)
    
    np.save(save_path, episode_data)
    print(f"Episode indices saved to: {save_path}")
    return save_path

def load_episode_from_indices(filepath, dataset, class_names=None):
    """
    Loads episode indices from a .npy file and constructs the data object.
    """
    episode_data = np.load(filepath, allow_pickle=True).item()

    support_indices = episode_data["support_indices"]
    query_indices = episode_data["query_indices"]
    selected_class_ids = episode_data["selected_class_ids"]

    support_set = [dataset[i] for i in support_indices]
    query_set = [dataset[i] for i in query_indices]

    selected_names = [class_names[i] for i in selected_class_ids] if class_names else selected_class_ids

    return {
        "support_indices": support_indices,
        "query_indices": query_indices,
        "support_set": support_set,
        "query_set": query_set,
        "classes": selected_names
    }
