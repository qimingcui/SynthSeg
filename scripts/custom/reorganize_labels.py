import numpy as np

def reorganize_arrays(generation_labels, generation_classes, classes_names, output_labels):
    """
    Reorganize arrays to follow the correct order for proper flipping:
    1. Background (0)
    2. Non-sided structures
    3. Left hemisphere structures
    4. Right hemisphere structures (in same order as left)
    """
    # Print debugging information
    print("Input arrays shapes:")
    print(f"generation_labels shape: {generation_labels.shape}")
    print(f"generation_classes shape: {generation_classes.shape}")
    print(f"classes_names shape: {classes_names.shape}")
    print(f"output_labels shape: {output_labels.shape}")
    
    print("\nUnique values in generation_labels:", np.unique(generation_labels))
    print("Unique values in generation_classes:", np.unique(generation_classes))
    
    # Initialize new arrays
    new_labels = np.zeros_like(generation_labels)
    new_classes = np.zeros_like(generation_classes)
    new_names = np.zeros_like(classes_names, dtype=object)
    new_output = np.zeros_like(output_labels)
    
    # Step 1: Put background first
    # Find the first occurrence of background (either 0 or the first label)
    bg_idx = 0  # Default to first position
    if 0 in generation_labels:
        bg_idx = np.where(generation_labels == 0)[0][0]
    new_labels[0] = generation_labels[bg_idx]
    new_classes[0] = generation_classes[bg_idx]
    new_names[0] = classes_names[bg_idx]
    new_output[0] = output_labels[bg_idx]
    
    # Step 2: Identify non-sided structures (ventricles, brainstem, CSF)
    non_sided = []
    for i, label in enumerate(generation_labels):
        if label in [4, 5, 14, 15, 16, 24, 43, 44]:  # Ventricles, brainstem, CSF
            non_sided.append(i)
    
    # Step 3: Identify left/right pairs
    left_structures = []
    right_structures = []
    for i, label in enumerate(generation_labels):
        if label in [2, 3, 7, 8, 10, 11, 12, 13, 17, 18, 26, 28]:  # Left structures
            left_structures.append(i)
        elif label in [41, 42, 46, 47, 49, 50, 51, 52, 53, 54, 58, 60]:  # Right structures
            right_structures.append(i)
    
    # Find any labels we haven't categorized
    all_handled_indices = set([bg_idx] + non_sided + left_structures + right_structures)
    all_indices = set(range(len(generation_labels)))
    unhandled_indices = all_indices - all_handled_indices
    
    if unhandled_indices:
        print("\nWARNING: Found unhandled labels:")
        for idx in unhandled_indices:
            print(f"Label {generation_labels[idx]}, Class {generation_classes[idx]}, Name {classes_names[idx]}")
        # Add unhandled labels to non_sided for now
        non_sided.extend(unhandled_indices)
    
    print("\nFound structures:")
    print(f"Non-sided structures: {[generation_labels[i] for i in non_sided]}")
    print(f"Left structures: {[generation_labels[i] for i in left_structures]}")
    print(f"Right structures: {[generation_labels[i] for i in right_structures]}")
    
    # Reorganize arrays
    current_idx = 1
    
    # Add non-sided structures
    for idx in non_sided:
        new_labels[current_idx] = generation_labels[idx]
        new_classes[current_idx] = generation_classes[idx]
        new_names[current_idx] = classes_names[idx]
        new_output[current_idx] = output_labels[idx]
        current_idx += 1
    
    # Add left structures
    for idx in left_structures:
        new_labels[current_idx] = generation_labels[idx]
        new_classes[current_idx] = generation_classes[idx]
        new_names[current_idx] = classes_names[idx]
        new_output[current_idx] = output_labels[idx]
        current_idx += 1
    
    # Add right structures
    for idx in right_structures:
        new_labels[current_idx] = generation_labels[idx]
        new_classes[current_idx] = generation_classes[idx]
        new_names[current_idx] = classes_names[idx]
        new_output[current_idx] = output_labels[idx]
        current_idx += 1
    
    return new_labels, new_classes, new_names, new_output

def main():
    # Load the original arrays
    print("Loading arrays...")
    generation_labels = np.load('priors/generation_labels.npy')
    generation_classes = np.load('priors/generation_classes.npy')
    classes_names = np.load('priors/classes_names.npy', allow_pickle=True)
    output_labels = np.load('priors/output_labels.npy')
    
    # Reorganize arrays
    print("\nReorganizing arrays...")
    new_labels, new_classes, new_names, new_output = reorganize_arrays(
        generation_labels, generation_classes, classes_names, output_labels
    )
    
    # Save the reorganized arrays
    print("\nSaving reorganized arrays...")
    np.save('priors/generation_labels_reorganized.npy', new_labels)
    np.save('priors/generation_classes_reorganized.npy', new_classes)
    np.save('priors/classes_names_reorganized.npy', new_names)
    np.save('priors/output_labels_reorganized.npy', new_output)
    
    # Print the new organization
    print("\nNew label organization:")
    for i, (label, class_, name, out) in enumerate(zip(new_labels, new_classes, new_names, new_output)):
        print(f"Position {i}: Label {label}, Class {class_}, Name {name}, Output {out}")

if __name__ == "__main__":
    main() 