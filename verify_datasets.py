
import os
import sys
from data.multi_task_sample import AutoTask

def verify_datasets():
    tasks = ['boolq', 'piqa', 'openbookqa', 'arc_e', 'arc_c']
    print(f"Checking datasets: {tasks}\n")

    for task_name in tasks:
        print(f"--- Verifying {task_name} ---")
        try:
            task_cls = AutoTask.get(task_name)
            # Load validation set as used in evaluation
            dataset = task_cls.get_dataset(split="validation")
            
            print(f"Dataset loaded successfully.")
            print(f"Size: {len(dataset)}")
            print(f"Label List: {getattr(task_cls, 'label_list', 'N/A')}")
            print(f"Target Map: {getattr(task_cls, 'target_map', 'N/A')}")
            
            if len(dataset) > 0:
                example = dataset[0]
                print(f"Sample[0] Prompt: {example['prompt'][:100]}...") # Truncate for display
                print(f"Sample[0] Label: {example['label']}")
                
                # Check label type
                label = example['label']
                print(f"Label Type: {type(label)}")
                
                # Verify if label needs mapping
                target_map = getattr(task_cls, 'target_map', {})
                if target_map:
                    if label in target_map:
                        print(f"Label '{label}' maps to '{target_map[label]}'")
                    elif label in target_map.values():
                        print(f"Label '{label}' is already a mapped value.")
                    else:
                        print(f"WARNING: Label '{label}' is NOT in target_map keys or values!")
            else:
                print("Dataset is empty!")
                
        except Exception as e:
            print(f"ERROR loading {task_name}: {e}")
        
        print("\n")

if __name__ == "__main__":
    verify_datasets()
