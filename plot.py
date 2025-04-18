import os
import matplotlib.pyplot as plt

# Base paths
data_dir = 'results'
torch_dir = 'torch-results'

# Organize data
sparsity_data = {
    'cpu+full': {},
    'cpu+layer': {},
    'cuda+full': {},
    'cuda+layer': {}
}

torch_lines = {
    'cpu+full': [],
    'cpu+layer': [],
    'cuda+full': [],
    'cuda+layer': []
}

# Helper to classify files
def classify(name):
    name = name.lower()
    if 'cpu' in name:
        device = 'cpu'
    elif 'cuda' in name:
        device = 'cuda'
    else:
        return None
    if 'full' in name:
        mode = 'full'
    elif 'layer' in name:
        mode = 'layer'
    else:
        return None
    return f'{device}+{mode}'

# Read sparsity vs time data
for filename in sorted(os.listdir(data_dir)):
    if filename.endswith('.txt') and not filename.startswith('torch'):
        path = os.path.join(data_dir, filename)
        key = classify(filename)
        if not key:
            continue
        with open(path) as f:
            sparsities, times = [], []
            for line in f:
                parts = line.strip().split(',')
                sparsity = float(parts[0].split(':')[1])
                time = float(parts[1].split(':')[1])
                sparsities.append(sparsity)
                times.append(time)
            sparsity_data[key][filename.replace('.txt', '')] = (sparsities, times)

# Read torch constant lines with comp status
for filename in sorted(os.listdir(torch_dir)):
    if filename.endswith('.txt'):
        key = classify(filename)
        if not key:
            continue
        label = 'torch-comp' if 'comptrue' in filename else 'torch-no-comp'
        path = os.path.join(torch_dir, filename)
        with open(path) as f:
            value = float(f.read().strip())
            torch_lines[key].append((value, label))

# Create plots
fig, axs = plt.subplots(2, 2, figsize=(12, 10))
combo_keys = ['cpu+full', 'cpu+layer', 'cuda+full', 'cuda+layer']
titles = ['bert-full (CPU)', 'bert-layer-0 (CPU)', 'bert-full (CUDA)', 'bert-layer-0 (CUDA)']

for ax, key, title in zip(axs.flat, combo_keys, titles):
    # Plot curves
    for label, (sparsities, times) in sparsity_data[key].items():
        label = label.split("-")[0]
        ax.plot(sparsities, times, marker='o', label=label)
    
    # Add torch lines with labels
    for val, label in torch_lines[key]:
        color = "green" if "no-comp" not in label else "red"

        ax.axhline(y=val, linestyle='--', color=color, label=label)

    ax.set_title(title)
    ax.set_xlabel('Sparsity')
    ax.set_ylabel('Time (s)')
    ax.grid(True)
    ax.legend()

plt.tight_layout()
plt.show()

