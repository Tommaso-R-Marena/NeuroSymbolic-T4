# Usage Guide

## Table of Contents

1. [Basic Usage](#basic-usage)
2. [Training](#training)
3. [Inference](#inference)
4. [Custom Rules](#custom-rules)
5. [Explanation Generation](#explanation-generation)
6. [Advanced Topics](#advanced-topics)

## Basic Usage

### Minimal Example

```python
import torch
from neurosymbolic import NeurosymbolicSystem

# Initialize
model = NeurosymbolicSystem()
model.eval()

# Process image
image = torch.randn(1, 3, 224, 224)
with torch.no_grad():
    output = model.forward(image)
```

### Understanding Output

```python
output = {
    "perception": {
        "neural": {
            "features": torch.Tensor,      # [B, feature_dim]
            "concepts": torch.Tensor,      # [B, num_concepts]
            "bboxes": torch.Tensor,        # [B, 4]
            "confidence": torch.Tensor,    # [B, 1]
        },
        "symbolic": [                      # List per batch
            [("concept1", 0.9), ("concept2", 0.7), ...]
        ]
    },
    "reasoning": [                         # List per batch
        {
            "num_derived": int,
            "derived_facts": [(pred, args, conf), ...],
            "all_facts": [Fact, ...],
        }
    ]
}
```

## Training

### Custom Dataset

```python
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, image_paths, concept_labels):
        self.image_paths = image_paths
        self.labels = concept_labels
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image = load_image(self.image_paths[idx])
        concepts = self.labels[idx]  # Multi-hot vector
        return image, concepts
```

### Training Loop

```python
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast

model = NeurosymbolicSystem()
optimizer = optim.AdamW(model.parameters(), lr=1e-3)
criterion = nn.BCELoss()
scaler = GradScaler()

for epoch in range(num_epochs):
    for images, concepts in dataloader:
        optimizer.zero_grad()
        
        with autocast():
            outputs = model.perception(images)
            loss = criterion(outputs["concepts"], concepts)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
```

### Fine-tuning

```python
# Load pre-trained weights
checkpoint = torch.load("pretrained.pt")
model.load_state_dict(checkpoint["model_state_dict"])

# Freeze backbone
for param in model.perception.backbone.parameters():
    param.requires_grad = False

# Train only heads
optimizer = optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-4
)
```

## Inference

### Batch Processing

```python
images = torch.stack([load_image(p) for p in image_paths])

with torch.no_grad():
    outputs = model.forward(images, threshold=0.6)

for i, scene in enumerate(outputs["perception"]["symbolic"]):
    print(f"Image {i}:")
    for concept, conf in scene:
        print(f"  {concept}: {conf:.3f}")
```

### Real-time Processing

```python
import cv2

cap = cv2.VideoCapture(0)
model.eval()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Preprocess
    image = preprocess(frame)  # Your preprocessing
    image = image.unsqueeze(0).cuda()
    
    # Inference
    with torch.no_grad():
        output = model.forward(image)
    
    # Display results
    display_results(frame, output)
```

## Custom Rules

### Adding Rules

```python
# Taxonomic rule
model.reasoner.add_rule(
    head=("animal", ("?x",)),
    body=[("dog", ("?x",))],
    confidence=1.0
)

# Multi-condition rule
model.reasoner.add_rule(
    head=("dangerous_situation", ("?s",)),
    body=[
        ("vehicle", ("?s",)),
        ("moving", ("?s",)),
        ("person", ("?s",)),
        ("near", ("?s",))
    ],
    confidence=0.95
)

# Spatial reasoning
model.reasoner.add_rule(
    head=("near", ("?x", "?y")),
    body=[("near", ("?y", "?x"))],  # Symmetry
    confidence=1.0
)
```

### Rule Templates

```python
# Create rule generator
def create_is_a_rule(specific, general, confidence=1.0):
    return (
        (general, ("?x",)),
        [(specific, ("?x",))],
        confidence
    )

# Use template
for specific, general in [("car", "vehicle"), ("truck", "vehicle")]:
    head, body, conf = create_is_a_rule(specific, general)
    model.reasoner.add_rule(head, body, conf)
```

## Explanation Generation

### Basic Explanation

```python
image = torch.randn(1, 3, 224, 224)
fact = ("dangerous", ("obj0",))

explanations = model.explain_prediction(image, fact)

for exp in explanations:
    print(exp)
    print()
```

### Interactive Explanation

```python
def explain_interactively(model, image):
    # Get all derived facts
    output = model.forward(image)
    facts = output["reasoning"][0]["derived_facts"]
    
    print("Derived facts:")
    for i, (pred, args, conf) in enumerate(facts):
        print(f"{i}: {pred}{args} [{conf:.3f}]")
    
    # Let user choose
    choice = int(input("\nExplain which fact? "))
    pred, args, _ = facts[choice]
    
    # Generate explanation
    explanations = model.reasoner.explain((pred, args))
    
    print(f"\nExplanations for {pred}{args}:")
    for exp in explanations:
        print(exp)
        print()
```

### Proof Visualization

```python
def visualize_proof(proof):
    """Create ASCII tree of proof."""
    def tree(proof, indent=0):
        prefix = "  " * indent
        print(f"{prefix}└─ {proof['proof'][0]} [{proof['confidence']:.3f}]")
        # Add children if needed
    
    tree(proof)
```

## Advanced Topics

### Multi-Object Scenes

```python
# Process multiple objects
for obj_id in range(num_objects):
    scene = extract_object_features(image, obj_id)
    
    # Add facts for this object
    for concept, conf in scene:
        model.reasoner.add_fact(concept, (f"obj{obj_id}",), conf)
    
    # Add spatial relations
    for other_id in range(obj_id):
        if is_near(obj_id, other_id):
            model.reasoner.add_fact(
                "near",
                (f"obj{obj_id}", f"obj{other_id}"),
                0.9
            )
```

### Temporal Reasoning

```python
# Track state over time
class TemporalReasoner:
    def __init__(self, model):
        self.model = model
        self.history = []
    
    def process_frame(self, image, timestamp):
        output = self.model.forward(image)
        self.history.append((timestamp, output))
        
        # Add temporal rules
        if len(self.history) >= 2:
            prev_t, prev = self.history[-2]
            curr_t, curr = self.history[-1]
            
            # Detect changes
            self.detect_motion(prev, curr)
    
    def detect_motion(self, prev, curr):
        # Compare states
        pass
```

### Uncertainty Handling

```python
# Threshold tuning
thresholds = [0.3, 0.5, 0.7, 0.9]

for t in thresholds:
    output = model.forward(image, threshold=t)
    num_concepts = len(output["perception"]["symbolic"][0])
    num_derived = output["reasoning"][0]["num_derived"]
    
    print(f"Threshold {t}: {num_concepts} concepts, {num_derived} derived")
```

### Ensemble Reasoning

```python
# Multiple reasoning strategies
results = []

# Strategy 1: Conservative (high threshold)
output1 = model.forward(image, threshold=0.8)
results.append(output1)

# Strategy 2: Permissive (low threshold)
output2 = model.forward(image, threshold=0.4)
results.append(output2)

# Combine results
combined = combine_reasoning_outputs(results)
```