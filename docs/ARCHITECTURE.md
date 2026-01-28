# Architecture Documentation

## Overview

NeuroSymbolic-T4 integrates neural perception with symbolic reasoning through a three-layer architecture:

1. **Neural Perception Layer**: Extract symbolic groundings from raw sensory input
2. **Symbolic Reasoning Layer**: Perform logical inference on symbolic representations
3. **Integration Layer**: Bridge neural and symbolic representations

## Neural Perception Module

### Components

#### Backbone Network
- **Architecture**: EfficientNet-B0
- **Input**: RGB images (224x224)
- **Output**: Spatial feature maps
- **Optimization**: Pre-trained on ImageNet, mixed precision training

#### Attention Pooling
- **Purpose**: Aggregate spatial features into global representation
- **Mechanism**: Multi-head attention with learnable query
- **Benefits**: Focus on relevant regions, better than average pooling

#### Concept Grounding Head
- **Input**: Global features
- **Output**: 100-dimensional concept vector
- **Activation**: Sigmoid (multi-label classification)
- **Concepts**: Objects, attributes, relations, actions, abstract properties

#### Object Detection Head
- **Bounding Box**: 4D coordinates (x, y, w, h)
- **Confidence**: Detection confidence score
- **Purpose**: Spatial grounding for reasoning

### Forward Pass

```
Image [B, 3, 224, 224]
  ↓
Backbone (EfficientNet)
  ↓
Spatial Features [B, C, H, W]
  ↓
Reshape + Attention Pool
  ↓
Global Features [B, feature_dim]
  ↓
├─ Concept Head → Concepts [B, 100]
├─ BBox Head → Boxes [B, 4]
└─ Confidence → Scores [B, 1]
```

## Symbolic Reasoning Engine

### Knowledge Representation

#### Facts
- **Structure**: `predicate(arg1, arg2, ...) [confidence]`
- **Example**: `car(obj1) [0.9]`
- **Storage**: Set with predicate indexing for fast retrieval

#### Rules
- **Structure**: `head :- body1 AND body2 AND ... [confidence]`
- **Example**: `vehicle(?x) :- car(?x) [1.0]`
- **Variables**: Start with `?` (e.g., `?x`, `?obj`)

### Inference Algorithms

#### Forward Chaining
1. Match rule bodies against known facts
2. Generate variable bindings
3. Instantiate rule heads with bindings
4. Compute confidence (product of rule and body confidences)
5. Add new facts above threshold
6. Repeat until fixpoint

**Complexity**: O(R * F^B) where R=rules, F=facts, B=max body size

#### Backward Chaining
1. Start with goal query
2. Find matching facts (direct proof)
3. Find matching rules
4. Recursively prove rule bodies
5. Combine proofs with confidence propagation
6. Return all proofs above threshold

**Features**:
- Goal-directed (only proves what's needed)
- Generates proof trees
- Supports explanation generation

### Probabilistic Logic

#### Confidence Propagation

**Conjunction** (AND):
```
C(A AND B) = C(A) * C(B)
```

**Rule Application**:
```
C(head) = C(rule) * ∏ C(body_i)
```

**Multiple Proofs**:
```
C(fact) = max(C(proof_i))
```

## Integration Layer

### Perception → Symbolic

```python
# Neural concepts → Symbolic facts
concepts = perception_output["concepts"]  # [B, 100]
for i, prob in enumerate(concepts[0]):
    if prob > threshold:
        reasoner.add_fact(concept_names[i], ("obj0",), prob)
```

### Reasoning Loop

```
1. Perceive(image) → concepts with confidence
2. Add concepts as facts to knowledge base
3. Forward chain to derive new facts
4. Query specific goals if needed
5. Generate explanations for predictions
```

## T4 GPU Optimizations

### Mixed Precision Training
- **FP16**: Forward pass, gradient computation
- **FP32**: Weight updates, loss computation
- **Benefit**: 2-3x speedup, 50% memory reduction

### Memory Optimization
- Efficient backbone (EfficientNet-B0)
- Batch size tuning (optimal: 32 for T4)
- Gradient accumulation for larger effective batches

### Compute Optimization
- TensorCore utilization (automatic in PyTorch 2.0+)
- Optimal matrix dimensions (multiples of 8)
- Kernel fusion via TorchScript

## Design Decisions

### Why EfficientNet?
- Best accuracy/efficiency trade-off
- Optimized for mobile/edge (perfect for T4)
- Compound scaling (depth, width, resolution)

### Why Forward + Backward Chaining?
- Forward: Efficient for deriving all consequences
- Backward: Efficient for specific queries
- Complementary strengths

### Why Probabilistic Logic?
- Neural outputs are inherently uncertain
- Real-world knowledge is incomplete
- Enables confidence-aware reasoning

## Extension Points

### Adding New Concepts
```python
model.concept_names.extend(["new_concept1", "new_concept2"])
model.concept_to_idx = {name: i for i, name in enumerate(model.concept_names)}
```

### Adding Domain Rules
```python
model.reasoner.add_rule(
    head=("domain_property", ("?x",)),
    body=[("condition1", ("?x",)), ("condition2", ("?x",))],
    confidence=0.9
)
```

### Custom Backbones
```python
model = NeurosymbolicSystem(
    perception_config={
        "backbone": "resnet50",  # Any timm model
        "feature_dim": 1024,
    }
)
```

## Performance Characteristics

### Perception
- **Latency**: ~20-25ms (T4, batch=1)
- **Throughput**: ~40-50 FPS
- **Memory**: ~2GB VRAM

### Reasoning
- **Forward Chain**: O(rules * facts^2) ≈ 1-10ms
- **Backward Chain**: O(depth * branching) ≈ <1ms per query
- **Scalability**: Up to 10K facts, 1K rules efficiently

### End-to-End
- **Total Latency**: ~25-35ms
- **Bottleneck**: Neural perception (95% of time)
- **Reasoning Overhead**: <5%