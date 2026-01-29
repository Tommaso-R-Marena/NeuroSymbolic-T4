#!/usr/bin/env python3
"""
Fix tuple unpacking in NeuroSymbolic_T4_Demo.ipynb

This script updates the demo notebook to handle the 3-tuple format
(concept, confidence, grounded_name) returned by the perception module.
"""

import json
import sys
from pathlib import Path

# Get repo root
repo_root = Path(__file__).parent.parent
notebook_path = repo_root / 'notebooks' / 'NeuroSymbolic_T4_Demo.ipynb'

print(f"Fixing notebook: {notebook_path}")
print("=" * 70)

# Read the notebook
try:
    with open(notebook_path, 'r', encoding='utf-8') as f:
        content = f.read()
    print("✓ Loaded notebook")
except FileNotFoundError:
    print(f"✗ Error: Notebook not found at {notebook_path}")
    sys.exit(1)

# Apply Fix 1: Cell 6 (perception demo)
fix1_old = 'for i, (concept, confidence) in enumerate(sorted(symbolic_scene'
fix1_new = 'for i, (concept, confidence, _) in enumerate(sorted(symbolic_scene'

if fix1_old in content:
    content = content.replace(fix1_old, fix1_new)
    print("✓ Applied Fix 1: Cell 6 perception demo")
else:
    print("⚠ Fix 1 pattern not found (may already be fixed)")

# Apply Fix 2: Cell 9/14 (ICML benchmark)
fix2_old = 'avg_conf = np.mean([c for _, c in symbolic])'
fix2_new = 'avg_conf = np.mean([c for _, c, _ in symbolic])'

if fix2_old in content:
    content = content.replace(fix2_old, fix2_new)
    print("✓ Applied Fix 2: Cell 9/14 ICML benchmark")
else:
    print("⚠ Fix 2 pattern not found (may already be fixed)")

# Validate JSON
try:
    nb = json.loads(content)
    num_cells = len(nb['cells'])
    print(f"✓ Valid JSON with {num_cells} cells")
except json.JSONDecodeError as e:
    print(f"✗ JSON validation error: {e}")
    sys.exit(1)

# Write back
with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=0, ensure_ascii=False)

print("✓ Fixes applied and saved successfully!")
print("=" * 70)
print("\nYou can now open the notebook in Google Colab.")
