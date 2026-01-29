#!/bin/bash
set -e

echo "🚨 EMERGENCY FIX - Applying tuple unpacking fixes..."

cd "$(dirname "$0")"

# Fix Cell 6: perception demo
sed -i 's/for i, (concept, confidence) in enumerate(sorted(symbolic_scene/for i, (concept, confidence, _) in enumerate(sorted(symbolic_scene/g' notebooks/NeuroSymbolic_T4_Demo.ipynb

# Fix Cell 14: ICML benchmark
sed -i 's/avg_conf = np.mean(\[c for _, c in symbolic\])/avg_conf = np.mean([c for _, c, _ in symbolic])/g' notebooks/NeuroSymbolic_T4_Demo.ipynb

echo "✅ Fixes applied!"
echo "📋 Validating JSON..."

python3 -c "import json; json.load(open('notebooks/NeuroSymbolic_T4_Demo.ipynb'))" && echo "✓ Valid JSON"

echo ""
echo "Now commit and push:"
echo "  git add notebooks/NeuroSymbolic_T4_Demo.ipynb"
echo "  git commit -m 'Fix tuple unpacking for ICML submission'"
echo "  git push origin main"
