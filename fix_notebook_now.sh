#!/bin/bash
# Temporary script to apply fixes
curl -s "https://raw.githubusercontent.com/Tommaso-R-Marena/NeuroSymbolic-T4/87e8b23/notebooks/NeuroSymbolic_T4_Demo.ipynb" | \
sed 's/for i, (concept, confidence) in enumerate(sorted(symbolic_scene/for i, (concept, confidence, _) in enumerate(sorted(symbolic_scene/g' | \
sed 's/avg_conf = np.mean(\[c for _, c in symbolic\])/avg_conf = np.mean([c for _, c, _ in symbolic])/g' > /tmp/fixed_notebook.ipynb
echo "Fixed notebook saved to /tmp/fixed_notebook.ipynb"