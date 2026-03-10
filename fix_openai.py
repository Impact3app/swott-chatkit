"""
fix_openai.py — Corrige automatiquement les bugs du code Python généré par OpenAI Agent Builder.
Usage : python fix_openai.py
"""

import re

FILE = "agents_openai.py"

with open(FILE, "r", encoding="utf-8") as f:
    code = f.read()

original = code

# FIX 1 : temperature=0["1"] → temperature=0.1
code = code.replace('temperature=0["1"]', 'temperature=0.1')

# FIX 2 : temperature=0["21"] → temperature=0.21
code = code.replace('temperature=0["21"]', 'temperature=0.21')

# FIX 3 : limit: integer → limit: int
code = code.replace('limit: integer', 'limit: int')

# FIX 4 : Supprimer tout à partir de "class WorkflowInput" jusqu'à la fin
match = re.search(r'^class WorkflowInput', code, re.MULTILINE)
if match:
    code = code[:match.start()].rstrip() + "\n"

# Compter les corrections
fixes = 0
if 'temperature=0["1"]' in original: fixes += 1
if 'temperature=0["21"]' in original: fixes += 1
if 'limit: integer' in original: fixes += 1
if re.search(r'^class WorkflowInput', original, re.MULTILINE): fixes += 1

with open(FILE, "w", encoding="utf-8") as f:
    f.write(code)

print(f"[fix_openai] {fixes} correction(s) appliquée(s) sur {FILE}")
print("[fix_openai] OK — prêt pour git add / commit / push")
