import json, sys
sys.stdout.reconfigure(encoding='utf-8')

nb = json.load(open('final_report_notebook.ipynb', 'r', encoding='utf-8'))
for i, c in enumerate(nb['cells']):
    if c['cell_type'] == 'markdown':
        print(f"===== CELL {i} =====")
        print(''.join(c['source']))
        print()
