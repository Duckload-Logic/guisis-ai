import sys
import os
import csv
from pathlib import Path

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", "..", ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.utils.text_cleaning import anonymize_text

path = Path('ai_models/distilbert/datasets/anonymizer_edge_cases.csv')
rows = list(csv.DictReader(path.open(newline='', encoding='utf-8')))
failures = []
for row in rows:
    actual = anonymize_text(row['input_text'])
    expected = row['expected_text']
    if actual != expected:
        failures.append((row['case_id'], actual, expected))

print('cases', len(rows))
print('failures', len(failures))
for case_id, actual, expected in failures[:20]:
    print('CASE', case_id)
    print('ACTUAL  ', actual)
    print('EXPECTED', expected)