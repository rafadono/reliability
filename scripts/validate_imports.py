#!/usr/bin/env python3
"""
Script para validar que los imports funcionan correctamente después de la migración.
"""

import sys
from pathlib import Path

# Agregar backend/ y backend/src/ al path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "backend"))
sys.path.insert(0, str(project_root / "backend" / "src"))

print("=" * 70)
print("VALIDACION DE IMPORTS")
print("=" * 70)

try_imports = [
    ("reliability_analysis.core.filters", "FilterManager"),
    ("reliability_analysis.core.data_processing", "DataProcessor"),
    ("reliability_analysis.analysis.models", ["ReliabilityFitter", "KijimaMontecarlo"]),
    ("reliability_analysis.analysis.metrics", ["calculate_aic_bic"]),
    ("reliability_analysis.analysis.kijima_model", "calculate_k2"),
    ("reliability_analysis.utils.config", "EXCLUDED_MODELS"),
    ("reliability_analysis.utils.logger_config", "setup_logging"),
]

print("\nIntentando importar modulos:")
success_count = 0
for module_path, *items in try_imports:
    try:
        module = __import__(module_path, fromlist=[''])
        item_list = items[0] if items else None
        
        if item_list:
            if isinstance(item_list, str) and item_list != "None":
                getattr(module, item_list)
                print(f"  [OK] {module_path}.{item_list}")
                success_count += 1
            elif isinstance(item_list, list):
                for item in item_list:
                    getattr(module, item)
                print(f"  [OK] {module_path} ({', '.join(item_list)})")
                success_count += 1
            else:
                print(f"  [OK] {module_path}")
                success_count += 1
        else:
            print(f"  [OK] {module_path}")
            success_count += 1
    except Exception as e:
        print(f"  [FAIL] {module_path}: {str(e)[:60]}")

print("\n" + "=" * 70)
print(f"Resultado: {success_count} modulos importados exitosamente")
print("=" * 70)

if success_count == len(try_imports):
    print("\nTodos los imports funcionan correctamente!")
    sys.exit(0)
else:
    print(f"\nAlerta: {len(try_imports) - success_count} imports fallaron")
    sys.exit(1)
