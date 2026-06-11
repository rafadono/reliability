#!/usr/bin/env python3
"""
Script to validate that imports work correctly after migration.
"""

import sys
from pathlib import Path

# Add backend/ and backend/src/ to system path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "backend"))
sys.path.insert(0, str(project_root / "backend" / "src"))

print("=" * 70)
print("IMPORT VALIDATION")
print("=" * 70)

try_imports = [
    ("reliability_analysis.core.filters", "FilterManager"),
    ("reliability_analysis.core.data_processing", "DataProcessor"),
    ("reliability_analysis.analysis.models", ["ReliabilityFitter", "KijimaFitter"]),
    ("reliability_analysis.analysis.metrics", ["calculate_aic_bic"]),
    ("reliability_analysis.analysis.kijima_model", "calculate_k2"),
    ("reliability_analysis.utils.config", "EXCLUDED_MODELS"),
    ("reliability_analysis.utils.logger_config", "setup_logging"),
]

print("\nAttempting to import modules:")
success_count = 0
for module_path, *items in try_imports:
    try:
        module = __import__(module_path, fromlist=[""])
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
print(f"Result: {success_count} modules successfully imported")
print("=" * 70)

if success_count == len(try_imports):
    print("\nAll imports function correctly!")
    sys.exit(0)
else:
    print(f"\nAlert: {len(try_imports) - success_count} imports failed")
    sys.exit(1)
