import sys
from coverage import Coverage

def main(THRESHOLDS: dict[float]) -> None:
    """Main function for check_coverage"""
    cov = Coverage(data_file=".coverage")
    cov.load()
    errors = 0

    for path, min_frac in THRESHOLDS.items():
        try:
            _file, statements, _excluded, missing, _lines = cov.analysis2(path)
        except Exception as e:
            print(f"WARNING: no coverage data for {path!r} ({e})")
            continue

        total = len(statements)
        covered = total - len(missing)
        frac = covered/total if total else 1.0

        if frac < min_frac:
            print(f"FAIL {path}: {covered}/{total} lines covered ({frac:.0%} < {min_frac:.0%})")
            errors += 1
        else:
            print(f"PASS {path}: {covered}/{total} lines covered ({frac:.0%} >= {min_frac:.0%})")

    sys.exit(errors)

if __name__ == "__main__":

    THRESHOLDS = {
        "RAGToolBox/loader.py": 0.70,
        "RAGToolBox/chunk.py": 0.80,
        "RAGToolBox/vector_store.py": 0.80,
        "RAGToolBox/index.py": 0.50,
        "RAGToolBox/retriever.py": 0.70,
        "RAGToolBox/augmenter.py": 0.70,
        }

    sys.exit(main(THRESHOLDS=THRESHOLDS))
