"""Download only the two pinned encoders used by the role-aware acceptance gate."""

import json
from pathlib import Path

from sentence_transformers import SentenceTransformer


def main():
    models = json.loads((Path(__file__).resolve().parents[1] / "tests/integration/role_models.json").read_text())
    for spec in models:
        model = SentenceTransformer(spec["name"], revision=spec["revision"], device="cpu")
        print(f"Cached {spec['name']} revision={spec['revision']} dimension={model.get_embedding_dimension()}")


if __name__ == "__main__":
    main()
