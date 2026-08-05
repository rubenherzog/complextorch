import json
from pathlib import Path


def test_example_notebooks_are_valid_and_compile():
    root = Path(__file__).resolve().parents[1] / "examples"
    notebooks = sorted(root.glob("*.ipynb"))
    assert {path.name for path in notebooks} == {
        "model_based_validation.ipynb",
        "mvgc_tutorial.ipynb",
    }
    for path in notebooks:
        notebook = json.loads(path.read_text(encoding="utf-8"))
        assert notebook["nbformat"] == 4
        assert notebook["cells"]
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] == "code":
                compile("".join(cell["source"]), f"{path.name}:cell-{index}", "exec")
