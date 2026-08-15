from pathlib import Path


def test_dd_and_diagnostics_cleanup_structure():
    package = Path(__file__).resolve().parents[1] / "src" / "complextorch"
    assert (package / "dd.py").is_file()
    assert (package / "dd_optimization.py").is_file()
    assert not (package / "dd_riemannian.py").exists()
    assert not (package / "dd_ssdi.py").exists()
    assert not (package / "_diagnostics_evaluation.py").exists()
