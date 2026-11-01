"""One fast entry point; weights, MLflow and real inference are not imported."""
from pathlib import Path
import sys
import unittest

root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root))
loader=unittest.TestLoader()
suite=unittest.TestSuite([loader.discover(str(root/'tests')),
                         loader.discover(str(root/'integration_tests'),top_level_dir=str(root/'integration_tests'))])
result=unittest.TextTestRunner(verbosity=2).run(suite)
raise SystemExit(not result.wasSuccessful())
