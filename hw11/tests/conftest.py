import sys
from pathlib import Path
root=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(root),str(root.parent/'hw8910')]
import service
sys.modules['previous_service']=service
