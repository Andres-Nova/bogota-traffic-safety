"""Configuración de pytest: añade la raíz del proyecto al sys.path."""
import sys
from pathlib import Path

# Permite importar src.* y dashboard.* desde cualquier test
sys.path.insert(0, str(Path(__file__).parent))
