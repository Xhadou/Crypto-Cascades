"""
Checkpoint Manager for Pipeline Intermediate Results

Saves and loads intermediate computation results (graph, communities,
states) so expensive steps can be skipped on re-run.  Checkpoints are
automatically invalidated when config.yaml or relevant source files
change.
"""

import hashlib
import json
import pickle
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from src.utils.logger import get_logger

# Source files whose changes should invalidate checkpoints
_WATCHED_SOURCES = [
    'src/network_analysis/metrics.py',
    'src/network_analysis/community_detection.py',
    'src/state_engine/state_assigner.py',
    'src/preprocessing/graph_builder.py',
    'src/preprocessing/orbitaal_parser.py',
]


class CheckpointManager:
    """Save / load intermediate pipeline results with hash validation."""

    def __init__(
        self,
        checkpoint_dir: Path,
        config_path: Path,
        project_root: Optional[Path] = None,
    ):
        self.logger = get_logger(__name__)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.checkpoint_dir / 'manifest.json'
        self.project_root = Path(project_root) if project_root else Path('.')

        current_hash = self._compute_hash(config_path)
        saved_hash = self._read_manifest_hash()

        if saved_hash and saved_hash != current_hash:
            self.logger.warning(
                "Config or code changed since last run. "
                "Invalidating all checkpoints."
            )
            self.invalidate()

        self._current_hash = current_hash
        self._manifest = self._load_manifest()

    # ------------------------------------------------------------------
    # Hash computation
    # ------------------------------------------------------------------

    def _compute_hash(self, config_path: Path) -> str:
        """SHA-256 over config contents + watched source file mtimes."""
        h = hashlib.sha256()

        # Hash config contents
        try:
            h.update(Path(config_path).read_bytes())
        except FileNotFoundError:
            h.update(b'no-config')

        # Hash mtimes of watched source files
        for rel in _WATCHED_SOURCES:
            p = self.project_root / rel
            try:
                mtime = str(p.stat().st_mtime)
                h.update(f"{rel}:{mtime}".encode())
            except FileNotFoundError:
                h.update(f"{rel}:missing".encode())

        return h.hexdigest()

    # ------------------------------------------------------------------
    # Manifest I/O
    # ------------------------------------------------------------------

    def _load_manifest(self) -> Dict:
        if self.manifest_path.exists():
            try:
                return json.loads(self.manifest_path.read_text())
            except (json.JSONDecodeError, OSError):
                return {'hash': self._current_hash, 'steps': {}}
        return {'hash': self._current_hash, 'steps': {}}

    def _save_manifest(self) -> None:
        self._manifest['hash'] = self._current_hash
        self.manifest_path.write_text(json.dumps(self._manifest, indent=2))

    def _read_manifest_hash(self) -> Optional[str]:
        if self.manifest_path.exists():
            try:
                data = json.loads(self.manifest_path.read_text())
                return data.get('hash')
            except (json.JSONDecodeError, OSError):
                return None
        return None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def has(self, step: str) -> bool:
        """Return True if a valid checkpoint exists for *step*."""
        info = self._manifest.get('steps', {}).get(step)
        if not info:
            return False
        fpath = self.checkpoint_dir / info['file']
        return fpath.exists()

    def save(self, step: str, data: Any, metadata: Optional[Dict] = None) -> None:
        """Persist *data* as a checkpoint for *step*.

        Format is auto-selected:
        - ``pd.DataFrame`` → parquet
        - JSON-serialisable dict (string keys, simple values) → json
        - Everything else → pickle
        """
        if isinstance(data, pd.DataFrame):
            fname = f"{step}.parquet"
            data.to_parquet(self.checkpoint_dir / fname)
        elif isinstance(data, dict) and self._is_json_safe(data):
            fname = f"{step}.json"
            (self.checkpoint_dir / fname).write_text(
                json.dumps(data, default=str)
            )
        else:
            fname = f"{step}.pkl"
            with open(self.checkpoint_dir / fname, 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

        entry = {
            'file': fname,
            'timestamp': datetime.now().isoformat(),
        }
        if metadata:
            entry.update(metadata)

        self._manifest.setdefault('steps', {})[step] = entry
        self._save_manifest()
        self.logger.debug(f"Checkpoint saved: {step} → {fname}")

    def load(self, step: str) -> Any:
        """Load checkpoint data for *step*."""
        info = self._manifest['steps'][step]
        fpath = self.checkpoint_dir / info['file']

        if fpath.suffix == '.parquet':
            return pd.read_parquet(fpath)
        elif fpath.suffix == '.json':
            return json.loads(fpath.read_text())
        else:
            with open(fpath, 'rb') as f:
                return pickle.load(f)

    def invalidate(self) -> None:
        """Remove all checkpoint files and the manifest."""
        if self.checkpoint_dir.exists():
            for child in self.checkpoint_dir.iterdir():
                if child.is_file():
                    child.unlink()
            self.logger.info("All checkpoints invalidated.")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_json_safe(obj: Any, _depth: int = 0) -> bool:
        """Quick check whether *obj* can be serialised as JSON."""
        if _depth > 3:
            return False
        if isinstance(obj, (str, int, float, bool, type(None))):
            return True
        if isinstance(obj, dict):
            return all(
                isinstance(k, str) and CheckpointManager._is_json_safe(v, _depth + 1)
                for k, v in list(obj.items())[:10]  # sample first 10
            )
        if isinstance(obj, (list, tuple)):
            return all(
                CheckpointManager._is_json_safe(v, _depth + 1)
                for v in list(obj)[:10]
            )
        return False
