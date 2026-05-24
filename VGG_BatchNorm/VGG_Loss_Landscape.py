"""Compatibility entry point for the provided loss-landscape script.

The original template intentionally left several sections blank.  The complete
implementation now lives in ``run_batchnorm_experiments.py`` so it can be run as
a normal Python module and reused from reports.
"""

from __future__ import annotations

try:
    from .run_batchnorm_experiments import main
except ImportError:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from VGG_BatchNorm.run_batchnorm_experiments import main


if __name__ == "__main__":
    main()
