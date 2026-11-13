"""Read the published release for status displays; candidates stay separate."""
import json
import logging
from pathlib import Path
from app.config import settings

logger = logging.getLogger(__name__)


def read_active_release(root=None):
    path = Path(root or settings.model_path)/'active.json'
    try:
        value = json.loads(path.read_text(encoding='utf-8'))
        return value if isinstance(value,dict) and value.get('version') else None
    except FileNotFoundError:
        return None
    except (OSError, ValueError):
        logger.warning('Unable to read published model release')
        return None
