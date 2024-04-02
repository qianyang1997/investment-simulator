from pathlib import Path
import yaml
from .logging import set_logger

REPO_DIR = Path(str(__file__)).parents[2].resolve()
DATA_DIR = REPO_DIR / "data"
CONF_DIR = REPO_DIR / "conf"

with open(f"{CONF_DIR}/credentials.yml", "r") as f:
    CREDS = yaml.safe_load(f)
