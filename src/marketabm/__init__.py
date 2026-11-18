"""marketabm: agent-based market model for studying a seller-side transaction tax.

Results describe the model's mechanism, not a forecast for the Philippine Stock Exchange.
"""

from marketabm.config import ModelConfig, load_model_config, with_overrides
from marketabm.market import run
from marketabm.results import SimulationResult

__all__ = ["ModelConfig", "SimulationResult", "load_model_config", "run", "with_overrides"]
