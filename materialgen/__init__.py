"""MaterialGen: concrete mix design & strength prediction.

Provides two pipelines:
* **Inverse** (existing): properties → composition via NEAT + BNN
* **Forward** (new): composition + time → strength via GAN + NEAT+BNN

Each stage exposes a single ``run_*`` function; see ``materialgen/cli.py``
for the CLI wiring.
"""

# --- Existing inverse pipeline ---
from .make_neat_to_bnn import run_make_neat_to_bnn
from .train_neat import run_train_neat

# --- New forward pipeline ---
from .generator import ConcreteGenerator, GeneratorConfig
from .discriminator import NeatBNNDiscriminator, DiscriminatorConfig
from .gan_trainer import ConcreteGAN, GANConfig
from .uncertainty import UncertaintyEstimator
from .transfer import TransferLearner, TransferConfig
from .data_preparation import load_and_unify_datasets, UnifiedDataset
from .physics import load_gost_table, GostTable
from .metrics import evaluate_model, FullEvaluation

__all__ = [
    # Inverse pipeline
    "run_train_neat",
    "run_make_neat_to_bnn",
    # Forward pipeline — models
    "ConcreteGenerator",
    "GeneratorConfig",
    "NeatBNNDiscriminator",
    "DiscriminatorConfig",
    "ConcreteGAN",
    "GANConfig",
    "UncertaintyEstimator",
    "TransferLearner",
    "TransferConfig",
    # Forward pipeline — data & metrics
    "load_and_unify_datasets",
    "UnifiedDataset",
    "load_gost_table",
    "GostTable",
    "evaluate_model",
    "FullEvaluation",
]
