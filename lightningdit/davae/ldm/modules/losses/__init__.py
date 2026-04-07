"""
Loss exports for the local `davae/ldm` package.

Avoid importing from external package names (e.g. `LightningDiT.*`) because this
repo is meant to run standalone. Also avoid relying on a top-level `ldm` module
which may collide with unrelated installed packages in some environments.
"""

from .contperceptual import LPIPSWithDiscriminator