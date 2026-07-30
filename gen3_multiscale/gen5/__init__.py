"""Gen5 full-expression ablation suite -- additive to gen3_multiscale.

See GEN5_CONTRACT.md (repo root of gen3_multiscale/) for the full design.
Depends on (imports from, never modifies) gen3_multiscale/gen4/ -- the
spatial conditioner, encoder-provider interfaces, and cache builders are
reused unmodified. Nothing in this package modifies any existing
gen3_multiscale or gen4 module in place.
"""
from __future__ import annotations
