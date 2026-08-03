"""Gen6 matched component-screen architectures.

Gen6 is intentionally an experiment suite, not a replacement training
framework.  It reuses the Gen3 data, masking, trainer, checkpoint and
evaluator contracts while varying one named component at a time.
"""

from .contract import GEN6_ARM_SPECS, Gen6ArmSpec, get_gen6_arm_spec

__all__ = ["GEN6_ARM_SPECS", "Gen6ArmSpec", "get_gen6_arm_spec"]
