from dataclasses import asdict, dataclass


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass
class Config:

    # ========================================================
    # DESIGN PARAMETERS
    # ========================================================

    M: float = 1.0
    step_size: float = 0.08
    passage_angle: float = 95.0
    required_weight: float = 0.15
    strut_width: float = 0.02

    # ========================================================
    # COMPUTATIONAL CONTROLS
    # ========================================================

    random_seed: int | None = None

    max_steps_per_particle: int = 1000
    max_particles: int = 1000

    # Number of candidate directions tested at one particle step
    max_direction_attempts: int = 200

    # Number of times a complete particle may be regenerated
    max_particle_attempts: int = 50
    max_generation_attempts: int = 20