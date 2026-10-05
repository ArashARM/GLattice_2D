import numpy as np


class Particle:

    def __init__(
        self,
        position,
        trace_direction
    ):
        """
        Parameters
        ----------
        position
            Current [x,y] position.

        trace_direction
            "up" or "down".
        """

        self.position = np.array(
            position,
            dtype=float
        )

        self.direction = None

        self.trace_direction = trace_direction


    # ============================================================
    # SET MOVEMENT DIRECTION
    # ============================================================

    def set_direction(
        self,
        direction
    ):

        direction = np.array(
            direction,
            dtype=float
        )

        length = np.linalg.norm(
            direction
        )

        if length == 0:

            raise ValueError(
                "Direction cannot be zero."
            )

        self.direction = (
            direction / length
        )


    # ============================================================
    # CALCULATE NEXT POINT
    # ============================================================

    def next_position(
        self,
        step_size
    ):

        if self.direction is None:

            raise ValueError(
                "Particle direction is undefined."
            )

        return (
            self.position
            +
            step_size
            * self.direction
        )


    # ============================================================
    # UPDATE POSITION
    # ============================================================

    def update_position(
        self,
        new_position
    ):

        self.position = np.array(
            new_position,
            dtype=float
        )