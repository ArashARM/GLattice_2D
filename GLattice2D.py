import numpy as np
import matplotlib.pyplot as plt

from ParTracing import Particle


class GLattice2D:
    """
    Generate a 2D G-Lattice inside one square sub-cell.

    Mirroring and tessellation are intentionally NOT included here.

    Generation rules
    ----------------
    1. First particle starts on the bottom boundary.
    2. First particle traces upward toward the top boundary.
    3. Later particles start on existing struts.
    4. Later particles trace upward or downward.
    5. Directions satisfy the passage-angle constraint.
    6. Untouched boundaries are deliberately targeted.
    7. Left/right boundaries may be touched and tracing then continues.
    8. Upward particles terminate at the top.
    9. Downward particles terminate at the bottom.
    10. A particle may terminate when it intersects an existing strut.
    11. Generation finishes only when:
            required weight is reached
            AND
            all four boundaries are touched.
    12. Failed directions/particles/generations are randomized again.
    """


    # ============================================================
    # INITIALISATION
    # ============================================================

    def __init__(self, domain, config):

        self.domain = domain
        self.config = config

        self.struts = []

        self.touched_boundaries = {
            "left": False,
            "right": False,
            "bottom": False,
            "top": False,
        }

        self.tol = 1e-8

        self.rng = np.random.default_rng(
            self.config.random_seed
        )


    # ============================================================
    # RESET GEOMETRY
    # ============================================================

    def reset_geometry(self):
        """
        Reset only the generated geometry.

        Important:
        The random-number generator is NOT reset here.

        Therefore, if one full generation attempt fails,
        the next generation attempt uses new random values.
        """

        self.struts = []

        self.touched_boundaries = {
            "left": False,
            "right": False,
            "bottom": False,
            "top": False,
        }


    # ============================================================
    # BOUNDARY INFORMATION
    # ============================================================

    def update_touched_boundaries(self, point):

        x, y = point
        M = self.domain.M

        if abs(x) <= self.tol:
            self.touched_boundaries["left"] = True

        if abs(x - M) <= self.tol:
            self.touched_boundaries["right"] = True

        if abs(y) <= self.tol:
            self.touched_boundaries["bottom"] = True

        if abs(y - M) <= self.tol:
            self.touched_boundaries["top"] = True


    def get_untouched_boundaries(self):

        return [
            name
            for name, touched
            in self.touched_boundaries.items()
            if not touched
        ]


    def all_boundaries_touched(self):

        return all(
            self.touched_boundaries.values()
        )


    # ============================================================
    # FIRST PARTICLE
    # ============================================================

    def create_first_particle(self):
        """
        First particle starts randomly on the bottom boundary.
        """

        x = self.rng.uniform(
            0.0,
            self.domain.M
        )

        position = np.array([
            x,
            0.0
        ])

        self.update_touched_boundaries(
            position
        )

        return Particle(
            position=position,
            trace_direction="up"
        )


    # ============================================================
    # PARTICLE ON EXISTING STRUT
    # ============================================================

    def create_particle_on_existing_strut(
        self,
        target_boundary=None
    ):
        """
        Randomly select an existing strut and a point along it.
        """

        if len(self.struts) == 0:

            raise RuntimeError(
                "Cannot create a particle because no struts exist."
            )


        # Random existing strut
        index = self.rng.integers(
            0,
            len(self.struts)
        )

        p0, p1 = self.struts[index]


        # Avoid selecting exactly at the end points
        t = self.rng.uniform(
            0.05,
            0.95
        )

        position = (
            p0
            +
            t * (p1 - p0)
        )


        # --------------------------------------------------------
        # Choose upward/downward tracing
        # --------------------------------------------------------

        if target_boundary == "top":

            trace_direction = "up"


        elif target_boundary == "bottom":

            trace_direction = "down"


        elif target_boundary in (
            "left",
            "right"
        ):

            # If very close to bottom,
            # upward tracing is more sensible.
            if position[1] <= self.tol:

                trace_direction = "up"


            # If very close to top,
            # downward tracing is more sensible.
            elif (
                position[1]
                >=
                self.domain.M - self.tol
            ):

                trace_direction = "down"


            else:

                trace_direction = (
                    "up"
                    if self.rng.random() < 0.5
                    else "down"
                )


        else:

            trace_direction = (
                "up"
                if self.rng.random() < 0.5
                else "down"
            )


        return Particle(
            position=position,
            trace_direction=trace_direction
        )


    # ============================================================
    # RANDOM POINT ON TARGET BOUNDARY
    # ============================================================

    def random_point_on_boundary(
        self,
        boundary,
        particle
    ):

        M = self.domain.M

        x, y = particle.position


        if boundary == "left":

            target_x = 0.0

            if particle.trace_direction == "up":

                low = min(
                    y + self.tol,
                    M
                )

                target_y = self.rng.uniform(
                    low,
                    M
                )

            else:

                high = max(
                    y - self.tol,
                    0.0
                )

                target_y = self.rng.uniform(
                    0.0,
                    high
                )


        elif boundary == "right":

            target_x = M

            if particle.trace_direction == "up":

                low = min(
                    y + self.tol,
                    M
                )

                target_y = self.rng.uniform(
                    low,
                    M
                )

            else:

                high = max(
                    y - self.tol,
                    0.0
                )

                target_y = self.rng.uniform(
                    0.0,
                    high
                )


        elif boundary == "top":

            target_x = self.rng.uniform(
                0.0,
                M
            )

            target_y = M


        elif boundary == "bottom":

            target_x = self.rng.uniform(
                0.0,
                M
            )

            target_y = 0.0


        else:

            raise ValueError(
                f"Unknown boundary: {boundary}"
            )


        return np.array([
            target_x,
            target_y
        ])


    # ============================================================
    # INITIAL DIRECTION
    # ============================================================

    def generate_initial_direction(
        self,
        particle,
        target_boundary=None
    ):
        """
        Generate the initial movement direction.

        When an untouched boundary is being targeted,
        aim directly toward a random point on that boundary.
        """

        if target_boundary is not None:

            target_point = (
                self.random_point_on_boundary(
                    target_boundary,
                    particle
                )
            )

            direction = (
                target_point
                -
                particle.position
            )

            length = np.linalg.norm(
                direction
            )

            if length > self.tol:

                return (
                    direction / length
                )


        # --------------------------------------------------------
        # No target boundary
        # --------------------------------------------------------

        if particle.trace_direction == "up":

            angle = self.rng.uniform(
                0.0,
                np.pi
            )


        elif particle.trace_direction == "down":

            angle = self.rng.uniform(
                np.pi,
                2.0 * np.pi
            )


        else:

            raise ValueError(
                "trace_direction must be 'up' or 'down'."
            )


        return np.array([
            np.cos(angle),
            np.sin(angle)
        ])


    # ============================================================
    # PASSAGE CONSTRAINT
    # ============================================================

    def generate_passage_direction(
        self,
        particle
    ):
        """
        Generate a random next direction satisfying
        the minimum passage-angle constraint.
        """

        if particle.direction is None:

            return None


        previous_angle = np.arctan2(
            particle.direction[1],
            particle.direction[0]
        )


        max_turn = np.radians(
            180.0
            -
            self.config.passage_angle
        )


        for _ in range(
            self.config.max_direction_attempts
        ):

            turn_angle = self.rng.uniform(
                -max_turn,
                max_turn
            )


            new_angle = (
                previous_angle
                +
                turn_angle
            )


            direction = np.array([
                np.cos(new_angle),
                np.sin(new_angle)
            ])


            # Upward trace must stay upward
            if (
                particle.trace_direction == "up"
                and
                direction[1] > self.tol
            ):

                return direction


            # Downward trace must stay downward
            if (
                particle.trace_direction == "down"
                and
                direction[1] < -self.tol
            ):

                return direction


        return None


    # ============================================================
    # ADD STRUT
    # ============================================================

    def add_strut(
        self,
        start_point,
        end_point
    ):

        p0 = np.asarray(
            start_point,
            dtype=float
        )

        p1 = np.asarray(
            end_point,
            dtype=float
        )


        if (
            np.linalg.norm(
                p1 - p0
            )
            <=
            self.tol
        ):

            return


        self.struts.append(
            (
                p0.copy(),
                p1.copy()
            )
        )


        self.update_touched_boundaries(
            p0
        )

        self.update_touched_boundaries(
            p1
        )


    # ============================================================
    # SEGMENT INTERSECTION
    # ============================================================

    def segment_intersection(
        self,
        p1,
        p2,
        p3,
        p4
    ):

        p1 = np.asarray(
            p1,
            dtype=float
        )

        p2 = np.asarray(
            p2,
            dtype=float
        )

        p3 = np.asarray(
            p3,
            dtype=float
        )

        p4 = np.asarray(
            p4,
            dtype=float
        )


        r = p2 - p1
        s = p4 - p3


        cross_rs = (
            r[0] * s[1]
            -
            r[1] * s[0]
        )


        if abs(cross_rs) <= self.tol:

            return None


        q_minus_p = (
            p3 - p1
        )


        t = (
            q_minus_p[0] * s[1]
            -
            q_minus_p[1] * s[0]
        ) / cross_rs


        u = (
            q_minus_p[0] * r[1]
            -
            q_minus_p[1] * r[0]
        ) / cross_rs


        # Ignore the particle's own start point.
        if (
            self.tol < t <= 1.0
            and
            self.tol < u < 1.0 - self.tol
        ):

            return (
                p1
                +
                t * r
            )


        return None


    # ============================================================
    # NEAREST INTERSECTION WITH EXISTING STRUT
    # ============================================================

    def find_existing_strut_intersection(
        self,
        start_point,
        end_point
    ):

        nearest_point = None
        nearest_distance = np.inf


        for old_start, old_end in self.struts:

            point = (
                self.segment_intersection(
                    start_point,
                    end_point,
                    old_start,
                    old_end
                )
            )


            if point is None:

                continue


            distance = np.linalg.norm(
                point - start_point
            )


            if distance < nearest_distance:

                nearest_distance = distance
                nearest_point = point


        return nearest_point


    # ============================================================
    # CHOOSE ONE VALID MOVEMENT
    # ============================================================

    def choose_valid_movement(
        self,
        particle,
        target_boundary=None,
        first_step=False
    ):
        """
        Try many random directions.

        If none works, return None.

        The calling function then randomizes the particle again.
        """

        for _ in range(
            self.config.max_direction_attempts
        ):

            # ----------------------------------------------------
            # Candidate direction
            # ----------------------------------------------------

            if first_step:

                direction = (
                    self.generate_initial_direction(
                        particle,
                        target_boundary
                    )
                )

            else:

                direction = (
                    self.generate_passage_direction(
                        particle
                    )
                )


            if direction is None:

                continue


            proposed_end = (
                particle.position
                +
                self.config.step_size
                * direction
            )


            # ----------------------------------------------------
            # Entire movement stays inside
            # ----------------------------------------------------

            if self.domain.is_inside(
                proposed_end
            ):

                return (
                    direction,
                    proposed_end,
                    None,
                    None
                )


            # ----------------------------------------------------
            # Movement reaches boundary
            # ----------------------------------------------------

            boundary_point, boundary_name = (
                self.domain.trim_to_boundary(
                    particle.position,
                    proposed_end
                )
            )


            distance_to_boundary = (
                np.linalg.norm(
                    boundary_point
                    -
                    particle.position
                )
            )


            # Particle is already on side boundary
            # and candidate direction immediately leaves domain.
            if (
                boundary_name in (
                    "left",
                    "right"
                )
                and
                distance_to_boundary
                <=
                self.tol
            ):

                continue


            return (
                direction,
                proposed_end,
                boundary_point,
                boundary_name
            )


        return None


    # ============================================================
    # TRACE ONE PARTICLE
    # ============================================================

    def trace_particle(
        self,
        particle,
        target_boundary=None
    ):
        """
        Trace one particle.

        Returns
        -------
        "top"
        "bottom"
        "intersection"
        "no_valid_direction"
        "maximum_steps"
        """

        first_step = True


        for _ in range(
            self.config.max_steps_per_particle
        ):

            movement = (
                self.choose_valid_movement(
                    particle,
                    target_boundary=target_boundary,
                    first_step=first_step
                )
            )


            # ----------------------------------------------------
            # Could not find any valid direction
            # ----------------------------------------------------

            if movement is None:

                return "no_valid_direction"


            (
                new_direction,
                proposed_end,
                boundary_point,
                boundary_name
            ) = movement


            start_point = (
                particle.position.copy()
            )


            # ----------------------------------------------------
            # Existing-strut intersection
            # ----------------------------------------------------

            intersection_point = (
                self.find_existing_strut_intersection(
                    start_point,
                    proposed_end
                )
            )


            events = []


            if intersection_point is not None:

                distance = np.linalg.norm(
                    intersection_point
                    -
                    start_point
                )

                events.append(
                    (
                        distance,
                        "intersection",
                        intersection_point,
                        None
                    )
                )


            if boundary_point is not None:

                distance = np.linalg.norm(
                    boundary_point
                    -
                    start_point
                )

                events.append(
                    (
                        distance,
                        "boundary",
                        boundary_point,
                        boundary_name
                    )
                )


            # ----------------------------------------------------
            # First event occurs before complete movement
            # ----------------------------------------------------

            if len(events) > 0:

                events.sort(
                    key=lambda event: event[0]
                )


                (
                    _,
                    event_type,
                    event_point,
                    event_boundary
                ) = events[0]


                self.add_strut(
                    start_point,
                    event_point
                )


                particle.set_direction(
                    new_direction
                )


                particle.update_position(
                    event_point
                )


                # Existing strut reached
                if event_type == "intersection":

                    return "intersection"


                # Upward trace reaches top
                if (
                    event_boundary == "top"
                    and
                    particle.trace_direction == "up"
                ):

                    return "top"


                # Downward trace reaches bottom
                if (
                    event_boundary == "bottom"
                    and
                    particle.trace_direction == "down"
                ):

                    return "bottom"


                # Left/right boundary:
                # mark as touched and keep tracing.
                if event_boundary in (
                    "left",
                    "right"
                ):

                    target_boundary = None
                    first_step = False

                    continue


            # ----------------------------------------------------
            # Normal movement
            # ----------------------------------------------------

            else:

                self.add_strut(
                    start_point,
                    proposed_end
                )


                particle.set_direction(
                    new_direction
                )


                particle.update_position(
                    proposed_end
                )


            first_step = False


        return "maximum_steps"


    # ============================================================
    # WEIGHT
    # ============================================================

    def calculate_weight(self):
        """
        Approximate 2D material fraction.

        area ~= total strut length * strut width
        """

        total_length = 0.0


        for p0, p1 in self.struts:

            total_length += np.linalg.norm(
                p1 - p0
            )


        approximate_area = (
            total_length
            *
            self.config.strut_width
        )


        domain_area = (
            self.domain.M ** 2
        )


        return (
            approximate_area
            /
            domain_area
        )


    # ============================================================
    # GENERATION CONDITION
    # ============================================================

    def generation_complete(self):

        weight_ok = (
            self.calculate_weight()
            >=
            self.config.required_weight
        )


        boundaries_ok = (
            self.all_boundaries_touched()
        )


        return (
            weight_ok
            and
            boundaries_ok
        )


    # ============================================================
    # TRY FIRST PARTICLE
    # ============================================================

    def generate_first_trace(self):
        """
        Randomize the first particle repeatedly until
        a complete valid first trace reaches the top.
        """

        for _ in range(
            self.config.max_particle_attempts
        ):

            # First trace is generated from scratch
            self.struts = []

            self.touched_boundaries = {
                "left": False,
                "right": False,
                "bottom": False,
                "top": False,
            }


            particle = (
                self.create_first_particle()
            )


            result = (
                self.trace_particle(
                    particle,
                    target_boundary="top"
                )
            )


            # Successful first trace
            if result == "top":

                return True


            # Otherwise randomize first particle again


        return False


    # ============================================================
    # TRY TO ADD ONE NEW PARTICLE
    # ============================================================

    def add_new_particle(
        self,
        target_boundary=None
    ):
        """
        Try multiple randomized particles.

        If a particle fails:
            restore previous geometry
            randomize a new particle
            try again.

        If an untouched target boundary was selected,
        a trace is accepted only if that boundary becomes touched.
        """

        for _ in range(
            self.config.max_particle_attempts
        ):

            # ----------------------------------------------------
            # Save current lattice
            # ----------------------------------------------------

            old_strut_count = len(
                self.struts
            )

            old_boundaries = (
                self.touched_boundaries.copy()
            )


            target_was_touched = (
                True
                if target_boundary is None
                else self.touched_boundaries[
                    target_boundary
                ]
            )


            # ----------------------------------------------------
            # Random new particle
            # ----------------------------------------------------

            particle = (
                self.create_particle_on_existing_strut(
                    target_boundary
                )
            )


            # ----------------------------------------------------
            # Trace it
            # ----------------------------------------------------

            result = (
                self.trace_particle(
                    particle,
                    target_boundary=target_boundary
                )
            )


            valid_result = (
                result
                in (
                    "top",
                    "bottom",
                    "intersection"
                )
            )


            # ----------------------------------------------------
            # Was targeted boundary successfully reached?
            # ----------------------------------------------------

            if target_boundary is None:

                target_satisfied = True


            elif target_was_touched:

                target_satisfied = True


            else:

                target_satisfied = (
                    self.touched_boundaries[
                        target_boundary
                    ]
                )


            # ----------------------------------------------------
            # Accept successful particle
            # ----------------------------------------------------

            if (
                valid_result
                and
                target_satisfied
            ):

                return True


            # ----------------------------------------------------
            # Failed:
            # rollback this particle completely
            # ----------------------------------------------------

            self.struts = (
                self.struts[
                    :old_strut_count
                ]
            )

            self.touched_boundaries = (
                old_boundaries
            )


        return False


    # ============================================================
    # ONE COMPLETE GENERATION ATTEMPT
    # ============================================================

    def generate_once(self):
        """
        Attempt to generate one valid sub-cell.

        Returns True on success.
        Returns False if this complete attempt becomes stuck.
        """

        self.reset_geometry()


        # --------------------------------------------------------
        # First trace
        # --------------------------------------------------------

        if not self.generate_first_trace():

            return False


        particle_number = 1


        # --------------------------------------------------------
        # Add later particles
        # --------------------------------------------------------

        while not self.generation_complete():

            if (
                particle_number
                >=
                self.config.max_particles
            ):

                return False


            # ----------------------------------------------------
            # Select an untouched boundary first
            # ----------------------------------------------------

            untouched = (
                self.get_untouched_boundaries()
            )


            if len(untouched) > 0:

                target_boundary = (
                    self.rng.choice(
                        untouched
                    )
                )

            else:

                target_boundary = None


            # ----------------------------------------------------
            # Try new randomized particles
            # ----------------------------------------------------

            success = (
                self.add_new_particle(
                    target_boundary
                )
            )


            if not success:

                # This full generation attempt got stuck.
                return False


            particle_number += 1


        return True


    # ============================================================
    # GENERATE
    # ============================================================

    def generate(self):
        """
        Main generative process.

        If one full generation cannot find a valid solution,
        discard it and randomize everything again.

        Only fails after max_generation_attempts complete attempts.
        """

        for generation_attempt in range(
            1,
            self.config.max_generation_attempts + 1
        ):

            print(
                f"Generation attempt "
                f"{generation_attempt}/"
                f"{self.config.max_generation_attempts}"
            )


            success = (
                self.generate_once()
            )


            if success:

                print()
                print(
                    "Valid G-Lattice generated."
                )

                print(
                    "Struts:",
                    len(self.struts)
                )

                print(
                    "Weight:",
                    round(
                        self.calculate_weight(),
                        4
                    )
                )

                print(
                    "Touched boundaries:",
                    self.touched_boundaries
                )

                return


            print(
                "Attempt failed - randomizing again..."
            )


        raise RuntimeError(
            "A valid G-Lattice could not be generated "
            "after all generation attempts. "
            "The selected parameter combination may be "
            "too restrictive."
        )


    # ============================================================
    # PLOT
    # ============================================================

    def plot(self):

        fig, ax = plt.subplots(
            figsize=(7, 7)
        )


        self.domain.plot(
            ax
        )


        for p0, p1 in self.struts:

            ax.plot(
                [p0[0], p1[0]],
                [p0[1], p1[1]],
                linewidth=2
            )


        M = self.domain.M


        ax.set_xlim(
            -0.05 * M,
            1.05 * M
        )

        ax.set_ylim(
            -0.05 * M,
            1.05 * M
        )


        ax.set_aspect(
            "equal"
        )


        ax.set_xlabel("x")
        ax.set_ylabel("y")


        ax.set_title(
            "2D G-Lattice - Generated Sub-cell"
        )


        status = (
            f"Weight = "
            f"{self.calculate_weight():.3f}\n"
            f"L={self.touched_boundaries['left']}  "
            f"R={self.touched_boundaries['right']}  "
            f"B={self.touched_boundaries['bottom']}  "
            f"T={self.touched_boundaries['top']}"
        )


        ax.text(
            0.02,
            0.98,
            status,
            transform=ax.transAxes,
            va="top"
        )


        plt.show()