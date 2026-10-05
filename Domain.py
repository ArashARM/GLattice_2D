import numpy as np


class Domain:

    def __init__(self, M):

        self.M = M


    # ============================================================
    # POINT INSIDE DOMAIN?
    # ============================================================

    def is_inside(self, point):

        x, y = point

        return (
            0.0 <= x <= self.M
            and
            0.0 <= y <= self.M
        )


    # ============================================================
    # BOUNDARY CROSSING
    # ============================================================

    def trim_to_boundary(
        self,
        start_point,
        end_point
    ):
        """
        If a movement goes outside the square,
        trim the segment exactly at the first boundary.

        Returns
        -------
        trimmed_point
        boundary_name

        boundary_name can be:
            "left"
            "right"
            "bottom"
            "top"
            None
        """

        p0 = np.array(
            start_point,
            dtype=float
        )

        p1 = np.array(
            end_point,
            dtype=float
        )

        # If P1 is already inside,
        # nothing needs trimming.
        if self.is_inside(p1):

            return p1, None


        direction = p1 - p0

        possible_hits = []


        # --------------------------------------------------------
        # LEFT boundary: x = 0
        # --------------------------------------------------------

        if direction[0] < 0:

            t = (
                0.0 - p0[0]
            ) / direction[0]

            y = (
                p0[1]
                + t * direction[1]
            )

            if (
                0.0 <= t <= 1.0
                and
                0.0 <= y <= self.M
            ):

                possible_hits.append(
                    (
                        t,
                        np.array([0.0, y]),
                        "left"
                    )
                )


        # --------------------------------------------------------
        # RIGHT boundary: x = M
        # --------------------------------------------------------

        if direction[0] > 0:

            t = (
                self.M - p0[0]
            ) / direction[0]

            y = (
                p0[1]
                + t * direction[1]
            )

            if (
                0.0 <= t <= 1.0
                and
                0.0 <= y <= self.M
            ):

                possible_hits.append(
                    (
                        t,
                        np.array([self.M, y]),
                        "right"
                    )
                )


        # --------------------------------------------------------
        # BOTTOM boundary: y = 0
        # --------------------------------------------------------

        if direction[1] < 0:

            t = (
                0.0 - p0[1]
            ) / direction[1]

            x = (
                p0[0]
                + t * direction[0]
            )

            if (
                0.0 <= t <= 1.0
                and
                0.0 <= x <= self.M
            ):

                possible_hits.append(
                    (
                        t,
                        np.array([x, 0.0]),
                        "bottom"
                    )
                )


        # --------------------------------------------------------
        # TOP boundary: y = M
        # --------------------------------------------------------

        if direction[1] > 0:

            t = (
                self.M - p0[1]
            ) / direction[1]

            x = (
                p0[0]
                + t * direction[0]
            )

            if (
                0.0 <= t <= 1.0
                and
                0.0 <= x <= self.M
            ):

                possible_hits.append(
                    (
                        t,
                        np.array([x, self.M]),
                        "top"
                    )
                )


        if len(possible_hits) == 0:

            raise RuntimeError(
                "No boundary intersection found."
            )


        # First boundary encountered
        possible_hits.sort(
            key=lambda hit: hit[0]
        )

        _, point, boundary = possible_hits[0]

        return point, boundary


    # ============================================================
    # PLOT
    # ============================================================

    def plot(self, ax):

        M = self.M

        ax.plot(
            [0, M, M, 0, 0],
            [0, 0, M, M, 0],
            linewidth=2
        )