import numpy as np
import matplotlib.pyplot as plt


class LatticeStructure2D:
    """
    Tessellate a complete 2D lattice unit cell side-by-side.

    The input unit cell has size:

        2M x 2M

    The unit cell is repeated:

        n_x times in the x-direction
        n_y times in the y-direction

    No mirroring is performed here.

    This class performs translation only.
    """


    # ============================================================
    # INITIALISATION
    # ============================================================

    def __init__(
        self,
        lattice_cell,
        n_x=3,
        n_y=3
    ):

        self.lattice_cell = lattice_cell

        self.n_x = n_x
        self.n_y = n_y

        self.M = lattice_cell.M

        # Full unit-cell size
        self.cell_size = (
            2.0 * self.M
        )

        # Complete tessellated structure
        self.structure_struts = []

        self.tol = 1e-8


    # ============================================================
    # TRANSLATE ONE POINT
    # ============================================================

    def translate_point(
        self,
        point,
        offset_x,
        offset_y
    ):

        point = np.asarray(
            point,
            dtype=float
        )

        return (
            point
            +
            np.array([
                offset_x,
                offset_y
            ])
        )


    # ============================================================
    # TRANSLATE ONE STRUT
    # ============================================================

    def translate_strut(
        self,
        strut,
        offset_x,
        offset_y
    ):

        p0, p1 = strut


        new_p0 = self.translate_point(
            p0,
            offset_x,
            offset_y
        )


        new_p1 = self.translate_point(
            p1,
            offset_x,
            offset_y
        )


        return (
            new_p0,
            new_p1
        )


    # ============================================================
    # BUILD TESSELLATED STRUCTURE
    # ============================================================

    def build(self):
        """
        Repeat the complete unit cell side-by-side.

        Translation:

            x' = x + i * cell_size
            y' = y + j * cell_size

        where:

            i = 0, ..., n_x - 1
            j = 0, ..., n_y - 1
        """

        if len(
            self.lattice_cell.cell_struts
        ) == 0:

            raise RuntimeError(
                "The lattice cell has not been built. "
                "Call lattice_cell.build() first."
            )


        self.structure_struts = []


        # ========================================================
        # TESSELLATION IN X AND Y
        # ========================================================

        for j in range(
            self.n_y
        ):

            for i in range(
                self.n_x
            ):

                offset_x = (
                    i
                    *
                    self.cell_size
                )

                offset_y = (
                    j
                    *
                    self.cell_size
                )


                # -----------------------------------------------
                # Copy every strut in the unit cell
                # -----------------------------------------------

                for strut in (
                    self.lattice_cell.cell_struts
                ):

                    translated_strut = (
                        self.translate_strut(
                            strut,
                            offset_x,
                            offset_y
                        )
                    )


                    self.structure_struts.append(
                        translated_strut
                    )


        return self.structure_struts


    # ============================================================
    # DRAW STRUT WITH PHYSICAL WIDTH
    # ============================================================

    def draw_strut(
        self,
        ax,
        p0,
        p1
    ):
        """
        Draw a strut using the actual geometric strut width.
        """

        p0 = np.asarray(
            p0,
            dtype=float
        )

        p1 = np.asarray(
            p1,
            dtype=float
        )


        direction = (
            p1 - p0
        )


        length = np.linalg.norm(
            direction
        )


        if length <= self.tol:

            return


        direction = (
            direction / length
        )


        # Perpendicular unit vector
        normal = np.array([
            -direction[1],
            direction[0]
        ])


        half_width = (
            self.lattice_cell.glattice.config.strut_width
            /
            2.0
        )


        offset = (
            half_width
            *
            normal
        )


        corners = np.array([
            p0 + offset,
            p1 + offset,
            p1 - offset,
            p0 - offset
        ])


        polygon = plt.Polygon(
            corners,
            closed=True
        )


        ax.add_patch(
            polygon
        )


    # ============================================================
    # PLOT TESSELLATED STRUCTURE
    # ============================================================

    def plot(
        self,
        show_cell_boundaries=True
    ):

        if len(
            self.structure_struts
        ) == 0:

            raise RuntimeError(
                "The structure has not been built. "
                "Call build() first."
            )


        fig, ax = plt.subplots(
            figsize=(10, 10)
        )


        # ========================================================
        # STRUTS
        # ========================================================

        for p0, p1 in (
            self.structure_struts
        ):

            self.draw_strut(
                ax,
                p0,
                p1
            )


        # ========================================================
        # UNIT-CELL BOUNDARIES
        # ========================================================

        if show_cell_boundaries:

            for j in range(
                self.n_y
            ):

                for i in range(
                    self.n_x
                ):

                    x0 = (
                        i
                        *
                        self.cell_size
                    )

                    y0 = (
                        j
                        *
                        self.cell_size
                    )


                    x1 = (
                        x0
                        +
                        self.cell_size
                    )

                    y1 = (
                        y0
                        +
                        self.cell_size
                    )


                    ax.plot(
                        [
                            x0,
                            x1,
                            x1,
                            x0,
                            x0
                        ],
                        [
                            y0,
                            y0,
                            y1,
                            y1,
                            y0
                        ],
                        linestyle="--",
                        color="green",
                        linewidth=1
                    )


        # ========================================================
        # WHOLE STRUCTURE BOUNDARY
        # ========================================================

        total_width = (
            self.n_x
            *
            self.cell_size
        )

        total_height = (
            self.n_y
            *
            self.cell_size
        )


        ax.plot(
            [
                0,
                total_width,
                total_width,
                0,
                0
            ],
            [
                0,
                0,
                total_height,
                total_height,
                0
            ],
            linestyle="--",
            color="red",
            linewidth=2
        )


        # ========================================================
        # PLOT SETTINGS
        # ========================================================

        ax.set_xlim(
            -0.02 * total_width,
            1.02 * total_width
        )

        ax.set_ylim(
            -0.02 * total_height,
            1.02 * total_height
        )


        ax.set_aspect(
            "equal"
        )


        ax.set_xlabel("x")
        ax.set_ylabel("y")


        ax.set_title(
            f"2D G-Lattice Structure "
            f"({self.n_x} × {self.n_y} Cells)"
        )


        plt.show()