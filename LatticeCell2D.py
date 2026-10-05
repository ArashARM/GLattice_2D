import numpy as np
import matplotlib.pyplot as plt


class LatticeCell2D:
    """
    Construct a complete 2D lattice unit cell by mirroring
    a generated G-Lattice sub-cell.

    Original generated sub-cell:

        D = [0, M] x [0, M]

    Complete unit cell:

        C = [0, 2M] x [0, 2M]

    The original sub-cell is mirrored:

        1. across x = M
        2. across y = M
        3. across both x = M and y = M

    This is the 2D analogue of the mirroring procedure used
    in the original 3D G-Lattice formulation.
    """


    # ============================================================
    # INITIALISATION
    # ============================================================

    def __init__(
        self,
        glattice
    ):

        self.glattice = glattice

        self.M = (
            glattice.domain.M
        )

        # Original generated struts
        self.subcell_struts = (
            glattice.struts
        )

        # Complete mirrored unit-cell struts
        self.cell_struts = []

        self.tol = 1e-8


    # ============================================================
    # MIRROR ONE POINT
    # ============================================================

    def mirror_point(
        self,
        point,
        mirror_x=False,
        mirror_y=False
    ):
        """
        Mirror one point across the internal symmetry lines.

        Parameters
        ----------
        point : array-like
            [x, y]

        mirror_x : bool
            Mirror across x = M.

        mirror_y : bool
            Mirror across y = M.

        Returns
        -------
        numpy.ndarray
            Mirrored point.
        """

        x, y = point


        # --------------------------------------------------------
        # Mirror across x = M
        # --------------------------------------------------------

        if mirror_x:

            x = (
                2.0 * self.M
                -
                x
            )


        # --------------------------------------------------------
        # Mirror across y = M
        # --------------------------------------------------------

        if mirror_y:

            y = (
                2.0 * self.M
                -
                y
            )


        return np.array(
            [x, y],
            dtype=float
        )


    # ============================================================
    # MIRROR ONE STRUT
    # ============================================================

    def mirror_strut(
        self,
        strut,
        mirror_x=False,
        mirror_y=False
    ):
        """
        Mirror both endpoints of a strut.
        """

        p0, p1 = strut


        new_p0 = self.mirror_point(
            p0,
            mirror_x=mirror_x,
            mirror_y=mirror_y
        )


        new_p1 = self.mirror_point(
            p1,
            mirror_x=mirror_x,
            mirror_y=mirror_y
        )


        return (
            new_p0,
            new_p1
        )


    # ============================================================
    # CHECK WHETHER TWO STRUTS ARE IDENTICAL
    # ============================================================

    def same_strut(
        self,
        strut_a,
        strut_b
    ):
        """
        Check whether two struts represent the same segment.

        Endpoint order does not matter.
        """

        a0, a1 = strut_a
        b0, b1 = strut_b


        same_direction = (
            np.linalg.norm(a0 - b0) < self.tol
            and
            np.linalg.norm(a1 - b1) < self.tol
        )


        opposite_direction = (
            np.linalg.norm(a0 - b1) < self.tol
            and
            np.linalg.norm(a1 - b0) < self.tol
        )


        return (
            same_direction
            or
            opposite_direction
        )


    # ============================================================
    # ADD STRUT WITHOUT DUPLICATES
    # ============================================================

    def add_unique_strut(
        self,
        strut
    ):
        """
        Add a strut only if the same segment does not
        already exist in the complete cell.
        """

        for existing_strut in self.cell_struts:

            if self.same_strut(
                strut,
                existing_strut
            ):

                return


        self.cell_struts.append(
            strut
        )


    # ============================================================
    # BUILD COMPLETE UNIT CELL
    # ============================================================

    def build(self):
        """
        Generate the four symmetric sub-cells.

        Sub-cell 1:
            original geometry

        Sub-cell 2:
            mirrored across x = M

        Sub-cell 3:
            mirrored across y = M

        Sub-cell 4:
            mirrored across both x = M and y = M
        """

        self.cell_struts = []


        # ========================================================
        # COPY 1
        # ORIGINAL SUB-CELL
        # ========================================================

        for strut in self.subcell_struts:

            mirrored_strut = (
                self.mirror_strut(
                    strut,
                    mirror_x=False,
                    mirror_y=False
                )
            )

            self.add_unique_strut(
                mirrored_strut
            )


        # ========================================================
        # COPY 2
        # MIRROR ACROSS x = M
        # ========================================================

        for strut in self.subcell_struts:

            mirrored_strut = (
                self.mirror_strut(
                    strut,
                    mirror_x=True,
                    mirror_y=False
                )
            )

            self.add_unique_strut(
                mirrored_strut
            )


        # ========================================================
        # COPY 3
        # MIRROR ACROSS y = M
        # ========================================================

        for strut in self.subcell_struts:

            mirrored_strut = (
                self.mirror_strut(
                    strut,
                    mirror_x=False,
                    mirror_y=True
                )
            )

            self.add_unique_strut(
                mirrored_strut
            )


        # ========================================================
        # COPY 4
        # MIRROR ACROSS BOTH x = M AND y = M
        # ========================================================

        for strut in self.subcell_struts:

            mirrored_strut = (
                self.mirror_strut(
                    strut,
                    mirror_x=True,
                    mirror_y=True
                )
            )

            self.add_unique_strut(
                mirrored_strut
            )


        return self.cell_struts


    # ============================================================
    # PLOT COMPLETE UNIT CELL
    # ============================================================

    def draw_strut(self, ax, p0, p1):

        p0 = np.asarray(p0, dtype=float)
        p1 = np.asarray(p1, dtype=float)

        direction = p1 - p0

        length = np.linalg.norm(direction)

        if length < self.tol:
            return

        # Unit direction
        direction = direction / length

        # Unit vector perpendicular to the strut
        normal = np.array([
            -direction[1],
            direction[0]
        ])

        # Half of the physical strut width
        half_width = (
            self.glattice.config.strut_width
            / 2.0
        )

        offset = (
            half_width * normal
        )

        # Four corners of the 2D strut
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
    def plot(self, plot_line=False):
        """
        Plot the complete mirrored 2D lattice unit cell.

        Dashed red square:
            original generated sub-cell

        Dashed green square:
            complete mirrored unit cell
        """

        if len(self.cell_struts) == 0:

            raise RuntimeError(
                "Unit cell has not been built yet. "
                "Call build() first."
            )


        fig, ax = plt.subplots(
            figsize=(8, 8)
        )


        # ========================================================
        # PLOT STRUTS
        # ========================================================

        for p0, p1 in self.cell_struts:
            if plot_line:
                ax.plot(
                    [p0[0], p1[0]],
                    [p0[1], p1[1]],
                    linewidth=2
                )
            else:
                self.draw_strut(ax, p0, p1)

        M = self.M

        cell_size = (
            2.0 * M
        )


        # ========================================================
        # ORIGINAL SUB-CELL
        # Dashed red square
        # ========================================================

        ax.plot(
            [
                0,
                M,
                M,
                0,
                0
            ],
            [
                0,
                0,
                M,
                M,
                0
            ],
            linestyle="--",
            color="red",
            linewidth=2,
            label="Original sub-cell"
        )


        # ========================================================
        # COMPLETE UNIT CELL
        # Dashed green square
        # ========================================================

        ax.plot(
            [
                0,
                cell_size,
                cell_size,
                0,
                0
            ],
            [
                0,
                0,
                cell_size,
                cell_size,
                0
            ],
            linestyle="--",
            color="green",
            linewidth=2,
            label="Complete unit cell"
        )


        # ========================================================
        # INTERNAL MIRROR LINES
        # ========================================================

        ax.axvline(
            x=M,
            linestyle=":",
            linewidth=1
        )

        ax.axhline(
            y=M,
            linestyle=":",
            linewidth=1
        )


        # ========================================================
        # PLOT SETTINGS
        # ========================================================

        ax.set_xlim(
            -0.05 * cell_size,
            1.05 * cell_size
        )

        ax.set_ylim(
            -0.05 * cell_size,
            1.05 * cell_size
        )


        ax.set_aspect(
            "equal"
        )


        ax.set_xlabel("x")
        ax.set_ylabel("y")


        ax.set_title(
            "Complete 2D G-Lattice Unit Cell"
        )


        #ax.legend()


        plt.show()