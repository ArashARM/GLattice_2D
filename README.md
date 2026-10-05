# GLattice2D

This repository contains a beginner-friendly 2D implementation of the particle-tracing idea used in the G-Lattice method.

The purpose of this code is not only to generate lattice structures, but also to help you understand:

- how a particle is represented in 2D,
- how movement directions are defined,
- how line segments are created,
- how geometric constraints are applied,
- how a lattice grows from multiple particle traces,
- how boundary conditions affect connectivity,
- how a sub-cell is mirrored into a complete lattice cell,
- and how the cell can be repeated to form a larger lattice structure.

You are not expected to understand the whole code immediately.

The best way to work with this project is to study it in small steps.

---

# 1. Recommended order to study the code

Do not start from `GLattice2D.py`.

Start with the simplest files first.

Recommended order:

```text
1. Config.py
2. Domain.py
3. ParTracing.py
4. Main.ipynb
5. GLattice2D.py
6. LatticeCell2D.py
7. LatticeStructure2D.py
