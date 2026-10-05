# GLattice2D

Beginner-friendly 2D re-implementation of the G-Lattice particle-tracing concept.

## Configuration

The current design parameters are stored in `config.py`:

- `M`: square design space `[0, M] x [0, M]`
- `step_size`: movement distance at every particle step
- `passage_angle`: passage-triangle angle in degrees
- `required_weight`: target relative material amount
- `strut_width`: width of each 2D strut

Computational controls:

- `random_seed`
- `max_iterations`

## Current implementation

The current `main.py` implements the first particle movement:

1. Create a particle `P0` on the bottom boundary.
2. Generate a random direction pointing into the square.
3. Move the particle by one `step_size`.
4. Obtain `P1 = P0 + step_size * direction`.
5. Store/plot the segment `P0 -> P1` as the first strut.

For this first learning step, the starting x-coordinate is kept slightly
away from the left and right boundaries so that the first movement stays
inside the square. This temporary restriction will be removed when
periodic boundary handling is implemented.

Run:

```bash
python main.py
```

## Next step

Construct the 2D passage triangle from the current particle position and
movement direction, then choose the next movement direction from inside it.
# GLattice_2D
