# `utils`

This directory contains utility scripts that are used by the other scripts in this repository.

- `__init__.py`: contains basic logging utilities used by the trainer.
- `eval_utils.py`: contains utility functions for evaluating gradient and curvature accuracy.
- `fd_utils.py`: contains utility functions for computing finite difference approximations of gradients and curvatures.
- `mesh_metrics.py`: utilities to measure quality of meshes extracted from the neural SDF. Adapted from [ldif](https://github.com/google/ldif/blob/master/eval.py).
- `polyfit_utils.py`: contains implementations of our polynomial-fitting gradient and curvature operators.
- `ray_tracer.py`: contains a ray-tracer for rendering meshes, based on [Mitsuba3](https://mitsuba.readthedocs.io/en/latest/src/quickstart/mitsuba_quickstart.html).
- `rendering.py`: contains utility functions for our rendering demo.
- `sphere_tracer.py`: a naive sphere-tracer for rendering neural SDFs.
- `utils.py`: common utilities like marching cubes, autodiff differential operators, etc.
- `viz.py`: visualization utilities to help with debugging, logging, etc.