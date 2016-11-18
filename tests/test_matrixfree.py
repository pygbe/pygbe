import pytest
from unittest.mock import patch

from pygbe.matrixfree import calc_s_start

Surface = patch('pygbe.classes.Surface')
s_starts = [
    (Surface, 'nope', 1, 20),
    (Surface, 'nope', 2, 40),
    (Surface, 'nope', 3, 60),
    (Surface, 'dirichlet_surface', 3, 30),
    (Surface, 'neumann_surface', 3, 30),
    (Surface, 'asc_surface', 3, 30),
]

@pytest.mark.parametrize("surface, surf_type, surf_index, expected", s_starts)
def test_calc_s_start(surface, surf_type, surf_index, expected):
    surface.xi = list(range(10))
    surface.surf_type = surf_type
    surf_array = [surface, surface, surface]

    assert calc_s_start(surf_index, surf_array) == expected
