import pytest
from unittest.mock import patch

from pygbe.matrixfree import locate_s_in_RHS

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
def test_locate_s_in_RHS(surface, surf_type, surf_index, expected):
    surface.xi = list(range(10))
    surface.surf_type = surf_type
    surf_array = [surface, surface, surface]

    assert locate_s_in_RHS(surf_index, surf_array) == expected
