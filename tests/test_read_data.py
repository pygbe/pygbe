from pygbe.util import read_data

import numpy

import pytest

def test_off_file_not_found():
    with pytest.raises(FileNotFoundError):
        read_data.read_off_file('notarealfile', numpy.float32)

def test_off_file_bad_type():
    with pytest.raises(TypeError):
        read_data.read_off_file('files/mesh.off', 'notatype')

def test_off_not_off_mesh():
    with pytest.raises(ValueError):
        read_data.read_off_file('files/notmesh.off', numpy.float32)

def test_off_bad_off_mesh():
    with pytest.raises(ValueError):
        read_data.read_off_file('files/badmesh.off', numpy.float32)

def test_read_off_mesh():
    vert, tri = read_data.read_off_file('files/mesh.off', numpy.float64)
    assert all(tri.ravel() == numpy.ones(10, dtype=numpy.int32))
    assert all(vert.ravel() == numpy.arange(1,16))
