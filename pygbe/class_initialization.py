"""
It contains the necessary functions to set up the surface to be solved.
"""
import numpy
from pygbe.util.read_data import read_fields, read_surface
from pygbe.classes import Field, Surface


def initialize_surface(field_array, filename, param):
    """
    Initialize the surface of the molecule.

    Arguments
    ---------
    field_array: array, contains the Field classes of each region on the surface.
    filename   : name of the file that contains the surface information.
    param      : class, parameters related to the surface.

    Returns
    -------
    surf_array : array, contains the surface classes of each region on the
                        surface.
    """

    surf_array = []

    # Read filenames for surfaces
    files, surf_type, phi0_file = read_surface(filename)
    Nsurf = len(files)

    for i in range(Nsurf):
        print('\nReading surface {} from file {}'.format(i, files[i]))

        s = Surface(Nsurf, surf_type[i], phi0_file[i])
        s.define_surface(files[i], param)
        s.define_regions(field_array, i)
        surf_array.append(s)

    return surf_array


def initialize_field(filename, param, field=None):
    """
    Initialize all the regions in the surface to be solved.

    Arguments
    ---------
    filename   : name of the file that contains the surface information.
    param      : class, parameters related to the surface.
    field      : dictionary with preloaded field values for programmatic
                 interaction with PyGBe

    Returns
    -------
    field_array: array, contains the Field classes of each region on the surface.
    """

    if not field:
        field = read_fields(filename)

    for key in ['E', 'kappa']:
        for i, e in enumerate(field[key]):
            if not numpy.iscomplexobj(e) and not isinstance(e, str):
                field[key][i] = param.REAL(field[key][i])

    Nfield = len(field['LorY'])
    field_array = []
    Nchild_aux = 0
    for i in range(Nfield):
        field_aux = Field(field['LorY'][i], field['kappa'][i], field['E'][i],
                          field['coulomb'][i], field['pot'][i])

        if int(field['charges'][i]) == 1:  # if there are charges
            field_aux.load_charges(field['qfile'][i], param.REAL)
        if int(field['Nparent'][i]) == 1:  # if it is an enclosed region
            field_aux.parent.append(int(field['parent'][i]))
            # pointer to parent surface (enclosing surface)
        if int(field['Nchild'][i]) > 0:  # if there are enclosed regions inside
            for j in range(int(field['Nchild'][i])):
                field_aux.child.append(int(field['child'][Nchild_aux + j])
                                       )  # Loop over children to get pointers
            Nchild_aux += int(field['Nchild'][i])  # Point to child for next surface
        if field['pot'][i] == 1:
            param.E_field.append(i)  # Field where surface energy is calculated

        field_array.append(field_aux)
    return field_array
