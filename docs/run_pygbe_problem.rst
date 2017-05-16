Programmatic use of PyGBe
-------------------------


.. note:: These options reflect what we do to configure problems in PyGBe, many
          of these choices are just that, choices.

.. code:: console

          awk '{print $6,$7,$8,$10}' 1K4R_pdb2pqr_modR.pqr > molecule.xyzr
          pdb2pqr --ff=amber input.pdb output.pqr
          msms -if lys.xyzr -of lys1 -prob 1.4 -d 2 -no_header


   Steps:

          Download a pdb file from http://www.rcsb.org/pdb/home/home.do Protein Data Bank


   Software needed:
          pdb2pqr (possible to use webserver: http://nbcr-222.ucsd.edu/pdb2pqr_2.1.1/)
   or install latest release from here: https://github.com/Electrostatics/apbs-pdb2pqr/releases

          msms (for proteins with < 10000 atoms, otherwise it chokes): http://mgltools.scripps.edu/downloads

          nanoshaper (works well) -- source is behind login wall but can download binary here: https://github.com/lwwilson1/mesh_routines/


in file named `surfaceConfiguration.prm`
    Grid_scale = 1.0
    Grid_perfil = 90.0 
    XYZR_FileName = molecule.xyzr
    Build_epsilon_maps = false
    Build_status_map = false
    Save_Mesh_MSMS_Format = true
    Compute_Vertex_Normals = true
    Surface = ses  
    Smooth_Mesh = true
    Skin_Surface_Parameter = 0.45
    Cavity_Detection_Filling = false
    Conditional_Volume_Filling_Value = 11.4
    Keep_Water_Shaped_Cavities = false
    Probe_Radius = 1.4
    Accurate_Triangulation = true
    Triangulation = true
    Check_duplicated_vertices = true
    Save_Status_map = false
    Save_PovRay = false




###############################################################################
###################### NanoShaper 0.3.1 Configuration file  ###################
###############################################################################

################################ Grid params ##################################

# Grid scale 
Grid_scale = 5

# Percentage that the surface occupies with respect to the total grid volume 
# default value is 90.0; in the case of very small molecules (e.g. fullerene) keep more margin 
# (e.g. lower the value to 50%)
Grid_perfil = 90.0 

# Input atoms xyzr file name
XYZR_FileName = dir_TestProteins/1vjw.xyzr
# if you enable fullerene (or a small molecule) use Grid_perfil = 70.0 or lower

############################## Internal maps ##################################

# Enable/Disable build of epsilon (dielectric) map
Build_epsilon_maps  = false

# Enable/Disable build of map for cavity detection and for a non analytical
# triangulation
Build_status_map = false

########################## Surface Type params ################################

# Possible values: skin,blobby,msms,mesh,ses
Surface = skin
Compute_Vertex_Normals = true
Save_Mesh_MSMS_Format = true

# Apply final surface smoothing
Smooth_Mesh = true

# Number of total threads
Number_thread = 32 #default value is 32

# Skin surface parameter [0.05,0.95]
# the extrema are possibly not numerically stable,
# the suggested range is [0.15,0.95]
# default value is 0.45
Skin_Surface_Parameter = 0.45

# Blobbyness value for the blobby surface [-0.5,-5.0]
# default value is -2.5
Blobbyness = -2.5 

# Name of the input surface file used if mesh Surface is enabled or msms.
# In case of msms remove any extension, .face and .vert file will be 
# automatically loaded.
# In case of mesh, .off and .ply files are supported
Surface_File_Name = triangulatedSurf.off

######################## Surface Processing params ############################
# Enable or disable cavity detection together with the volume conditional
# filling of voids and cavities
Cavity_Detection_Filling = false

# It is the value of the minimal volume of a cavity to get filled if 
# cavity detection is enabled. 
# The default value is an approximation of the volume of the water molecule 
# default value is 11.4, this is the approximate volume of a water molecule 
# in Angstrom
Conditional_Volume_Filling_Value = 11.4

# If this flag is true, cavities where a sphere of Probe_Radius cannot fit, 
# are removed.
# Use this feature when cavity detection is enabled to filter out bad shaped 
# cavities whose
# volume is higher than Conditional_Volume_Filling_Value.
Keep_Water_Shaped_Cavities = false

# The radius of the sphere that represents a water molecule in Angstrom
# default value is 1.4 Angstrom
Probe_Radius = 1.4

# Enable accurate triangulation: if accurate triangulation is enable all points 
# are sampled from the original surface. If disabled the points are not 
# analytically sampled and an high memory saving can be obtained
# together with a 3x speed-up on ray casting if both epsmap is disabled
# MC phase will be slower because vertices are calculated on the fly
Accurate_Triangulation = true

# Perform triangulation using a single ray-casting process. Vertex data is 
# inferred.
Triangulation = true

# Check duplicated vertices when reading
Check_duplicated_vertices = true

# If true save the status map. Enable this for cavity detection and 
# visualization of the coloured FD grid
Save_Status_map = false

# Save Skin/SES in a PovRay file for ray-tracing. 
# This is a purely graphics representation because the surface is not in
# a left handed system as it should in Pov-Ray
Save_PovRay = false

####################### Acceleration Data Structures ##########################

# Mesh Projection(3D)/Ray Casting (3D) acceleration grid
# Increase *_size to increase performance and memory usage.
# Increase *_cell if requested by NanoShaper

# default 100
Max_mesh_auxiliary_grid_size = 100
# default 250
Max_mesh_patches_per_auxiliary_grid_cell = 250;
# default 100
Max_mesh_auxiliary_grid_2d_size = 100
# default 250
Max_mesh_patches_per_auxiliary_grid_2d_cell = 250 

# SES Projection(3D)/Ray Casting (3D) acceleration grid
# default 100
Max_ses_patches_auxiliary_grid_size = 100
# default 400
Max_ses_patches_per_auxiliary_grid_cell = 400
# defualt 40
Max_ses_patches_auxiliary_grid_2d_size = 50
# default 400
Max_ses_patches_per_auxiliary_grid_2d_cell = 400

# Skin Projection(3D)/Ray Casting (3D) acceleration grid
# default 100
Max_skin_patches_auxiliary_grid_size = 100
# default 400
Max_skin_patches_per_auxiliary_grid_cell = 400
# default 150
Max_skin_patches_auxiliary_grid_2d_size = 150
# default 400
Max_skin_patches_per_auxiliary_grid_2d_cell = 200

###############################################################################
