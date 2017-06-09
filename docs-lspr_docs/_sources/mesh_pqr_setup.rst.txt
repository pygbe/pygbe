Generate meshes and pqr
-----------------------

For the problems that involve proteins we use MSMS to generate the corresponding 
mesh. In order to achieve this we need to follow few steps:

1. Download a `pdb` file from the Protein Data Bank http://www.rcsb.org/pdb/home/home.do
2. Get the `pqr` file from the `pdb`.
3. Install pdb2pqr from here: https://github.com/Electrostatics/apbs-pdb2pqr/releases or use the webserver: http://nbcr-222.ucsd.edu/pdb2pqr_2.1.1/
4. Create `pqr` file. If you are running from command line:
        .. code:: console

            pdb2pqr --ff=amber input.pdb output.pqr

        where we picked `amber` as the force field.

3. Create `xyzr` file:
    .. code:: console
        
        awk '{print $6,$7,$8,$10}' protein.pqr > protein.xyzr
    
    We are extracting certain columns from the `protein.pqr` we create in the 
    previous step. We need the `xyzr` file as input for MSMS.

4. Download MSMS from http://mgltools.scripps.edu/downloads.
5. Create mesh from command line:
    
    .. code:: console

        /msms_XX_XX/msms_XX_XX. -if protein.xyzr -of protein1 -prob 1.4 -d 2 -no_header
        
    If you want all the components, inside cavities you will add the flag
    
    .. code:: console
    
        -all_components

    Where _XX_XX correspond to the version you download. You can find information
    about the different flags here http://mgl.scripps.edu/people/sanner/html/msms_man.html
    