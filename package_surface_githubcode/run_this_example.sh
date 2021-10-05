#!/bin/bash
#
# This script runs the mesher and solver using this example setup from the paper :
# Bottero Alexis, Cristini Paul, Komatitsch Dimitri, Asch Mark. An axisymmetric time-domain spectral-element
# method for full-wave simulations: Application to ocean acoustics. J. Acoust. Soc. Am. 140, 3520 (2016).
# http://scitation.aip.org/content/asa/journal/jasa/140/5/10.1121/1.4965964?aemail=author

echo "Paper axisym JASA simulation: `date`"
currentdir=`pwd`

# Sets up directory structure in current example directoy
echo
echo " Setting up the simulation..."
echo

# mkdir -p OUTPUT_FILES
# mkdir -p DATA

# Clean output files
if [ "$1" != "noclean" ]; then
  echo "cleaning OUTPUT_FILES/"
  rm -rf OUTPUT_FILES/*
fi

# Sets up local DATA/ directory
cd DATA/
ln -s Par_file Par_file
ln -s SOURCE SOURCE
cd ../

# Compiles executables in root directory
cd ../../
# make clean
#./configure CC=icc FC=ifort MPIFC=mpif90 --with-mpi --enable-double-precision --with-scotch-dir=/home/cristini/CODES/scotch_6.0.0
#./configure CC=icc FC=ifort --with-scotch-dir=/home/cristini/CODES/scotch_6.0.0 # Without MPI
# make # > tmp.log
cd $currentdir

# Link executables
rm -f xmeshfem2D xspecfem2D
#ln -s /home/yjgao/install/specfem2d-master_20180907/specfem2d/bin/xmeshfem2D
#ln -s /home/yjgao/install/specfem2d-master_20180907/specfem2d/bin/xspecfem2D
ln -s /home/yjgao/install/specfem2d-devel/bin/xmeshfem2D
ln -s /home/yjgao/install/specfem2d-devel/bin/xspecfem2D


# Store setup
cp DATA/Par_file OUTPUT_FILES/
cp DATA/SOURCE OUTPUT_FILES/

# Get the number of processors
NPROC=`grep NPROC DATA/Par_file | cut -d = -f 2 | cut -d \# -f 1 | tr -d ' '`

# Run database generation
echo
echo " Running mesher..."
echo
./xmeshfem2D

if [ "$NPROC" -eq 1 ]; then # This is a serial simulation
  echo
  echo " Running solver..."
  echo
  ./xspecfem2D
else # This is a MPI simulation
  echo
  echo " Running solver on $NPROC processors..."
  echo
  /usr/bin/mpirun -np $NPROC ./xspecfem2D
fi

# Store output
cp DATA/*SOURCE* DATA/*STATIONS* OUTPUT_FILES

echo
echo "See results in directory: OUTPUT_FILES/"
echo
echo "Done!"
date
