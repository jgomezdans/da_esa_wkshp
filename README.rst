ESA Land DA workshop
=======================

This repository contains material and instructions for the ESA Land DA workshop that will take place at University College London on Mar 31/Apr 1 2014.

To install the required Python modules
*****************************************

We assume that you have numpy, scipy and matplotlib already installed (you can try Anaconda python to install the whole Python stack). Then, issue the following commands:

.. code::

   pip install ~/SHARED/GomezDans/eoldas_ng-0.1.tar.gz --user --upgrade
   pip install ~/SHARED/GomezDans/prosail_fortran-1.1.0.tar.gz --user --upgrade
   pip install ~/SHARED/GomezDans/gp_emulator-1.2.tar.gz --user --upgrade
   ~/SHARED/GomezDans/test.py
   mkdir eoldas_tutorial
   cp ~/SHARED/GomezDans/*.txt eoldas_tutorial/
   cp ~/SHARED/GomezDans/*ipynb eoldas_tutorial/
   cp ~/SHARED/GomezDans/*.npz eoldas_tutorial/
   cd eoldas_tutorial
   ipython notebook 
   #


You can probably just copy and paste those lines into an UNIX terminal. Note that these commands will only work if you have an account on the UCL system, and an account with an username like `daesaXX`, where `XX` are two digits. In case you don't have that, you can get hold of the required packages by cloning the current repository with git, changing to that directory and issuing the following commands:

.. code::

   pip install ./eoldas_ng-0.1.tar.gz --user --upgrade
   pip install ./prosail_fortran-1.1.0.tar.gz --user --upgrade
   pip install ./gp_emulator-1.2.tar.gz --user --upgrade
   ./GomezDans/test.py

Any issues or comments, add an issue here on github, or send me an email on `j.gomez-dans@ucl.ac.uk`.
