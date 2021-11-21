# Mitochondrial-Network-Model


1. Main.py / Main.update.py
   Main program to run the fitting process.
   Use multiprocessing to speed up.
   1 variable * 3 version is used in the paper, where the other two have bugs with bandwidth of KDE.
   
   Main.py is used for glucose fitting, and Main_update.py is used for toxicity fitting.
   Main_update.py is the latest version, probabily with least bugs.
  
2. KLDivergence.py / KLDivergence_update.py
   Functions of KLD and KDE.
   KLDivergence is used in Main.py / Main_update.py, where KLDivergence_update.py still have some bugs with bandwidth of KDE.
   
3. networkmodel.py
   The agent-based network model used in Main.py / Main_update.py.
