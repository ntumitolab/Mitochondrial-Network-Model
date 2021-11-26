# Mitochondrial-Network-Model


1. Main.py / Main_update.py
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
   
4. myGA.py
   Self coded Genetic Algorithm, but didn't be used in the fittings (used package instead).
   Not used in Main.py / Main_update.py.

5. oneVSmulti_demo.py
   A demo showing the advantages of multiprocessing. Not used in Main.py / Main_update.py.
   
6. XXX_fitting.csv
   Image analysis data used in the toxicity fittings.
   
7. image_processing_macro.txt
   Codes of macro for image processing using ImageJ. Results in csv form would be outputted, which are used for Main.py / Main_update.py and plot_demo.ipynb.
   
8. plot_demo.ipynb
   python codes for plotting the boxplot of the results of image analysis (csv from image_processing_macro)
