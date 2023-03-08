# Mitochondrial Network Model

By Ching-Hsian Chu


1. `run.py`
   Main program to run the fitting process.
   Multiprocessing is used to speed up.

2. `KLDivergence.py` / `KLDivergence_update.py`
   Functions of KLD and KDE.
   KLDivergence is used in Main.py / Main_update.py, where KLDivergence_update.py still have some bugs with bandwidth of KDE.

3. `networkmodel.py`
   The agent-based network model used in Main.py / Main_update.py.

4. `myGA.py`
   Self coded Genetic Algorithm, but didn't be used in the fittings (used package instead).
   Not used in Main.py / Main_update.py.

5. `oneVSmulti_demo.py`
   A demo showing the advantages of multiprocessing. Not used in Main.py / Main_update.py.

6. `data/XXX_fitting.csv`
   Image analysis data used in the toxicity fittings.

7. `image_processing_macro.txt`
   Codes of macro for image processing using ImageJ. Results in csv form would be outputted, which are used for Main.py / Main_update.py and plot_demo.ipynb.

8. `plot_demo.ipynb`
   python codes for plotting the boxplot of the results of image analysis (csv from image_processing_macro)

9. `netinfo_plotting.ipynb`
   python code for plotting the simulated network (agent-based).

10. `smalltest.py`
   Small-scale test script for continuous integration to make sure the code could run.


Workflow:

1. Raw images (fluorescent images of mitochondria)
2. Use ImageJ (FIJI) to run the preprocessing (codes in image_processing_macro.txt), and you will get the csv outputs. of image analysis.
3. Use plot_demo to plot the result in step 2.
4. Extract Ng1/N, Ng2/N, AvgDeg information from step 2, saved them as the form of XXX_fitting.csv.
5. Run Main.py for glucose fitting (XXX_fitting.csv is needed, for example 0X_fitting.csv, 3X_fitting.csv etc.), and run Main_update.py for toxicity fitting (XXX_fitting.csv is needed, for example FCCP_fitting.csv, control_fitting.csv etc.) to get the answer of C1, C2 by GA.
6. Run step 5. several times and manually collect the results (i.e. several C1, C2), which could be used for further analysis.
7. For example, you could calculate the mean value of C1, C2 for each condition, and use netinfo_plotting.ipynb to plot the simulated network (agent-based, 11 reactions)


## Dependencies

python: 3.8+ (Current tested against Python 3.11)

Package: See `requirements.txt`
1. numpy
2. pandas
3. matplolib
4. seaborn
5. multiprocessing
7. sklearn
8. scipy
9. geneticalgorithm
10. networkx
