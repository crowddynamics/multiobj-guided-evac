# multiobj-guided-evac

Research code used [1].

The manuscript presents a procedure for solving the minimum time evacuation using rescue guides when accounting for different behavioral scenarios. The problem is formulated as a bi-objective scenario optimization problem, and it is solved with a combined numerical simulation and genetic algorithm (GA) procedure. The GA iteratively searches for the optimal evacuation plan, while the evacuation plan is evaluated with numerical simulations. This repository is its implementation. The numerical simulation model is the physics-inspired agent-based social force model. The GA is the Nondominated Sorting Genetic Algorithm II (NSGA-II) [1,2]. It is suited for solving bi-objective optimization problems; this repository includes my own implementation of it.

The numerical evacuation simulation and its GUI is based on research assistant Jaan Tollander's codes https://github.com/jaantollander/crowddynamics and https://github.com/jaantollander/crowddynamics-qtgui, which he created when he was working in our research group in Aalto University School of Science, Department of Mathematics and Systems Analysis summers 2016 and 2017.


<h4>Contents of repository</h4>

The repository includes codes for NSGA-II, simulation and a graphical user interface (GUI). The folders in the repository:

    crowddynamics-simulation contains files for running the GUI
    crowddynamics-qtgui contains the files that build the GUI
    crowddynamics contains all files for simulating the movement of a crowd
    genetic algorithm includes files to run the combined numerical simulation and NSGA-II
    misc includes files used in configuring the optimization problem

The numerical evacuation simulations are implemented in Python and the NSGA-II is implemented in Python and Bash that were run on a high performance computing cluster. It should be noted that the procedure is currently computationally very demanding.

See the "readme.txt" file in each folder for a more detailed overview of the codes in each folder.


<h4>Installing</h4>

Using Linux is recommended. The code works at least on Ubuntu 16.04. Do the following steps to install the repository:

    Install anaconda (https://docs.anaconda.com/anaconda/install/linux)
    Set environment variables export PATH=/.../anaconda3/bin:$PATH" and export PYTHONPATH=/.../anaconda3/bin:$PYTHONPATH"
    Clone the repository
    On terminal run conda config --add channels conda-forge
    Create a conda environment from the file crowddynamics/environment.yml
    On terminal run source activate optimal-guided-evacuation
    On terminal, in folder crowddynamics, run pip install --editable .
    Change to folder crowddynamics-qtgui and run pip install -r requirements.txt
    Run conda install pyqt=4
    Run conda install pyqtgraph==0.10.0
    Run conda install scikit-fmm==0.0.9
    Run pip install anytree==2.1.4
    Run pip install --editable .

You might occur problems in installing some of the python packages. You can install these packages manually using conda install or pip install.


<h4>References</h4>

[1] von Schantz, A., Ehtamo, H., & Hostikka, S. (2020). The minimum time evacuation of a crowd using rescue guides: a scenario-based approach. manuscript.

[2] Deb, K., Pratap, A., Agarwal, S., & Meyarivan, T. A. M. T. (2002). A fast and elitist multiobjective genetic algorithm: NSGA-II. IEEE transactions on evolutionary computation, 6(2), 182-197.

[3] Fortin, F. A., & Parizeau, M. (2013, July). Revisiting the NSGA-II crowding-distance computation. In Proceedings of the 15th annual conference on Genetic and evolutionary computation (pp. 623-630).
