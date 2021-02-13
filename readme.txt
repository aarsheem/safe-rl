COMPSCI 687 Project
This is the final project in the course COMPSCI 687 at UMass, Amherst.


Installation
To run it requires numpy, matplotlib, scipy and cma: 

pip install numpy
pip install matplotlib
pip install scipy
pip install cma(>2.7.0)
pip install gym(>0.15.4)

To run the project(may take a long time to run):
python -m saferl.hcpi4

This will generate 20 files such as 'data/multiple1.csv' and 'data/multiple1.txt'. First file consists of theta parameters while second contains information of performance for these parameters.

Once the results are obtained, to generate paramters run:
python -m saferl.plot

This will generate 100 files like 'results/1.csv' which will contain the best parameters.

Authors
Aarshee Mishra - Graduate Student

License
This project is licensed under the MIT License
