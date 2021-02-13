# Safe Reinforcement Learning
Safe Reinforcement Learning Using Off-Policy Policy Evaluation

This project simulates the application of RL to a real problem, like a medical application or business application, where deploying a policy worse than the current one would be dangerous or costly. The goal here is to generate good policies without the knowledge of underlying MDP using only the past data.


## Installation
To run it requires numpy, matplotlib, scipy and cma: 

pip install numpy
pip install matplotlib
pip install scipy
pip install cma(>2.7.0)
pip install gym(>0.15.4)

## Run
To run the project(may take a long time to run):
python -m saferl.hcpi4

This will generate 20 files such as 'data/multiple1.csv' and 'data/multiple1.txt'. First file consists of theta parameters while second contains information of performance for these parameters.

## Params
Once the results are obtained, to generate paramters run:
python -m saferl.plot

This will generate 100 files like 'results/1.csv' which will contain the best parameters for the new policy.

## Authors
Aarshee Mishra

License
This project is licensed under the MIT License
