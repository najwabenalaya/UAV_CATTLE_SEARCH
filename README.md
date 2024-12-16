# UAV_CATTLE_SEARCH
  

- This repository is the official implementation MILP to solve UAV cattle search path planning problem
- The paper: https://ieeexplore.ieee.org/document/9963839
- The authors: Najoua Benalaya,  Ichrak Amdouni, Cedric Adjih, Anis Laouiti, Leila Saidane
- Three MILP were developed to solve three instances of the problem

# License Requirement:
Gurobi license is required to run our implmentation:
Link:https://www.gurobi.com/academia/academic-program-and-licenses/?gad_source=1&gclid=Cj0KCQiAvP-6BhDyARIsAJ3uv7Y5TjlriRlay88wyPwjWnrvo7DVFER2K_nue6JItohRK4ZJqAcMU0MaAiuvEALw_wcB

# Running the UCS-ST (Stationary target):
python stationary_target.py --x-size 6 --y-size 6 --hover-time 1 --uniform-poc --nb-poc  3  --speed 1 --seed 1 

# Running UCS-SMT (Single moving target):
python moving_target.py --max-time 20 --mobility 0.03 --x-size 11 --y-size 11 --nb-poc 10 --min-pd 0.8

# Running UCS-TMT (Two moving target):
python two_moving_target.py --x-size 6 --y-size 6 --max-time 10 --mobility-c1 0.01 --mobility-c2 0.01 --nb-poc  3  --min-pd 0.99 --seed 1 --objective-pd


