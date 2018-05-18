import aco
import numpy as np
import pandas as pd

dist_mat = pd.read_csv('tc4.csv', header=None).values

g = aco.Graph(dist_mat)

ac = aco.ACO(g, num_colony=10,iter=5,beta=3,rho=0.5)
sol = ac.solve_step()

cost_over_time = []

for i in sol:
    final = i
    cost_over_time.append(i.travel_dist)

thefile = open('cost_over_time4.txt', 'w')
for item in cost_over_time:
  thefile.write("%s\n" % item)
thefile.close()
thefile = open('solution4.txt', 'w')
for item in final.visited_city:
  thefile.write("%s\n" % item)
thefile.close()
