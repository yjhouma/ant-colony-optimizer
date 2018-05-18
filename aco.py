import numpy as np
from tqdm import tqdm

class Graph:
    def __init__(self, distance_matrix):
        self.d = distance_matrix
        self.n = len(distance_matrix)
        self.t = self.initialize_pheromone()

    def initialize_pheromone(self):
        min_pher = 0.01
        return np.full((self.n,self.n), min_pher)

    def total_tour_distance(self, tour):
        # Tour is list of point
        dist = 0
        for i in range(len(tour)):
            origin = i % self.n
            dest = (i+1) % self.n
            dist += self.d[origin][dest]

        return dist

    def __len__(self):
        return self.n



class Ant:
    def __init__(self, city, beta, graph):
        self.city = city
        self.visited_city = [city]
        self.beta = beta
        self.travel_dist = 0
        self.start_city = city
        self.graph = graph
        self.solution = []
        self.available_city = [i for i in range(len(graph)) if i != city]
        self.ant_phero = np.zeros((self.graph.n,self.graph.n))

    def in_nest(self):
        return len(self.visited_city) == (len(self.graph) +1)

    def move(self):
        dest = self.choose_dest(self.available_city)
        return self.make_move(dest)

    def choose_dest(self, avail):
        if len(avail) == 0:
            return None
        elif len(avail) == 1:
            return avail[0]
        else:
            prob = []
            for i in avail:
                prob.append(self.calculate_prob(self.city, i))
            return np.random.choice(
                avail,
                p = prob
            )

    def calculate_prob(self, point_i, point_j):
        p = self.graph.t[point_i][point_j] * (self.graph.d[point_i][point_j]**(-1*self.beta))
        tot = 0
        for i in self.available_city:
            tot += self.graph.t[point_i][i] * (self.graph.d[point_i][i]**(-1*self.beta))
        p /= tot
        return p

    def make_move(self,dest):
        origin = self.city

        if dest is None:
            if self.in_nest == True:
                return None
            dest = self.start_city
        else:
            self.available_city.remove(dest)

        self.solution.append(dest)
        self.visited_city.append(dest)
        self.city = dest
        self.travel_dist += self.graph.d[origin][dest]
        self.ant_phero[origin][dest] = 1

        return (origin,dest)

    def __eq__(self, other):
        return self.travel_dist == other

    def __lt__(self,other):
        return self.travel_dist < other.travel_dist


class ACO:
    def __init__(self,graph, num_colony, beta=5, rho=0.7,iter=10):
        self.graph = graph
        self.num_col = num_colony
        self.beta=beta
        self.rho =rho
        self.min_pher = 0.01
        self.num_gen = iter
        self.ants = []

    def create_colony(self):
        self.ants = []
        points = [point for point in range(self.graph.n)]
        for ant in range(self.num_col):
            ant_city = np.random.choice(points)
            self.ants.append(Ant(ant_city,self.beta,self.graph))

    def solve(self):
        glob_best = None
        for gen in tqdm(range(self.num_gen),desc='solving'):
            self.create_colony()
            local_best = self.solve_for_this_gen()
            if glob_best is None or local_best < glob_best:
                glob_best = local_best
        return glob_best

    def solve_step(self):
        glob_best = None
        for gen in tqdm(range(self.num_gen),desc='solving'):
            self.create_colony()
            local_best = self.solve_for_this_gen()
            if glob_best is None or local_best < glob_best:
                glob_best = local_best
            yield glob_best

    def solve_for_this_gen(self):
        self.find_path()
        self.global_update()
        return sorted(self.ants)[0]

    def find_path(self):
        ant_is_home = 0
        while ant_is_home < self.num_col:
            for ant in self.ants:
                if not ant.in_nest():
                    org ,dest = ant.move()
                    self.local_update(org,dest)
                else:
                    ant_is_home+=1


    def local_update(self, org, dest):
        self.graph.t[org][dest] = max(self.min_pher, self.graph.t[org][dest]*self.rho)

    def global_update(self):
        ants = sorted(self.ants)
        pher = (1-self.rho)*self.graph.t
        for ant in ants:
            bst = 1/ant.travel_dist
            x = bst * ant.ant_phero
            pher += x
