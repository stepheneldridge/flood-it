# Stephen Eldridge
from timeit import default_timer as now
import copy
import itertools
import json
import math
import os
import random
import traceback


def runWisdomOfCrowds(width, height, colors, pipe, new_path=False):
    try:
        if new_path:
            puzzle = Puzzle(range(colors), Puzzle.createGrid(width=width, height=height, colors=colors))
        else:
            try:
                file = File("flood%sx%sc%s.json" % (width, height, colors))
                puzzle = Puzzle(range(file.colors), file.grid)
            except Exception:
                puzzle = Puzzle(range(colors), Puzzle.createGrid(width=width, height=height, colors=colors))
                File.write(puzzle)
        woc = WisdomOfCrowds(puzzle, update=lambda x: pipe.send(x))
        woc.run()
        pipe.send(puzzle)
        pipe.send("%s seconds" % woc.delta)
    except Exception as e:
        traceback.print_exc()
        pipe.send(e)
    pipe.close()


class File():
    def __init__(self, file_name):
        data = self.parse_file(file_name)
        # sets all variables in file to class attributes
        for i in data:
            setattr(self, i.lower(), data[i])

    def parse_file(self, file_name):
        assert os.path.isfile(file_name)
        file = open(file_name, 'r')
        data = json.load(file)
        file.close()
        return data

    @staticmethod
    def write(puzzle):
        file_name = "flood%sx%sc%s.json" % (puzzle.width, puzzle.height, len(puzzle.colors))
        with open(file_name, 'w') as file:
            data = {
                "colors": len(puzzle.colors),
                "width": puzzle.width,
                "height": puzzle.height,
                "grid": puzzle.grid
            }
            json.dump(data, file)


class Puzzle():
    def __init__(self, colors, grid):
        self.colors = colors
        self.grid = grid
        self.width = len(grid)
        self.height = len(grid[0])
        self.shortest_path = []
        self.path_weight = 1 << 32

    def __str__(self):
        return "%sx%s c=%s" % (self.width, self.height, len(self.colors))

    def copy(self):
        return Puzzle(self.colors, self.grid)

    def solve(self, solution, update):
        self.update = update
        self.get_solution_weight(solution)
        self.grid = self.test_grid

    def change_color(self, grid, x, y, color, new_color):
        if x >= 0 and x < self.width and y >= 0 and y < self.height and grid[x][y] == color:
            grid[x][y] = new_color
            return True
        return False

    def get_solution_weight(self, solution):
        self.test_grid = copy.deepcopy(self.grid)
        for index, color in enumerate(solution + [-1]):
            base_color = self.test_grid[0][0]
            if base_color == color:
                continue
            filled = self.flood_fill(0, 0, self.test_grid, base_color, color)
            if hasattr(self, 'update'):
                self.update(Puzzle(self.colors, self.test_grid))
        return filled / ((self.width * self.height) + len(solution))

    def flood_fill(self, x, y, grid, base_color, new_color):
        count = 0
        if self.change_color(grid, x, y, base_color, new_color):
            count += 1
            count += self.flood_fill(x + 1, y, grid, base_color, new_color)
            count += self.flood_fill(x - 1, y, grid, base_color, new_color)
            count += self.flood_fill(x, y + 1, grid, base_color, new_color)
            count += self.flood_fill(x, y - 1, grid, base_color, new_color)
        return count

    @staticmethod
    def createGrid(width=10, height=10, colors=2):
        return [
            [random.randint(0, colors - 1) for _ in range(height)] for _ in range(width)
        ]


class WisdomOfCrowds():
    def __init__(self, puzzle, update=lambda x: None):
        self.update = update
        self.puzzle = puzzle

    def run(self):
        self.start_time = now()
        self.update(self.puzzle)
        ga = GeneticAlgorithm(self.puzzle.copy(), update=self.update)
        ga.run(100, 10)
        self.puzzle.solve(ga.puzzle.shortest_path, self.update)
        self.delta = now() - self.start_time
        return
        paths = []
        for _ in range(100):  # runs 100 genetic algorithms
            ga = GeneticAlgorithm(self.puzzle.copy())
            ga.distances = self.distances
            ga.run(10, 100)
            print(ga.path.short_length)
            paths.extend(self.dedupe(sorted(ga.paths, key=lambda a: a['length'])))
        self.combine_paths(paths)

    def dedupe(self, paths):  # removed duplicate paths
        unique = []
        for i in paths:
            if i['path'] not in unique:
                unique.append(i['path'])
        return unique

    def combine_paths(self, paths):
        print(len(paths))  # how many paths were left after deduping
        node = 1
        self.path.short_path = [1]
        length = len(self.path.keys)
        unused = self.path.keys.copy()
        unused.remove(1)
        for _ in range(len(unused)):
            counts = [0] * len(unused)
            for i in paths:
                index = i.index(node)
                next_node = i[(index + 1) % length]
                prev_node = i[index - 1]
                if next_node in unused:  # counts the occurances of the next node amoung solutions
                    counts[unused.index(next_node)] += 1
                if prev_node in unused:
                    counts[unused.index(prev_node)] += 1
            node = unused[counts.index(max(counts))]
            unused.remove(node)
            self.path.short_path.append(node)
        self.path.short_length = GeneticAlgorithm.get_path_length(self, self.path.short_path)
        self.update(self.path)
        print(self.path)

    def distance(self, a, b):  # generic distance calculation
        return math.sqrt(sum([(a[i] - b[i]) ** 2 for i in range(len(a))]))

    def generate_lookup(self):
        # creates 2D matrix to store precalculated distances
        self.distances = {}
        keys = self.path.keys
        for i in keys:
            self.distances[i] = {}
            for j in keys:
                if i != j:  # ignores distances to itself
                    self.distances[i][j] = self.distance(self.path.nodes[i], self.path.nodes[j])


class GeneticAlgorithm():
    def __init__(self, puzzle, update=lambda x: None):
        self.update = update
        self.puzzle = puzzle

    def run(self, size, gens):
        # array of path arrays
        paths = self.random_generation(size)
        self.mutation_rate = 0.5
        generation = 1
        while generation <= gens:
            weights = []
            for path in paths:
                weights.append(self.puzzle.get_solution_weight(path))
            self.paths = paths
            best_path = paths[weights.index(max(weights))]
            self.puzzle.shortest_path = best_path
            self.puzzle.path_weight = weights.index(max(weights))
            self.update(best_path)
            paths = self.generate_generation(paths, weights, size)  # generate children
            generation += 1

    def generate_generation(self, paths, weights, count):
        top = [paths[weights.index(max(weights))]]
        children = top
        for i in range(count - len(top)):
            parents = random.choices(paths, weights=weights, k=2)  # random weighted selection with replacement
            path = self.cross_parents(parents[0], parents[1])  # creates child path
            self.mutate(path, r=self.mutation_rate)  # tries to mutate
            children.append(path)
        return children  # returns the children

    def mutate(self, path, r=0):
        if random.random() < r:
            if random.random() < 0.5:
                del path[random.randint(0, len(path) - 1)]
            else:
                path.insert(random.randint(0, len(path) + 1), random.choice(self.puzzle.colors))

    def cross_parents(self, a, b):
        child = [a[0]]
        current = 0
        a_index = 1
        b_index = 0
        while(True):
            if current == 0:
                if a_index >= len(a):
                    break
                if child[-1] != a[a_index]:
                    child.append(a[a_index])
                a_index += 1
                if b_index < len(b) and child[-1] == b[b_index]:
                    b_index += 1
                    current = 1
            else:
                if b_index >= len(b):
                    break
                if child[-1] != b[b_index]:
                    child.append(b[b_index])
                b_index += 1
                if a_index < len(a) and child[-1] == a[a_index]:
                    a_index += 1
                    current = 0
        return child

    def random_generation(self, count):
        side = max(self.puzzle.width, self.puzzle.height)
        solved = False
        perms = list(map(list, itertools.permutations(self.puzzle.colors)))
        paths = []
        while(len(paths) != count):
            lst = []
            for i in range(side * 2):
                lst.append(random.choice(perms))
            temp = self.puzzle.is_solved(lst)
            if temp:
                paths.append(temp)
        return paths


if __name__ == '__main__':  # starts the application and creates the window only if called from command line
    from gui import init
    init()
