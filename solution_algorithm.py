# Stephen Eldridge
from timeit import default_timer as now
import copy
import json
import math
import os
import random
import traceback
from collections import Counter


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

    def change_color(self, grid, x, y, color, new_color):
        if x >= 0 and x < self.width and y >= 0 and y < self.height and grid[x][y] == color:
            grid[x][y] = new_color
            return True
        return False

    def fill(self, n, color):
        unchecked_neighbors = n
        bad_neighbors = []
        added = 0
        while len(unchecked_neighbors) > 0:
            neighbor = unchecked_neighbors.pop()
            if self.test_grid[neighbor[0]][neighbor[1]] == -1:
                continue
            if self.test_grid[neighbor[0]][neighbor[1]] != color:
                bad_neighbors.append(neighbor)
                continue
            self.test_grid[neighbor[0]][neighbor[1]] = -1
            added += 1
            unchecked_neighbors.extend(self.get_neighbors(*neighbor))
        return bad_neighbors, added

    def is_solved(self, solution):
        self.test_grid = copy.deepcopy(self.grid)
        neighbors = [[0, 0]]
        base_color = self.test_grid[0][0]
        to_remove = []
        neighbors, total = self.fill(neighbors, base_color)
        size = self.width * self.height
        for index, color in enumerate(solution):
            neighbors, added = self.fill(neighbors, color)
            total += added
            if added == 0:
                to_remove.insert(0, index)
            if total == size:
                solution = solution[0:index + 1]
                break
        for i in to_remove:
            solution.pop(i)
        return solution if total == size else None

    def get_neighbors(self, x, y):
        neighbors = []
        if self.is_valid_neighbor(x + 1, y):
            neighbors.append([x + 1, y])
        if self.is_valid_neighbor(x - 1, y):
            neighbors.append([x - 1, y])
        if self.is_valid_neighbor(x, y + 1):
            neighbors.append([x, y + 1])
        if self.is_valid_neighbor(x, y - 1):
            neighbors.append([x, y - 1])
        return neighbors

    def is_valid_neighbor(self, x, y):
        return x >= 0 and x < self.width and y >= 0 and y < self.height

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
        ga.run(100, 100)
        self.delta = now() - self.start_time
        self.combine_paths(ga.paths)

    def combine_paths(self, paths):
        solution = []
        while len(paths) > 0:
            first_value = []
            for i in paths:
                first_value.append(i[0])
            first_node = Counter(first_value).most_common(1)[0][0]
            solution.append(first_node)
            for j in paths:
                if j[0] == first_node:
                    j.pop(0)
                    if len(j) == 0:
                        paths.remove(j)
            else:
                continue
        solved = self.puzzle.is_solved(solution)
        print("Final Solution: ", solved)


class GeneticAlgorithm():
    def __init__(self, puzzle, update=lambda x: None):
        self.update = update
        self.puzzle = puzzle

    def run(self, size, gens):
        # array of path arrays
        paths = self.random_generation(size)
        self.mutation_rate = 0.1
        generation = 1
        while generation <= gens:
            weights = []
            for path in paths:
                weights.append(1 / len(path))
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
            for i in range(random.randint(1, 10)):
                path.insert(random.randint(0, len(path) + 1), random.choice(self.puzzle.colors))

    def split(self, path):
        offset = random.randint(1, len(path) - 2)
        return [path[offset:], path[:offset]]

    def lcs(self, a, b):
        for i in range(len(a)):
            match = True
            for j in range(len(a) - i):
                if j >= len(b):
                    break
                if a[i + j] != b[j]:
                    match = False
                    break
            if match:
                return a[0:i] + b
        return a + b

    def cross_parents(self, a, b):
        _a = self.split(a)
        _b = self.split(b)
        child = self.lcs(_a[0], _b[1]) + _b[0] + _a[1]
        return self.puzzle.is_solved(child)

    def random_generation(self, count):
        side = max(self.puzzle.width, self.puzzle.height)
        C = len(self.puzzle.colors)
        upper_bound = int(math.ceil(2 * side + math.sqrt(2 * C) * side + C))
        paths = []
        while(len(paths) != count):
            lst = [random.randint(0, C) for _ in range(upper_bound)]
            temp = self.puzzle.is_solved(lst)
            if temp:
                paths.append(temp)
        return paths


if __name__ == '__main__':  # starts the application and creates the window only if called from command line
    # from gui import init
    # init()
    from unittest.mock import Mock
    print("DEBUG")
    runWisdomOfCrowds(12, 12, 6, Mock(send=print), False)
    print("DONE")
