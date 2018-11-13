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
        if new_path:  # new puzzle without saving
            puzzle = Puzzle(range(colors), Puzzle.createGrid(width=width, height=height, colors=colors))
        else:
            try:  # if the file exists, use it, otherwise make a new one
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

    def parse_file(self, file_name):  # reads the json file with the puzzle data
        assert os.path.isfile(file_name)
        file = open(file_name, 'r')
        data = json.load(file)
        file.close()
        return data

    @staticmethod
    def write(puzzle):
        file_name = "flood%sx%sc%s.json" % (puzzle.width, puzzle.height, len(puzzle.colors))
        with open(file_name, 'w') as file:  # writes json data to file
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

    def __str__(self):  # string output
        return "%sx%s c=%s" % (self.width, self.height, len(self.colors))

    def copy(self):  # deep copy of Puzzle
        return Puzzle(self.colors, self.grid)

    def fill(self, n, color):  # iterative flood fill
        unchecked_neighbors = n
        bad_neighbors = []
        added = 0
        while len(unchecked_neighbors) > 0:  # while there are still more points to check
            neighbor = unchecked_neighbors.pop()
            if self.test_grid[neighbor[0]][neighbor[1]] == -1:  # -1 is the color to prevent infinite loops
                continue
            if self.test_grid[neighbor[0]][neighbor[1]] != color:
                bad_neighbors.append(neighbor)  # these neightbors have a different color and will be used the next time
                continue
            self.test_grid[neighbor[0]][neighbor[1]] = -1  # set color to show tile was visited
            added += 1
            unchecked_neighbors.extend(self.get_neighbors(*neighbor))  # add the neighbors of that tile
        return bad_neighbors, added

    def is_solved(self, solution):
        self.test_grid = copy.deepcopy(self.grid)  # creates copy of board to solve
        neighbors = [[0, 0]]  # starts in the top corner
        base_color = self.test_grid[0][0]  # color of the top corner
        to_remove = []  # useless color changes
        neighbors, total = self.fill(neighbors, base_color)  # fill in your starting area
        size = self.width * self.height
        for index, color in enumerate(solution):
            neighbors, added = self.fill(neighbors, color)  # for every color, fill in with that color
            total += added
            if added == 0:  # if nothing is added, remove the color
                to_remove.insert(0, index)
            if total == size:  # if the puzzle is finished then stop
                solution = solution[0:index + 1]
                break
        for i in to_remove:  # remove useless colors
            solution.pop(i)
        return solution if total == size else None  # if it solved the puzzle return the solution

    def get_neighbors(self, x, y):
        neighbors = []  # checks all 4 cardinal directions to see if the tile is valid
        if self.is_valid_neighbor(x + 1, y):
            neighbors.append([x + 1, y])
        if self.is_valid_neighbor(x - 1, y):
            neighbors.append([x - 1, y])
        if self.is_valid_neighbor(x, y + 1):
            neighbors.append([x, y + 1])
        if self.is_valid_neighbor(x, y - 1):
            neighbors.append([x, y - 1])
        return neighbors  # returns all valid tiles

    def is_valid_neighbor(self, x, y):
        return x >= 0 and x < self.width and y >= 0 and y < self.height  # is the tile in the board

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
        ga.run(100, 100)  # runs population fo 100 for 100 generations
        self.delta = now() - self.start_time
        self.combine_paths(ga.paths)  # combines paths into 1 for WOC

    def combine_paths(self, paths):  # computers shortest common super sequence for all paths
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
        solved = self.puzzle.is_solved(solution)  # the final path is trimmed to optimize since the SCS is long
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
                weights.append(1 / len(path))  # weight is inverse to the path length, shorter is higher weight
            self.paths = paths
            best_path = paths[weights.index(max(weights))]
            self.puzzle.shortest_path = best_path
            self.puzzle.path_weight = weights.index(max(weights))
            self.update(best_path)
            paths = self.generate_generation(paths, weights, size)  # generate children
            generation += 1

    def generate_generation(self, paths, weights, count):
        top = [paths[weights.index(max(weights))]]  # keeps best parent
        children = top
        for i in range(count - len(top)):
            parents = random.choices(paths, weights=weights, k=2)  # random weighted selection with replacement
            path = self.cross_parents(parents[0], parents[1])  # creates child path
            self.mutate(path, r=self.mutation_rate)  # tries to mutate
            children.append(path)
        return children  # returns the children

    def mutate(self, path, r=0):
        if random.random() < r:
            for i in range(random.randint(1, 10)):  # randomly adds a new section of colors to the path
                path.insert(random.randint(0, len(path) + 1), random.choice(self.puzzle.colors))

    def split(self, path):  # randomly splits an array in 2
        offset = random.randint(1, len(path) - 2)
        return [path[offset:], path[:offset]]

    def lcs(self, a, b):  # actually returns the shortest common super sequence
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
        _a = self.split(a)  # cuts parents in 2 randomly
        _b = self.split(b)
        child = self.lcs(_a[0], _b[1]) + _b[0] + _a[1]
        return self.puzzle.is_solved(child)

    def random_generation(self, count):
        side = max(self.puzzle.width, self.puzzle.height)
        C = len(self.puzzle.colors)
        upper_bound = int(math.ceil(2 * side + math.sqrt(2 * C) * side + C))  # upper bound for number of steps
        paths = []
        while(len(paths) != count):
            lst = [random.randint(0, C) for _ in range(upper_bound)]  # generated random ints(colors) for the path
            temp = self.puzzle.is_solved(lst)  # runs through a solution checker
            if temp:
                paths.append(temp)  # if valid add to list
        return paths


if __name__ == '__main__':  # starts the application and creates the window only if called from command line
    # from gui import init
    # init()
    from unittest.mock import Mock
    print("DEBUG")  # runs code without gui
    runWisdomOfCrowds(12, 12, 6, Mock(send=print), False)
    print("DONE")
