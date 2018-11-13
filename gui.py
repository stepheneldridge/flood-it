# Stephen Eldridge
from multiprocessing import Process, Pipe
from solution_algorithm import Puzzle, runWisdomOfCrowds
from time import sleep
from tkinter import *


class Window(Frame):
    def __init__(self):  # initialized the gui
        master = Tk()
        Frame.__init__(self, master)
        master.geometry('500x500')
        master.title('Traveling Salesman')
        self.pack()
        self.createButtons()
        self.createConsole()
        self.createCanvas()
        self.addMessage("Program Loaded")

    def createCanvas(self):  # create a canvas element and clears it out
        self.canvas = Canvas(self, width=400, height=400, background="white")

        def mousePressed(event):
            print(event)
        self.canvas.bind('<Button-1>', mousePressed)
        self.canvas.grid()

    def createConsole(self):  # creates the output console with scrollbar
        scrollbar = Scrollbar(self)
        scrollbar.grid(column=1, row=2)
        self.console = Text(self, height=4, width=55, yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.console.yview)
        self.console.grid(column=0, row=2)

    def createButtons(self):  # creates the start and exit buttons
        self.startButton = Button(self, text='Start', command=self.execute)
        self.startButton.grid(row=1)
        self.quitButton = Button(self, text='Exit', command=self.quit)
        self.quitButton.grid(row=1, column=1)

    def execute(self):
        self.addMessage("Running...")
        main_pipe, child_pipe = Pipe(False)
        # creates child process to run the algorithm outside of the gui process
        process = Process(target=runWisdomOfCrowds, args=(100, 100, 5, child_pipe, False))
        process.start()
        child_pipe.close()  # closes main processes instance of the child pipe
        while True:
            try:  # runs until the child_pipe in process is closed
                puzzle = main_pipe.recv()
            except EOFError:
                break
            self.addMessage(puzzle)
            if isinstance(puzzle, list):
                continue
            elif not isinstance(puzzle, Puzzle):
                break
            self.canvas.delete("all")  # clears canvas to rerender
            size = min(400 / len(puzzle.grid), 400 / len(puzzle.grid[0]))  # adjusts tile size to fit canvas
            colors = ["red", "green", "blue", "yellow", "purple", "orange", "cyan"]  # list of colors (can add more)
            for x, column in enumerate(puzzle.grid):
                for y, color in enumerate(column):  # fills canvas with correctly colored tiles
                    self.canvas.create_rectangle(x * size, y * size, x * size + size, y * size + size, fill=colors[color])
            self.master.update()
            sleep(1)  # frame delay
        self.addMessage("Done")
        process.join()  # closes child process

    def addMessage(self, string):
        self.console.insert(END, "\n%s" % string)
        self.console.yview_moveto(1)
        self.master.update()


def init():
    app = Window()
    app.mainloop()


if __name__ == '__main__':  # starts the application and creates the window only if called from command line
    init()
