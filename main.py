import pygame, random

from settings import *
from game import *
from score import *

class Main():
    def __init__(self) -> None:
        pygame.init()
        pygame.display.set_caption('TETRIS')
        self.display_surface = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        self.clock = pygame.time.Clock()

        self.next_shape = [random.choice(list(TETS.keys())) for shape in range(3)]

        self.game = Game(self.get_next_shape, self.update_score)
        self.score = Score()
    
    def get_next_shape(self):
        next_shape = self.next_shape.pop(0)
        self.next_shape.append(random.choice(list(TETS.keys())))
        return next_shape

    #debugging function
    def print_map(self, m):
        print("---------")
        for x in m:
            print(x)

    def update_score(self, score, level):
        self.score.score = score
        self.score.level = level

    def get_state(self):
        new_map = [[0 for x in range(COLUMNS)] for y in range(ROWS)]
        for row_index, row in enumerate(self.game.map):
            for colo_index, point in enumerate(row):
                if not point == 0:
                    new_map[row_index][colo_index] = 2
    
        for block in self.game.tet.blocks:
            new_map[int(block.pos.y)][int(block.pos.x)] = 1
        
        # self.print_map(new_map)

        return new_map

    def run(self, action = 0):
        #Refresh Screen 
        self.display_surface.fill("black")

        #Run Objects
        self.game.run(action)
        self.score.run()

        self.get_state()

        #Basic Update
        pygame.display.update()
        self.clock.tick()
    
    def start(self):
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit()

            #Basic Update
            pygame.display.update()
            self.clock.tick()

if __name__ == '__main__':
    main = Main()
    main.start()