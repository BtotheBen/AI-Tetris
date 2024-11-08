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

    def update_score(self, score, level):
        self.score.score = score
        self.score.level = level

    def run(self):
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit()

            #Refresh Screen 
            self.display_surface.fill("black")

            #Run Objects
            self.game.run()
            self.score.run()

            #Basic Update
            pygame.display.update()
            self.clock.tick()

if __name__ == '__main__':
    main = Main()
    main.run()