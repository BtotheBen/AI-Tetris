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

        self.current_frame = 0
    
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
            if int(block.pos.y) >= 0:
                new_map[int(block.pos.y)][int(block.pos.x)] = 1
        
        self.print_map(new_map)
        print(self.next_shape)

        return new_map, self.next_shape, self.score.score

    def run(self, action = 0):
        #Refresh Screen 
        self.display_surface.fill("black")

        #Run Objects
        self.game.run(action, self.current_frame)
        self.current_frame += 1
        if self.current_frame == FRAME_SPEED * 100:
            self.current_frame = FRAME_SPEED
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

                #TESTING FOR RUNNING BY ACTION
                keys = pygame.key.get_pressed()
                if keys[pygame.K_DOWN]:
                    self.run(0)
                elif keys[pygame.K_LEFT]:
                    self.run(1)
                elif keys[pygame.K_RIGHT]:
                    self.run(2)
                elif keys[pygame.K_UP]:
                    self.run(3)

            #Basic Update
            pygame.display.update()
            self.clock.tick()

if __name__ == '__main__':
    main = Main()
    main.start()