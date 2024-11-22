import pygame, random

from settings import *
from game import *
from score import *

class Main():
    def __init__(self) -> None:
        pygame.init()
        pygame.display.set_caption('TETRIS')
        self.display_surface = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))

        self.next_shape = [random.choice(list(TETS.keys())) for shape in range(3)]

        self.game = Game(self.get_next_shape, self.update_score)
        self.score = Score()

        self.current_frame = 0

        pygame.display.update()

    def get_next_shape(self):
        next_shape = self.next_shape.pop(0)
        self.next_shape.append(random.choice(list(TETS.keys())))
        return next_shape

    def update_score(self, score, level):
        self.score.score = score
        self.score.level = level

    def get_state(self):
        new_map = [0 for _ in range(ROWS*COLUMNS)]
        for row_index, row in enumerate(self.game.map):
            for colo_index, point in enumerate(row):
                if not point == 0:
                    new_map[row_index * ROWS + colo_index] = 1

        for block in self.game.tet.blocks:
            if int(block.pos.y) >= 0:
                new_map[int(block.pos.y) * ROWS + int(block.pos.x)] = 2

        for next_shape in self.next_shape:
                new_map.append(TETS[next_shape]['index'])

        return new_map

    def run(self, action = 0):
        prev_score = self.score.score

        self.display_surface.fill("black")

        terminated = self.game.run(action, self.current_frame)
        self.current_frame += 1
        if self.current_frame == FRAME_SPEED * 100:
            self.current_frame = FRAME_SPEED
        self.score.run()

        self.get_state()

        pygame.display.update()
        return 10 + self.score.score - prev_score, terminated
