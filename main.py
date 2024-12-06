import pygame, random

from settings import *
from game import *
from score import *
import os


class Main():
    def __init__(self) -> None:
        pygame.init()
        pygame.display.set_caption('TETRIS')
        self.display_surface = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))

        self.next_shape = [random.choice(list(TETS.keys())) for shape in range(3)] #MAIN_TET

        self.game = Game(self.get_next_shape, self.update_score)
        self.score = Score()

        self.draw = False
        
        self.current_frame = 0

        if self.draw:
            pygame.display.update()

    def get_next_shape(self):
        next_shape = self.next_shape.pop(0)
        self.next_shape.append(random.choice(list(TETS.keys()))) #MAIN_TET

        return next_shape

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
        
        #self.print_map(new_map)
        #print(self.next_shape)

        next_shapes_index = []
        for next_shape in self.next_shape:
            next_shapes_index.append(TETS[next_shape]['index'])
        #print(next_shapes_index)

        ret = []
        for _ in new_map: ret.extend(_)
        ret.extend(next_shapes_index)
        ret.append(self.score.score)

        return ret


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

        new_map.append(self.score.score)

        return new_map

    def calc_free_spots_reward(self):
        new_map = [[0 for x in range(COLUMNS)] for y in range(ROWS)]
        for row_index, row in enumerate(self.game.map):
            for colo_index, point in enumerate(row):
                if not point == 0:
                    new_map[row_index][colo_index] = 2

        for x in range(COLUMNS):
            for y in range(ROWS):
                if new_map[y][x] == 0:
                    new_map[y][x] = 3
                if new_map[y][x] == 2:
                    break
                    
        ret = 0

        for row in new_map:
            for point in row:
                if point == 0:
                    ret -= 0.35
        
        return ret

    def calc_height_block_reward(self) -> int:
        new_map = [[0 for x in range(COLUMNS)] for y in range(ROWS)]
        for row_index, row in enumerate(self.game.map):
            for colo_index, point in enumerate(row):
                if not point == 0:
                    new_map[row_index][colo_index] = 2
        
        highest = []
        for x in range(COLUMNS):
            for y in range(ROWS):
                if new_map[y][x] == 2:
                    highest.append(ROWS-y)
                    break
                if y == (ROWS-1):
                    highest.append(0)
        
        ret = 0
        for height in highest:
            ret -= 0.5 * height

        return ret
    
        '''
        new_map = [[0 for x in range(COLUMNS)] for y in range(ROWS)]
        for row_index, row in enumerate(self.game.map):
            for colo_index, point in enumerate(row):
                if not point == 0:
                    new_map[row_index][colo_index] = 2
        
        ret = 0

        for x in range(COLUMNS):
            for y in range(ROWS//2):
                if new_map[y][x] == 2:
                    
                    ret -= 2

        return ret
        '''

    def bumpiness_reward(self) -> int:
        new_map = [[0 for x in range(COLUMNS)] for y in range(ROWS)]
        for row_index, row in enumerate(self.game.map):
            for colo_index, point in enumerate(row):
                if not point == 0:
                    new_map[row_index][colo_index] = 2
        
        highest = []
        for x in range(COLUMNS):
            for y in range(ROWS):
                if new_map[y][x] == 2:
                    highest.append(ROWS-y)
                    break
                if y == (ROWS-1):
                    highest.append(0)
        
        ret = 0
        for i in range(len(highest)-1):
            ret -= 0.18 * abs(highest[i+1] - highest[i])

        return ret

    def run(self, action = 0):
        prev_score = self.score.score

        self.display_surface.fill("black")

        terminated = self.game.run(action, self.current_frame, self.draw)
        self.current_frame += 1
        if self.current_frame == FRAME_SPEED * 100:
            self.current_frame = FRAME_SPEED
        
        self.score.run()

        self.get_state()

        if self.draw:
            pygame.display.update()

        #Calculate Reward
        reward = 1
        reward += ((self.score.score - prev_score)/40)**2 * COLUMNS 

        reward += self.bumpiness_reward()
        reward += self.calc_height_block_reward()
        reward += self.calc_free_spots_reward()

        if terminated:
            reward -= 20

        os.system('clear')
        print(reward)
        return reward, terminated
