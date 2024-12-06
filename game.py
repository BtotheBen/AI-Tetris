import pygame, sys, importlib

from settings import *
from random import choice

class Game():
    def __init__(self, get_next_shape, update_score) -> None:
        self.surface = pygame.Surface((GAME_WIDTH, GAME_HEIGHT))
        self.display_surface = pygame.display.get_surface()
        self.rect = self.surface.get_rect(topleft = (PADDING, PADDING))

        self.get_next_shape = get_next_shape
        self.update_score = update_score

        self.sprites = pygame.sprite.Group()

        self.map = [[0 for x in range(COLUMNS)] for y in range(ROWS)]
<<<<<<< HEAD

        if DRAW_MAIN:
            self.tet = Tet(MAIN_TET, self.sprites, self.create_new_tet, self.map)
        else: 
            self.tet = Tet(choice(list(TETS.keys())), self.sprites, self.create_new_tet, self.map)
=======
        self.tet = Tet(choice(list(TETS.keys())), self.sprites, self.create_new_tet, self.map)
        #MAIN_TET
>>>>>>> f5d1006fe14654f50604f19b8e94d341edafcee3

        self.down_speed = UPDATE_START_SPEED
        self.down_speed_fast = UPDATE_START_SPEED * 0.3

        self.current_level = 1
        self.current_score = 0

        self.cyclenumber = 0

        self.terminated = False


    def score_update(self, num_lines):
        self.current_score += SCORE_DATA[num_lines] * self.current_level
        if self.current_score / 100 > self.current_level:
            self.current_level += 1
            #self.current_level = min(self.current_level) #Making the max level 10

            #self.down_speed = LEVEL_DATA[self.current_level]
            #self.down_speed_fast = self.down_speed * 0.3
            #self.timers['vertical move'].duration = self.down_speed
        self.update_score(self.current_score, min(self.current_level, 10))

    def check_game_over(self):
        for block in self.tet.blocks:
            if block.pos.y < 0:
                print(f"You reached the {self.current_level} level with a score of {self.current_score}!")
                self.terminated = True

    def create_new_tet(self):
        self.check_game_over()
        self.check_finished_rows()
        self.tet = Tet(self.get_next_shape(), self.sprites, self.create_new_tet, self.map)

    def move_down(self, current_frame):
        if current_frame % FRAME_SPEED == 0:
            self.tet.move_down()

    def draw_grid(self):
        for coloumn in range(1, COLUMNS):
            x = coloumn * CELL_SIZE
            pygame.draw.line(self.surface, "white", (x, 0), (x, GAME_HEIGHT), 1)
        for row in range(1, ROWS):
            y = row * CELL_SIZE
            pygame.draw.line(self.surface, "white", (0, y), (GAME_WIDTH, y), 1)

    def input(self, action):
        if action == 0:
            pass
        elif action == 1:
            self.tet.move_horizontal(-1)
        elif action == 2:
            self.tet.move_horizontal(1)
        elif action == 3:
            self.tet.rotate()

    def check_finished_rows(self):
        delete_rows = []
        for i, row in enumerate(self.map):
            if all(row):
                delete_rows.append(i)

        if delete_rows:
            for delete_row in delete_rows:
                for block in self.map[delete_row]:
                    block.kill()

                for row in self.map:
                    for block in row:
                        if block and block.pos.y < delete_row:
                            block.pos.y += 1

            self.map = [[0 for x in range(COLUMNS)] for y in range(ROWS)]
            for block in self.sprites:
                self.map[int(block.pos.y)][int(block.pos.x)] = block

            self.score_update(len(delete_rows))

    def run(self, action, current_frame, draw):
        self.sprites.update()

        self.input(action)
        self.move_down(current_frame)

        self.sprites.update()

        if draw:
            self.surface.fill("black")
            self.sprites.draw(self.surface)
            #self.draw_grid()
            self.display_surface.blit(self.surface, (PADDING, PADDING))
            pygame.draw.rect(self.display_surface, "white", self.rect, 2, 2)

        return self.terminated

class Tet():
    def __init__(self, shape, group, create_new_tet, map) -> None:
        self.shape = shape
        self.block_positions = TETS[shape]['shape']
        self.color = TETS[shape]['color']
        self.create_new_tet = create_new_tet
        self.map = map

        #create the blocks
        self.blocks = []
        for pos in self.block_positions:
            self.blocks.append(Block(group, pos, self.color))

    def rotate(self):
        if self.shape != 'O':
            pivot_pos = self.blocks[0].pos

            new_block_positions = [block.rotate(pivot_pos) for block in self.blocks]

            for pos in new_block_positions:
                if pos.y >= ROWS:
                    return
                if pos.x < 0 or pos.x >= COLUMNS:
                    return
                if self.map[int(pos.y)][int(pos.x)]:
                    return

            for i, block in enumerate(self.blocks):
                block.pos = new_block_positions[i]

    def horizontal_collide(self, amount):
        collision_list = []
        for block in self.blocks:
            collision_list.append(block.horizontal_collide(int(block.pos.x + amount), self.map))
        if any(collision_list):
            return True
        else:
            return False

    def vertical_collide(self):
        collision_list = []
        for block in self.blocks:
            collision_list.append(block.vertical_collide(int(block.pos.y + 1), self.map))
        if any(collision_list):
            return True
        else:
            return False

    def move_horizontal(self, amount):
        if not self.horizontal_collide(amount):
            for block in self.blocks:
                block.pos.x += amount

    def move_down(self):
        if not self.vertical_collide():
            for block in self.blocks:
                block.pos.y += 1
        else:
            for block in self.blocks:
                self.map[int(block.pos.y)][int(block.pos.x)] = block
            self.create_new_tet()

class Block(pygame.sprite.Sprite):

    def __init__(self, group, position, color):
        super().__init__(group)
        self.image = pygame.Surface((CELL_SIZE,CELL_SIZE))
        self.image.fill(color)

        #pos
        self.pos = pygame.Vector2(position) + BLOCK_OFFSET
        x = self.pos.x * CELL_SIZE
        y = self.pos.y * CELL_SIZE
        self.rect = self.image.get_rect(topleft = (x, y))

    def rotate(self, pivot_pos):
        return pivot_pos + (self.pos - pivot_pos).rotate(90)

    def horizontal_collide(self, x, map):
        if not COLUMNS > x >= 0:
            return True

        if map[int(self.pos.y)][x]:
            return True
        return False

    def vertical_collide(self, y, map):
        if not y < ROWS:
            return True

        if y >= 0 and map[y][int(self.pos.x)]:
            return True
        return False

    def update(self):
        self.rect.topleft = self.pos * CELL_SIZE
