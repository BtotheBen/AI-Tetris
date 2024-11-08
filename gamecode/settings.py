import pygame 

# Game Size 
COLUMNS = 8
ROWS = 15
CELL_SIZE = 40
GAME_WIDTH, GAME_HEIGHT = COLUMNS * CELL_SIZE, ROWS * CELL_SIZE

# Window
PADDING = 20
WINDOW_WIDTH = PADDING + GAME_WIDTH + PADDING + 10*PADDING + PADDING
WINDOW_HEIGHT = PADDING + GAME_HEIGHT + PADDING

# Game Data 
FPS = 60
UPDATE_START_SPEED = 300
MOVE_WAIT_TIME = 100
ROTATE_WAIT_TIME = 200
BLOCK_OFFSET = pygame.Vector2(COLUMNS // 2, -1)

# shapes
TETS = {
	'T': {'shape': [(0,0), (-1,0), (1,0), (0,-1)], 'color': "purple"},
	'O': {'shape': [(0,0), (0,-1), (1,0), (1,-1)], 'color': "yellow"},
	'J': {'shape': [(0,0), (0,-1), (0,1), (-1,1)], 'color': "blue"},
	'L': {'shape': [(0,0), (0,-1), (0,1), (1,1)], 'color': "orange"},
	'I': {'shape': [(0,0), (0,-1), (0,-2), (0,1)], 'color': "cyan"},
	'S': {'shape': [(0,0), (-1,0), (0,-1), (1,-1)], 'color': "green"},
	'Z': {'shape': [(0,0), (1,0), (0,-1), (-1,-1)], 'color': "red"}
}

SCORE_DATA = {1: 40, 2: 100, 3: 300, 4: 1200}