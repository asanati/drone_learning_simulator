import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np

pygame.init()
font = pygame.font.SysFont('arial', 15)

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

Point = namedtuple('Point', 'x, y')

# rgb colors
WHITE = (255, 255, 255)
RED = (200, 0, 0)
GREEN = (0, 200, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)
GREY = (100, 100, 100)

BLOCK_SIZE = 20
SPEED = 1

class DroneEnvironment():
    def __init__(self, width=640, height=480, num_obstacles=5):
        self.width = width
        self.height = height
        
        # init display
        self.display = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption('Drone Visualizer')
        self.clock = pygame.time.Clock()
        self.num_obstacles = num_obstacles

        self.reset()
    
    def reset(self):
        # init game state
        self.direction = Direction.RIGHT

        self.midpoint = Point(self.width/2, self.height/2)
        self.gripper = Point(self.midpoint.x, self.midpoint.y+2*BLOCK_SIZE)
        self.drone = None
        self._update_drone_pos()

        self.score = 0

        self.obstacles = None
        self._place_obstacles()

        self.objects = None
        self._place_object()
        self.frame_iteration = 0

    def _update_drone_pos(self):
        self.gripper = Point(self.midpoint.x, self.midpoint.y+2*BLOCK_SIZE)
        self.drone = [self.midpoint, Point(self.midpoint.x-BLOCK_SIZE, self.midpoint.y), Point(self.midpoint.x-2*BLOCK_SIZE, self.midpoint.y),
                                     Point(self.midpoint.x+BLOCK_SIZE, self.midpoint.y), Point(self.midpoint.x+2*BLOCK_SIZE, self.midpoint.y),
                                     Point(self.midpoint.x-BLOCK_SIZE, self.midpoint.y-BLOCK_SIZE), Point(self.midpoint.x+BLOCK_SIZE, self.midpoint.y-BLOCK_SIZE),
                                     Point(self.midpoint.x-2*BLOCK_SIZE, self.midpoint.y-BLOCK_SIZE), Point(self.midpoint.x+2*BLOCK_SIZE, self.midpoint.y-BLOCK_SIZE),
                                     Point(self.midpoint.x-3*BLOCK_SIZE, self.midpoint.y-BLOCK_SIZE), Point(self.midpoint.x+3*BLOCK_SIZE, self.midpoint.y-BLOCK_SIZE),
                                     Point(self.midpoint.x, self.midpoint.y+BLOCK_SIZE), self.gripper]
        
    
    def _place_object(self):
        x = random.randint(3, (self.width-4*BLOCK_SIZE)//BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(3, (self.height-BLOCK_SIZE)//BLOCK_SIZE) * BLOCK_SIZE
        # y = self.height-BLOCK_SIZE
        self.object = Point(x, y)
        if self.object in self.drone or self.object in self.obstacles:
            self._place_object()
    
    def _place_obstacles(self):
        obstacles = []
        idx = 0
        while idx < self.num_obstacles:
            x = random.randint(0, (self.width-BLOCK_SIZE)//BLOCK_SIZE) * BLOCK_SIZE
            y = random.randint(0, (self.height-BLOCK_SIZE)//BLOCK_SIZE) * BLOCK_SIZE
            obstacle = Point(x, y)
            if obstacle not in self.drone and obstacle not in obstacles:
                obstacles.append(obstacle)
                idx += 1
        self.obstacles = obstacles


    def _update_ui(self):
        self.display.fill(WHITE)

        for pt in self.drone:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x+4, pt.y+4, 12, 12))
        
        for pt in self.obstacles:
            pygame.draw.rect(self.display, GREY, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
        
        pygame.draw.rect(self.display, GREEN, pygame.Rect(self.object.x, self.object.y, BLOCK_SIZE, BLOCK_SIZE))

        text = font.render("Score: " + str(self.score), True, BLACK)
        self.display.blit(text, [0, 0])
        pygame.display.flip()

    def play_step(self, action=None):
        self.frame_iteration += 1
        # 1. collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            if action is None and event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    self.direction = Direction.LEFT
                elif event.key == pygame.K_RIGHT:
                    self.direction = Direction.RIGHT
                elif event.key == pygame.K_UP:
                    self.direction = Direction.UP
                elif event.key == pygame.K_DOWN:
                    self.direction = Direction.DOWN
        # # 2. move
        self._move(action) # update the head
        self._update_drone_pos()
        # self.drone.insert(0, self.midpoint)

        # # 3. check if game over
        reward = 0
        game_over = False
        if self.is_collision():
            game_over = True
            reward = -50
            return reward, game_over, self.score
        if self.frame_iteration > 100 * (self.score+3):
            game_over = True
            reward = -30
            return reward, game_over, self.score
        # 4. place new food or just move
        if self.gripper == self.object:
            self.score += 1
            reward = 100
            self._place_object()
        # else:
        #     self.drone.pop()
        
        # 5. update ui and clock
        self._update_ui()
        self.clock.tick(SPEED)

        # 6. return game over and score
        return reward, game_over, self.score
    
    def is_collision(self):
        # hits boundary
        for pt in self.drone:
            if pt.x > self.width - BLOCK_SIZE or pt.x < 0 or pt.y > self.height - BLOCK_SIZE or pt.y < 0:
                return True
            # hits itself
            # if self.midpoint in self.drone[1:]:
            #     return True
            # hits an obstacle
            if pt in self.obstacles:
                return True
        
        return False
    
    def collides(self, point):
        if point.x > self.width - BLOCK_SIZE or point.x < 0 or point.y > self.height - BLOCK_SIZE or point.y < 0:
            return True
            # hits itself
            # if self.midpoint in self.drone[1:]:
            #     return True
            # hits an obstacle
        if point in self.obstacles:
            return True
        
        return False

    def _move(self, action):
        if action is not None:
            clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
            idx = clock_wise.index(self.direction)

            if np.array_equal(action, [1, 0, 0, 0]):
                new_dir = clock_wise[idx] # no change
            elif np.array_equal(action, [0, 1, 0, 0]):
                next_idx = (idx + 1) % 4
                new_dir = clock_wise[next_idx]
            elif np.array_equal(action, [0, 0, 1, 0]):
                next_idx = (idx + 2) % 4
                new_dir = clock_wise[next_idx]
            else:
                next_idx = (idx + 3) % 4
                new_dir = clock_wise[next_idx]
            
            self.direction = new_dir

        x = self.midpoint.x
        y = self.midpoint.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        
        self.midpoint = Point(x, y)

def test_placement():
    width = 640
    height = 480
    display = pygame.display.set_mode((width, height))
    pygame.display.set_caption('Test Visualizer')
    clock = pygame.time.Clock()
    display.fill(WHITE)
    pygame.display.flip()
    for _ in range(1000):
        x = random.randint(3, (width-4*BLOCK_SIZE)//BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(3, (height-BLOCK_SIZE)//BLOCK_SIZE) * BLOCK_SIZE
        # y = self.height-BLOCK_SIZE
        obj = Point(x, y)
        pygame.draw.rect(display, GREEN, pygame.Rect(obj.x, obj.y, BLOCK_SIZE, BLOCK_SIZE))
        pygame.display.flip()
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
    


if __name__ == '__main__':
    # test_placement()
    game = DroneEnvironment()

    # game loop
    while True:
        reward, game_over, score = game.play_step()

        # break if game over
        if game_over:
            break
    
    print('Final Score:', score)
    
    pygame.quit()