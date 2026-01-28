#NOTE This simulation is incomplete and is still being worked on.


import pygame
import random
from settings import width, height, cell_size, grid, events
from Particle import Particle, active_cells

pygame.init()
screen = pygame.display.set_mode((width, height))

paused = False
running = True
clock = pygame.time.Clock()

def update_grid():
    for cell in active_cells:
        if hasattr(cell, "life"):
            if cell.check_life() == False:
                if hasattr(cell, 'product'):

                    grid[cell.row][cell.col] = cell.product(cell.row, cell.col)
                    active_cells.remove(cell)
                    active_cells.append(grid[cell.row][cell.col])
                else:
                    grid[cell.row][cell.col] = 0
                    active_cells.remove(cell)
                continue

        if cell.heat_transfer() == False:
            continue
        cell.check_state() 
        cell.move()





    for event in events.values():

        if len(event) == 1:
            cell = event[0]
            cell.swap_places(cell.target[0], cell.target[1])
        else:
            selection  = random.randint(0, len(event)+1)
            if selection:
                cell = event[0]
            else:
                cell = event[1]
            cell.swap_places(cell.target[0], cell.target[1])

def draw_grid():
    """This goes through all the active cells and draws them"""
    for cell in active_cells:
        #screen.blit(cell.image, (cell.x, cell.y))
        pygame.draw.rect(screen, (cell.color), [cell.col*cell_size, cell.row*cell_size, cell_size, cell_size], 0)


class Sand(Particle):
    def __init__(self, row, col):
        super().__init__(row, col)
        self.temp = 22
        self.phase = 'Solid'
        self.melting_point = 1700
        self.vaporization_point = 100000
        self.density = 5
        self.motion = 'Moveable'
        self.color = random.choice([(233,233,22), (220,220,24), (199,199,35)])

class Rock(Particle):
    def __init__(self, row, col):
        super().__init__(row, col)
        self.temp = 10
        self.phase = 'Solid'
        self.melting_point = 1200
        self.vaporization_point = 10000
        self.density = 5
        self.motion = 'Moveable'
        self.color = random.choice([(96,96,96), (144,144,144), (112,112,112)])

class Iron(Particle):
    def __init__(self, row, col):
        super().__init__(row, col)
        self.temp = 22
        self.phase = 'Solid'
        self.melting_point = 2000
        self.vaporization_point = 10000
        #self.image = pygame.image.load()
        self.density = 5
        self.motion = 'Still'
        self.color = (122,124,144)

class Water(Particle):
    def __init__(self, row, col):
        super().__init__(row, col)
        self.temp = 20
        self.phase = 'Liquid'
        self.melting_point = 0
        self.vaporization_point = 100
        #self.image = pygame.image.load()
        self.density = 1
        self.motion = 'Moveable'
        self.color = (15, 82 ,172)
        self.color_ramp = [(167,172,245),(15, 82 ,172), (223, 224, 235)]

class Fire(Particle):
    def __init__(self, row, col):
        super().__init__(row, col)
        self.temp = 700
        self.phase = 'Gas'
        self.life = random.randint(5, 10)
        self.product = Smoke
        #self.image = pygame.image.load()
        self.density = 1.3
        self.motion = 'Moveable'
        self.color = random.choice([(128,17,0), (182,34,3), (215,53,2), (252,100,0)])

class Smoke(Particle):
    def __init__(self, row, col):
        super().__init__(row, col)
        self.temp = 180
        self.phase = 'Gas'
        self.life = random.randint(1, 10)
        #self.image = pygame.image.load()
        self.density = 0.3
        self.motion = 'Moveable'
        self.color = (112, 140, 152)

class Thermite(Particle):
    def __init__(self, row, col):
        super().__init__(row, col)
        self.temp = 22
        self.ignited = False
        self.melting_point = 300
        self.reaction_temp = 4000
        self.phase = 'Solid'
        self.flamable = True
        #self.image = pygame.image.load()
        self.density = 1.3
        self.motion = 'Moveable'
        self.color = (127, 128, 129)

class Liquid_Nitrogen(Particle):
    def __init__(self, row, col):
        super().__init__(row, col)
        self.temp = -330
        self.phase = 'Liquid'
        self.melting_point = -346
        self.vaporization_point = -196
        self.vapor_type = 'Disapate'
        self.dicipate = False
        #self.image = pygame.image.load()
        self.density = 1
        self.motion = 'Moveable'
        self.color = (15, 82 ,172)


selected_font = pygame.font.SysFont('Garamond', 30)
temp_font = pygame.font.SysFont('Garamond', 30)

def mouse_click():
    x, y = pygame.mouse.get_pos()
    col, row = round(x//cell_size), round(y//cell_size)
    if erase:
        if isinstance(grid[row][col], Particle):
            cell = grid[row][col]
            grid[row][col] = 0
            active_cells.remove(cell)
    else:
        if not isinstance(grid[row][col], Particle) and change == False:
            cell = active_element(row, col)
            grid[row][col] = cell
            active_cells.append(cell)
        elif isinstance(grid[row][col], Particle) and change:
            cell = grid[row][col]
            cell.temp += 100



active_element = Sand
erase = False
change = False

dragging = False

while running:
    x, y = pygame.mouse.get_pos()
    col, row = round(x//cell_size), round(y//cell_size)
    selected_particle_text = selected_font.render(f'Current Particle: {active_element.__name__}', False, (200, 200, 200))

    textRect = selected_particle_text.get_rect()
    if dragging:
        mouse_click()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                paused = not paused
            elif event.key == pygame.K_1:
                active_element = Sand
            elif event.key == pygame.K_2:
                active_element = Rock
            elif event.key == pygame.K_3:
                active_element = Water
            elif event.key == pygame.K_4:
                active_element = Iron
            elif event.key == pygame.K_5:
                active_element = Fire
            elif event.key == pygame.K_6:
                active_element = Thermite
            elif event.key == pygame.K_7:
                active_element = Liquid_Nitrogen
                #change = not change

        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                dragging = True
            elif event.button == 3:
                erase = True
                dragging = True
        if event.type == pygame.MOUSEBUTTONUP:
            dragging = False
            erase = False
    
    screen.fill((0,0,0))
    draw_grid()
    if not paused:
        events.clear()
        update_grid()
    clock.tick(60)

    screen.blit(selected_particle_text, (0, 0))
    if isinstance(grid[row][col], Particle):
        temp_particle_text = temp_font.render(f'Temp: {grid[row][col].temp} C', False, (200, 200, 200))
        tempRect = temp_particle_text.get_rect()
        screen.blit(temp_particle_text, (0, 20))
    pygame.display.set_caption(f"Falling Sand Test V2 - FPS: {int(clock.get_fps())}        Objects: {len(active_cells)}")
    pygame.display.update()
