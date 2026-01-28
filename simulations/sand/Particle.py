from settings import cell_size, grid_height, grid_width, grid, events
from Rigid import Solid
from NonRigid import Liquid, Gas

active_cells = []

class Particle: 
    def __init__(self, row, col):
        self.row = row
        self.col = col
        self.x = self.col*cell_size
        self.y = self.row*cell_size
        self.updated = False
        self.target = None
        self.dispersion = None

    def move(self):
        if self.phase == 'Solid':
            Solid.move(self)
        elif self.phase == 'Liquid':
            Liquid.move(self)
        elif self.phase == 'Gas':
            Gas.move(self)

    def check_life(self):
        self.life -=1
        if self.life > 0:
            pass
        else:
            return False
    

    def update_positions(self):
        self.x = self.col*cell_size
        self.y = self.row*cell_size

    def check_position(self, row, col, check_cell = False):
        if (row < grid_height and row >= 0) and (col < grid_width and col >= 0):
            if isinstance(grid[row][col], Particle):
                if check_cell:
                    return True
                target_cell = grid[row][col]
                if (self.phase != target_cell.phase) and self.density > target_cell.density:
                    return True
            elif not isinstance(grid[row][col], Particle) and check_cell == False:
                return True

    def heat_transfer(self):
        neighbors = set()

        neighbors.add(grid[self.row][self.col])

        if self.check_position(self.row-1, self.col, check_cell=True):
            neighbors.add(grid[self.row-1][self.col])

        if self.check_position(self.row, self.col+1, check_cell=True):
            neighbors.add(grid[self.row][self.col+1])

        if self.check_position(self.row+1, self.col, check_cell=True):
            neighbors.add(grid[self.row+1][self.col])

        if self.check_position(self.row, self.col-1, check_cell=True):
            neighbors.add(grid[self.row][self.col-1])

        if len(neighbors) > 1:
            self.dicipate = False

            average = float(format(sum([cell.temp for cell in neighbors]) / len(neighbors), ".2f"))
            for neighbor in neighbors:
                neighbor.temp = average

    def check_state(self):

        if hasattr(self, 'melting_point'):
            if self.temp < self.melting_point:
                self.phase = 'Solid'
                return
            elif self.temp >= self.melting_point:
                if hasattr(self, 'reaction_temp') and self.ignited != True:
                    self.temp += self.reaction_temp
                    self.ignited = True

                self.phase = 'Liquid'
        if hasattr(self, 'vaporization_point'):
            if self.temp >= self.vaporization_point:
                self.phase = 'Gas'
                return
        
        if hasattr(self, 'color_ramp'):
            pass
            


    def swap_places(self, row, col):
        target_cell = grid[row][col]

        if isinstance(target_cell, Particle):
            target_cell.row = self.row
            target_cell.col = self.col
            target_cell.update_positions()

        grid[self.row][self.col] = target_cell
        grid[row][col] = self


        self.row = row
        self.col = col
        self.update_positions()

    def add_position(self, row, col):
        self.target = (row, col)

        if (row, col) in events.keys():
            events[(row, col)].append(self)
        else:
            events[(row, col)] = [self]
