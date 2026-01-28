import random

class Solid:
    
    def move(self):
        if self.motion == 'Moveable':
            if self.check_position(self.row+1, self.col):
                self.add_position(self.row+1, self.col)

            elif (self.check_position(self.row+1, self.col+1) and self.check_position(self.row+1, self.col-1)):
                selection = random.randint(0, 1)
                if selection:
                    self.add_position(self.row+1, self.col+1)
                else:
                    self.add_position(self.row+1, self.col-1)
            
            elif self.check_position(self.row+1, self.col+1):
                self.add_position(self.row+1, self.col+1)
            
            elif self.check_position(self.row+1, self.col-1):
                self.add_position(self.row+1, self.col-1)
        else:
            self.add_position(self.row, self.col)
