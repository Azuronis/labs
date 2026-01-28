cell_size = 10
width, height = 1200, 800
grid_height, grid_width = height//cell_size, width//cell_size
grid = [[0 for _ in range(grid_width)] for _ in range(grid_height)]
events = {}
