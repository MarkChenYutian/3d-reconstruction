import math

def get_poses():
    diameter = 146.05

    radius = diameter/2
    x_coordinate = 0
    y_coordinate = 0
    L = []
    offset = 5
    angle = math.pi*2/8
    for i in range(8):
        centx = (x_coordinate + math.sin(angle*i))*radius
        centy = (y_coordinate + math.cos(angle*i))*radius
        x_offset = offset * math.cos(angle*i)
        y_offset = offset * math.sin(angle*i)
        x = centx + x_offset
        y = centy + y_offset
        L.append((x, y))
        print(f"x: {x}, y: {y}")

    return L

