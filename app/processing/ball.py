import math

class RedBall:
    def __init__(self, spawn_x, spawn_y, boss_x, boss_y, speed):
        self.x = spawn_x
        self.y = spawn_y

        self.direction_x = spawn_x - boss_x - 400
        self.direction_y = spawn_y - boss_y - 400
        
        length = math.sqrt(self.direction_x**2 + self.direction_y**2)
        self.direction_x /= length
        self.direction_y /= length
        self.speed = speed

    def move(self):
        self.x += self.direction_x * self.speed
        self.y += self.direction_y * self.speed

    def is_offscreen(self, width, height):
        return self.x < 0 or self.x > width or self.y < 0 or self.y > height