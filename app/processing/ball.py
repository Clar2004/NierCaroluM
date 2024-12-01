import math

class RedBall:
    def __init__(self, spawn_x, spawn_y, boss_x, boss_y, speed):
        self.x = spawn_x
        self.y = spawn_y
        # Calculate the direction vector (from boss to spawn position)
        self.direction_x = spawn_x - boss_x - 400  # Direction X (outward from boss)
        self.direction_y = spawn_y - boss_y - 400  # Direction Y (outward from boss)
        
        # Normalize the direction vector so the ball moves at a constant speed
        length = math.sqrt(self.direction_x**2 + self.direction_y**2)
        self.direction_x /= length  # Normalize X direction
        self.direction_y /= length  # Normalize Y direction
        self.speed = speed

    def move(self):
        """Move the red ball away from the boss, based on its direction vector and speed."""
        self.x += self.direction_x * self.speed
        self.y += self.direction_y * self.speed

    def is_offscreen(self, width, height):
        """Check if the red ball is offscreen."""
        return self.x < 0 or self.x > width or self.y < 0 or self.y > height