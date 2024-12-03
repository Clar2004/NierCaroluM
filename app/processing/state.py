# state.py
class GameState:
    def __init__(self):
        self.isGameOneDone = False
        self.isGameTwoDone = False
        self.isGameThreeDone = False

# Global instance of the game state
game_state = GameState()

# Function to get the current game state
def get_game_state():
    return game_state
