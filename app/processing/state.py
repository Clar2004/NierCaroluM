class GameState:
    def __init__(self):
        self.isPlayDemo = False
        self.isGameStart = False
        
        #Image Matching game
        self.isCountDownStart = False
        self.isDrawingStart = False
        self.isCountDownEnd = False
        self.targetImageIndex = 0
        self.isSendAccuracy = False
        self.match_accuracy = None
        self.current_seconds = 0
        self.current_seconds_2 = 0
        self.current_seconds_3 = 0
        self.isStartTriggered = False
        
        #Combat game
        self.isReset = False
        self.is_game_one_done = False
        self.is_mini_game_one_done = False
        self.is_game_two_done = False
        self.is_mini_game_two_done = False
        self.is_game_three_done = False
        self.is_mini_game_three_done = False
        self.is_game_four_done = False
        self.is_mini_game_four_done = False
        
        self.is_cheat = False
        self.is_update = False
        self.is_dead = False
        self.current_health = 3
        
game_state = GameState()