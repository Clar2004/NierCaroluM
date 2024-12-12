class GameState:
    def __init__(self):
        self.isPlayDemo = False
        self.isGameStart = False
        
        #Image Matching game
        self.isCountDownStart = False
        self.isDrawingStart = False
        self.isCountDownEnd = False
        self.targetImageIndex = None
        self.isSendAccuracy = False
        self.match_accuracy = None
        self.current_seconds = 0
        
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
        
game_state = GameState()