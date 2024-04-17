# field dimensions
FIELD_X_RANGE = 30
FIELD_Y_RANGE = 30

# move-to-target: target position
# constant target position
# TARGET_X = 20
# TARGET_Y = 20

# battle control
PLAYER_START = 0
ROUNDS_PER_BATTLE = 4
STATE_SIZE = 4

# movement
VISGRAPH_WORKERS = 1

# model
SCALER_INIT_ITERATIONS = 1024
BATCH_SIZE = 32
BUFFER_SIZE = 512
N_HIDDEN_LAYERS = 2
HIDDEN_DIM = 16

# rewards
REW_DIST_MULT = -0.1
REW_FIELD_EDGE = -10
REW_REACHED_TARGET = 20

# control
PRINT_EVERY_NTH_BATTLE = 20

# paths
MODELS_FOLDER = "data/digger_models"
STATS_FOLDER = "data/stats"
FLD_IMG_RESULTS = 'img/results'
