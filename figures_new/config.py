import numpy as np
PROJECT_DIR = "/Volumes/Neuro/WallisLab/rotation_EricPlanning/PlanningDynamics"

# Matplotlib settings
HUE_COL    = "subject"
PALETTE    = {"B": "#3A86FF", "L": "#FF6B6B"}
EDGE_COLOR = "#1a1a2e"
GRID_COLOR = "#e0e0e0"

# Bin definitions ── stored outside the function for easy customisation
STEPS_EXTRA_BINS = {               # label -> predicate (applied to raw column)
    "0":  lambda x: x == 0,
    "1":  lambda x: x == 1,
    "2":  lambda x: x == 2,
    "3":  lambda x: x == 3,
    "4":  lambda x: x == 4,
    "5":  lambda x: x == 5,
    ">5":  lambda x: x >5,
}

PLANNING_BINS = {                  # label -> (lo_inclusive, hi_inclusive)
    "0":     (0,  0),
    "1":     (1,  1),
    "2":     (2,  2),
    "3":     (3,  3),
    "4":     (4,  4),
    "5":     (5,  5),
    "6":     (6,  6),
    "7":     (7,  7),
    "8":     (8,  8),
    "9":     (9,  9),
    "10":    (10, 10),
    "11–15": (11, 15),
    "16–20": (16, 20),
    "≥20":   (21, np.inf),
}