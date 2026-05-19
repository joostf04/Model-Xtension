T = 500
P = 200
S = 20
R_TRUE = 3
R_MAX = 10

TAU_GRID     = [0.5, 0.7]
OVERLAP_GRID = [0.0, 0.25, 0.5, 0.75, 1.0]
SNR_GRID     = [2, 4]
# Simulation grid: 4 values → 2×5×2×4 = 80 cells (proposal Section 4.3).
# The empirical section uses a 5-value grid with an additional δ=0.98 for
# finer resolution near 1; see DELTA_GRID_EMPIRICAL below.
DELTA_GRID          = [0.95, 0.97, 0.99, 1.00]
DELTA_GRID_EMPIRICAL = [0.95, 0.97, 0.98, 0.99, 1.00]

H_FRAC  = 0.5
H_GRID  = [0.3, 0.5, 0.7]

Q           = 0.20
ALPHA_OCMT  = 0.05
DELTA_OCMT  = 1.0
N_KNOCKOFFS = 100
M           = 500
