PIECE_TYPES = [
    PAWN,
    LANCE,
    KNIGHT,
    SILVER,
    GOLD,
    BISHOP,
    ROOK,
    KING,
] = range(8)
PROM_PIECE_TYPES = [
    PROM_PAWN,
    PROM_LANCE,
    PROM_KNIGHT,
    PROM_SILVER,
    PROM_BISHOP,
    PROM_ROOK,
] = range(8, 14)
HAND_PIECE_TYPES = [
    HAND_PAWN,
    HAND_LANCE,
    HAND_KNIGHT,
    HAND_SILVER,
    HAND_GOLD,
    HAND_BISHOP,
    HAND_ROOK,
] = range(14, 21)
k_dim = len(PIECE_TYPES) + len(PROM_PIECE_TYPES)
K_dim = k_dim + len(HAND_PIECE_TYPES)

Pr_dim = 2
P_dim = 9*9
F_dim = 9*9
T_dim = 9*9