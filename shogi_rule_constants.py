import torch

from constants import *


a_left = torch.ones(K_dim, 9, 9, dtype=torch.bool)
a_left[PAWN, 0, :] = False
a_left[LANCE, 0, :] = False
a_left[KNIGHT, 0, :] = False
a_left[KNIGHT, 1, :] = False
a_left = a_left.reshape(K_dim, T_dim)


a_move = torch.zeros(K_dim, F_dim, T_dim, dtype=torch.bool)
def set_value_if(mat, k, x, y, nx, ny):
    if nx < 0 or nx >= 9 or ny < 0 or ny >= 9: return
    mat[k, x*9+y, nx*9+ny] = True
for k in HAND_PIECE_TYPES:
    a_move[k, 80, :] = True
for p_y in range(9):
    for p_x in range(9):
        set_value_if(a_move, PAWN, p_x, p_y, p_x-1, p_y)
        set_value_if(a_move, KNIGHT, p_x, p_y, p_x-2, p_y-1)
        set_value_if(a_move, KNIGHT, p_x, p_y, p_x-2, p_y+1)
        set_value_if(a_move, SILVER, p_x, p_y, p_x-1, p_y-1)
        set_value_if(a_move, SILVER, p_x, p_y, p_x-1, p_y)
        set_value_if(a_move, SILVER, p_x, p_y, p_x-1, p_y+1)
        set_value_if(a_move, SILVER, p_x, p_y, p_x+1, p_y-1)
        set_value_if(a_move, SILVER, p_x, p_y, p_x+1, p_y+1)
        set_value_if(a_move, GOLD, p_x, p_y, p_x-1, p_y-1)
        set_value_if(a_move, GOLD, p_x, p_y, p_x-1, p_y)
        set_value_if(a_move, GOLD, p_x, p_y, p_x-1, p_y+1)
        set_value_if(a_move, GOLD, p_x, p_y, p_x, p_y-1)
        set_value_if(a_move, GOLD, p_x, p_y, p_x+1, p_y)
        set_value_if(a_move, GOLD, p_x, p_y, p_x, p_y+1)
        for t_x in range(p_x):
            set_value_if(a_move, LANCE, p_x, p_y, t_x, p_y)
        for i in range(1, 9):
            set_value_if(a_move, ROOK, p_x, p_y, p_x - i, p_y)
            set_value_if(a_move, ROOK, p_x, p_y, p_x + i, p_y)
            set_value_if(a_move, ROOK, p_x, p_y, p_x, p_y - i)
            set_value_if(a_move, ROOK, p_x, p_y, p_x, p_y + i)
            set_value_if(a_move, BISHOP, p_x, p_y, p_x - i, p_y - i)
            set_value_if(a_move, BISHOP, p_x, p_y, p_x + i, p_y + i)
            set_value_if(a_move, BISHOP, p_x, p_y, p_x - i, p_y + i)
            set_value_if(a_move, BISHOP, p_x, p_y, p_x + i, p_y - i)
a_move[PROM_PAWN] = a_move[PROM_LANCE] = a_move[PROM_KNIGHT] = a_move[PROM_SILVER] = a_move[GOLD]
a_move[PROM_BISHOP] = a_move[BISHOP]
a_move[PROM_ROOK] = a_move[ROOK]
for p_y in range(9):
    for p_x in range(9):
        for k in [PROM_BISHOP, PROM_ROOK, KING]:
            set_value_if(a_move, k, p_x, p_y, p_x-1, p_y-1)
            set_value_if(a_move, k, p_x, p_y, p_x-1, p_y)
            set_value_if(a_move, k, p_x, p_y, p_x-1, p_y+1)
            set_value_if(a_move, k, p_x, p_y, p_x, p_y-1)
            set_value_if(a_move, k, p_x, p_y, p_x, p_y+1)
            set_value_if(a_move, k, p_x, p_y, p_x+1, p_y-1)
            set_value_if(a_move, k, p_x, p_y, p_x+1, p_y)
            set_value_if(a_move, k, p_x, p_y, p_x+1, p_y+1)
a_oppo_move = torch.flip(a_move[:k_dim, :, :], [1, 2])


a_prom = torch.zeros(Pr_dim, K_dim, K_dim, dtype=torch.bool)
for k in range(k_dim):
    a_prom[0, k, k] = True
for k in HAND_PIECE_TYPES:
    a_prom[0, k, k-k_dim] = True
a_prom[1, PAWN, PROM_PAWN] = True
a_prom[1, LANCE, PROM_LANCE] = True
a_prom[1, KNIGHT, PROM_KNIGHT] = True
a_prom[1, SILVER, PROM_SILVER] = True
a_prom[1, BISHOP, PROM_BISHOP] = True
a_prom[1, ROOK, PROM_ROOK] = True

a_prom_able = torch.zeros(Pr_dim, 9, 9, 9, 9, dtype=torch.bool)
a_prom_able[0, :, :, :, :] = True
for p_x in range(3):
    a_prom_able[1, p_x, :, :, :] = True
    a_prom_able[1, :, :, p_x, :] = True
a_prom_able = a_prom_able.reshape(Pr_dim, F_dim, T_dim)


a_bougai = torch.zeros(K_dim, F_dim, T_dim, P_dim, dtype=torch.bool)
a_nifu = torch.zeros(k_dim, P_dim, K_dim, T_dim, dtype=torch.bool)
for p_x in range(9):
    for p_y in range(9):
        for t_x in range(9):
            a_nifu[PAWN,  p_x*9+p_y, HAND_PAWN, t_x*9+p_y] = True
        for i in range(1, 9):
            f_x = p_x - i
            f_y = p_y - i
            if f_x >= 0 and f_y >= 0:
                for j in range(1, 9):
                    t_x = p_x + j
                    t_y = p_y + j
                    if t_x >= 9 or t_y >= 9: break
                    a_bougai[BISHOP, f_x*9+f_y, t_x*9+t_y, p_x*9+p_y] = True
                    a_bougai[BISHOP, t_x*9+t_y, f_x*9+f_y, p_x*9+p_y] = True
            f_x = p_x - i
            f_y = p_y + i
            if f_x >= 0 and f_y < 9:
                for j in range(1, 9):
                    t_x = p_x + j
                    t_y = p_y - j
                    if t_x >= 9 or t_y < 0: break
                    a_bougai[BISHOP, f_x*9+f_y, t_x*9+t_y, p_x*9+p_y] = True
                    a_bougai[BISHOP, t_x*9+t_y, f_x*9+f_y, p_x*9+p_y] = True
        for f_x in range(p_x):
            for t_x in range(p_x+1, 9):
                a_bougai[ROOK, f_x*9+p_y, t_x*9+p_y, p_x*9+p_y] = True
                a_bougai[ROOK, t_x*9+p_y, f_x*9+p_y, p_x*9+p_y] = True
                a_bougai[LANCE, t_x*9+p_y, f_x*9+p_y, p_x*9+p_y] = True
        for f_y in range(p_y):
            for t_y in range(p_y+1, 9):
                pass
                a_bougai[ROOK, p_x*9+f_y, p_x*9+t_y, p_x*9+p_y] = True
                a_bougai[ROOK, p_x*9+t_y, p_x*9+f_y, p_x*9+p_y] = True
a_bougai[PROM_ROOK, :, :, :] = a_bougai[ROOK, :, :, :]
a_bougai[PROM_BISHOP, :, :, :] = a_bougai[BISHOP, :, :, :]
a_jibougai = torch.eye(T_dim, dtype=torch.bool)
a_uchibougai = torch.zeros(K_dim, T_dim, P_dim, dtype=torch.bool)
for k in HAND_PIECE_TYPES:
    for t in range(T_dim):
        a_uchibougai[k, t, t] = True


a_use_piece = torch.zeros(K_dim, K_dim-k_dim, F_dim, T_dim, dtype=torch.bool)
a_take_piece = torch.zeros(K_dim-k_dim, K_dim-k_dim, F_dim, T_dim, dtype=torch.bool)
a_captured = torch.zeros(k_dim, K_dim-k_dim, dtype=torch.bool)
for k in range(k_dim):
    a_use_piece[k, :] = torch.eye(F_dim, dtype=torch.bool)
for k in HAND_PIECE_TYPES:
    for k_p in range(K_dim-k_dim):
        if k-k_dim != k_p:
            a_use_piece[k, k_p] = torch.eye(F_dim, dtype=torch.bool)
        else:
            for p in range(1, T_dim):
                a_use_piece[k, k_p, p-1, p] = True
for k in range(k_dim):
    k_t = k-len(PIECE_TYPES) if k in PROM_PIECE_TYPES else k
    if k_t != KING: a_captured[k, k_t] = True
for k_t in range(K_dim-k_dim):
    for k_p in range(K_dim-k_dim):
        if k_t != k_p:
            a_take_piece[k_t, k_p] = torch.eye(F_dim, dtype=torch.bool)
        else:
            for p in range(1, T_dim):
                a_take_piece[k_p, k_p, p, p-1] = True