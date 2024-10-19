import torch
import shogi
import numpy
import shogi.KIF


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
    
from io_utils import *
    
def my_einsum(eq, *args):
    return torch.from_numpy(numpy.einsum(eq, *args) > 0)

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
    
def invert_order(board_black, hand_black, board_white, hand_white):
    return rotate_180deg(board_white), hand_white, rotate_180deg(board_black), hand_black

def apply_action_mat(board_black, hand_black, board_white, hand_white, A):
    # optimizations for tensor A by splitting it into tensors with lower dim
    BT = my_einsum("BxyTz->BT", A)
    BKF = my_einsum("BKFxy->BKF", A)
    
    S_black = torch.cat((board_black, hand_black), dim=1)
    after_removing = S_black ^ (S_black & BKF)
    after_removing_board_black, after_removing_hand_black = torch.split(after_removing, [k_dim, K_dim-k_dim], dim=1)
    after_removing_hand_black = my_einsum("BkF,KkFT,BKx->BkT", after_removing_hand_black, a_use_piece, BKF)
    #print(after_removing_hand_black.to_sparse())
    
    new_board_black = after_removing_board_black + my_einsum("BkF,BkFTp,pkK->BKT", S_black, A, a_prom[:,:,:k_dim])
    captured_piece = my_einsum("BKT,BT,Kk->Bk", board_white, BT, a_captured)
    if_captured = my_einsum("Bk->B", captured_piece)
    new_hand_black_if_captured = my_einsum("BKF,Bk,kKFT->BKT", after_removing_hand_black, captured_piece, a_take_piece) + torch.cat((torch.zeros(captured_piece.shape[0], captured_piece.shape[1], 80, dtype=torch.bool), captured_piece.unsqueeze(2)), dim=2)
    new_hand_black = ~if_captured * after_removing_hand_black + if_captured * new_hand_black_if_captured
    new_board_white = board_white ^ my_einsum("BKT,BT->BKT", board_white, BT)
    #print(new_hand_black.to_sparse())
    
    #TODO: uchifu-tsumi, open_oute
    #bougai_after = my_einsum("BKP,kFTP->BkFT", new_board_black + new_board_white, a_bougai)
    
    return invert_order(new_board_black, new_hand_black, new_board_white, hand_white)
    

def calc_legal_moves_mat(board_black, hand_black, board_white):
    S_black = torch.cat((board_black, hand_black), dim=1)
    
    ok_to_move = my_einsum("BKF,KFT,pKk,pFT,kT->BKFTp", S_black, a_move, a_prom, a_prom_able, a_left)
    
    uchibougai = my_einsum("BKP,kTP->BkT", board_black + board_white, a_uchibougai)
    nifu = my_einsum("BKP,KPkT->BkT", board_black, a_nifu)
    bougai = my_einsum("BKP,kFTP->BkFT", board_black + board_white, a_bougai)
    jibougai = my_einsum("BKP,TP->BT", board_black, a_jibougai)
    oute = torch.zeros(board_black.shape[0], K_dim, T_dim, dtype=torch.bool)
    # bug: update bougai by removing black KING
    oute[:, KING, :] = my_einsum("Bkp,kpT,BkpT->BT", board_white, a_oppo_move, ~bougai[:,:k_dim,:,:])
    total_forbidden = (jibougai.unsqueeze(1) + uchibougai + oute + nifu).unsqueeze(2) + bougai
    ok_to_go = ~total_forbidden

    legal_moves_mat = my_einsum("BKFTp,BKFT->BKFTp", ok_to_move, ok_to_go)

    return legal_moves_mat
    
def test_legal_moves(board, mat, is_white):
    ok_mat = calc_legal_moves_mat(mat[0], mat[1], mat[2])

    my = sorted(action_mat_2_usi(ok_mat, is_white)[0])
    std = board.legal_moves
    diff = [_ for _ in std if _.usi() not in set(my)]
    diff2 = [m for m in my if m not in set(_.usi() for _ in std)]
    if len(diff) != 0 or len(diff2) != 0:
        print(len(my), len(std), my)
        print(sorted([_.usi()+board.piece_at(_.from_square).japanese_symbol() if _.from_square is not None else (_.usi()+shogi.PIECE_JAPANESE_SYMBOLS[_.drop_piece_type]) for _ in diff]))
        print(sorted(diff2))
        return False
    return True

    
kif = shogi.KIF.Parser.parse_file("my.kif")[0]['moves']
board = shogi.Board()
board_black, hand_black, board_white, hand_white = board_2_mat(board)
for step in range(len(kif)):
    is_white = step % 2 != 0
    print(step)
    print(mat_2_boards(board_black, hand_black, board_white, hand_white, is_white)[0].kif_str())
    print('----------------')
    usi_move = kif[step]
    A = get_action_mat([usi_2_act_id(usi_move, is_white)])
    debug_usi = action_mat_2_usi(A, is_white)
    if len(debug_usi[0]) != 1 or usi_move not in debug_usi[0]:
        print(step, board.kif_str())
        print(debug_usi, usi_move)
        raise 'action mat conversion error'
    
    if not test_legal_moves(board, (board_black, hand_black, board_white), is_white):
        print(step, board.kif_str())
        print(mat_2_boards(board_black, hand_black, board_white, hand_white, is_white)[0].kif_str())
        raise 'legal moves not matched'
    
    board_black, hand_black, board_white, hand_white = apply_action_mat(board_black, hand_black, board_white, hand_white, A)
    board.push(shogi.Move.from_usi(usi_move))