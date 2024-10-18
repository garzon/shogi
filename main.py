import torch
import shogi
import numpy
import shogi.KIF

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


###############################################################

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

def get_action_mat(act_ids):
    A = torch.zeros(len(act_ids), K_dim, F_dim, T_dim, Pr_dim, dtype=torch.bool)
    for ind, act_id in enumerate(act_ids):
        # F * T * prom (13112) + HAND_PIECE_TYPES * T (567)
        if act_id < 567:
            piece_type = act_id // 81
            t = act_id % 81
            f = 80
            prom = 0
            A[ind, piece_type + k_dim, f, t, prom] = True
        else:
            act_id -= 567
            f = act_id // 81 // 2
            t = act_id // 2 % 81
            prom = act_id % 2
            A[ind, :, f, t, prom] = True
    return A

def action_mat_2_usi(A):
    ret = []
    for B in range(A.shape[0]):
        temp = []
        for (K, F, T, Pr) in A[B].to_sparse().indices().transpose(0, 1):
            K, F, T, Pr = map(lambda x: x.item(), [K, F, T, Pr])
            if K < k_dim:
                temp.append(shogi.SQUARE_NAMES[F]+shogi.SQUARE_NAMES[T]+("+" if Pr == 1 else ""))
                continue
            temp.append(shogi.PIECE_SYMBOLS[K-k_dim+1].upper()+"*"+shogi.SQUARE_NAMES[T])
        ret.append(temp)
    return ret
            

def board_2_mat(board):
    board_black = torch.zeros(1, k_dim, F_dim, dtype=torch.bool)
    board_white = torch.zeros(1, k_dim, F_dim, dtype=torch.bool)
    hand_black = torch.zeros(1, K_dim - k_dim, F_dim, dtype=torch.bool)
    hand_white = torch.zeros(1, K_dim - k_dim, F_dim, dtype=torch.bool)
    for f in range(F_dim):
        piece = board.piece_at(f)
        if piece is None: continue
        k = piece.piece_type - 1
        if piece.color == shogi.BLACK:
            board_black[0, k, f] = True
        else:
            board_white[0, k, f] = True
    for k in HAND_PIECE_TYPES:
        k_id = k - k_dim + 1
        for _ in range(board.pieces_in_hand[shogi.BLACK][k_id]):
            hand_black[0, k_id-1, 80-_] = True
        for _ in range(board.pieces_in_hand[shogi.WHITE][k_id]):
            hand_white[0, k_id-1, 80-_] = True
    return board_black, hand_black, board_white, hand_white
    
def my_einsum(eq, *args):
    #args = tuple(map(lambda _: (_ > 0).type(torch.uint8), args))
    #print([arg.dtype for arg in args])
    ret = torch.from_numpy(numpy.einsum(eq, *args) > 0)
    return ret

#a_move, a_prom, a_prom_able, a_left, a_bougai, a_jibougai, a_uchibougai = tuple(map(lambda _: _.type(torch.uint8), [a_move, a_prom, a_prom_able, a_left, a_bougai, a_jibougai, a_uchibougai]))
def calc_legal_moves_mat(board_black, hand_black, board_white):
    S_black = torch.cat((board_black, hand_black), dim=1)
    
    ok_to_move = my_einsum("BKF,KFT,pKk,pFT,kT->BKFTp", S_black, a_move, a_prom, a_prom_able, a_left)

    uchibougai = my_einsum("BKP,kTP->BkT", board_black + board_white, a_uchibougai)
    nifu = my_einsum("BKP,KPkT->BkT", board_black, a_nifu)
    bougai = my_einsum("BKP,kFTP->BkFT", board_black + board_white, a_bougai)
    jibougai = my_einsum("BKP,TP->BT", board_black, a_jibougai)
    oute = torch.zeros(board_black.shape[0], K_dim, T_dim, dtype=torch.bool)
    oute[:, KING, :] = my_einsum("Bkp,kpT,BkpT->BT", board_white, a_oppo_move, ~bougai[:,:k_dim,:,:])
    total_forbidden = (jibougai.unsqueeze(1) + uchibougai + oute + nifu).unsqueeze(2) + bougai
    ok_to_go = ~total_forbidden

    legal_moves_mat = my_einsum("BKFTp,BKFT->BKFTp", ok_to_move, ok_to_go)

    return legal_moves_mat
    
def test_legal_moves(board):
    board_black, hand_black, board_white, hand_white = board_2_mat(board)
    ok_mat = calc_legal_moves_mat(board_black, hand_black, board_white)

    my = sorted(action_mat_2_usi(ok_mat)[0])
    std = board.legal_moves
    diff = [_ for _ in std if _.usi() not in set(my)]
    diff2 = [m for m in my if m not in set(_.usi() for _ in std)]
    #if len(diff) != 0 or len(diff2) != 0:
    print(len(my), len(std), my)
    print(sorted([_.usi()+board.piece_at(_.from_square).japanese_symbol() if _.from_square is not None else (_.usi()+shogi.PIECE_JAPANESE_SYMBOLS[_.drop_piece_type]) for _ in diff]))
    print(sorted(diff2))

    
kif = shogi.KIF.Parser.parse_file("my.kif")[0]['moves']
board = shogi.Board()
for _ in range(40):#len(kif)):
    board.push(shogi.Move.from_usi(kif[_]))
if _ % 2 == 1:
    print(board.kif_str())
    test_legal_moves(board)