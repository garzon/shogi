import shogi
import torch

from constants import *

def rotate_180deg(arg):
    return torch.flip(arg, [2])
    
def invert_order(board_black, hand_black, board_white, hand_white):
    return rotate_180deg(board_white), hand_white, rotate_180deg(board_black), hand_black

def usi_2_act_id(usi, is_white):
    prom = 1 if len(usi) == 5 else 0
    t = shogi.SQUARE_NAMES.index(usi[2:4])
    if is_white: t = 80 - t
    if usi[1] == '*':
        hand_piece = shogi.PIECE_SYMBOLS.index(usi[0].lower())-1
        return hand_piece * 81 + t
    f = shogi.SQUARE_NAMES.index(usi[:2])
    if is_white: f = 80 - f
    return f * 81 * 2 + t * 2 + prom + 567

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
            A[ind, :k_dim, f, t, prom] = True
    return A

def action_mat_2_usi(A, is_white):
    ret = []
    for B in range(A.shape[0]):
        temp = set()
        for (K, F, T, Pr) in A[B].to_sparse().indices().transpose(0, 1):
            K, F, T, Pr = map(lambda x: x.item(), [K, F, T, Pr])
            if is_white:
                F = 80 - F
                T = 80 - T
            if K < k_dim:
                temp.add(shogi.SQUARE_NAMES[F]+shogi.SQUARE_NAMES[T]+("+" if Pr == 1 else ""))
                continue
            temp.add(shogi.PIECE_SYMBOLS[K-k_dim+1].upper()+"*"+shogi.SQUARE_NAMES[T])
        ret.append(temp)
    return ret
            
def mat_2_boards(board_black, hand_black, board_white, hand_white, is_white):
    boards = []
    colors = [shogi.BLACK, shogi.WHITE]
    if is_white: colors = colors[::-1]
    for b in range(len(board_black)):
        b_black = board_black[b]
        b_white = board_white[b]
        if is_white:
            b_black = rotate_180deg(board_black)[b]
            b_white = rotate_180deg(board_white)[b]
        board = shogi.Board()
        board.clear()
        for k in range(K_dim-k_dim):
            for p in range(80, -1, -1):
                if not hand_black[b, k, p]: break
            board.add_piece_into_hand(k+1, colors[0], 80-p)
            for p in range(80, -1, -1):
                if not hand_white[b, k, p]: break
            board.add_piece_into_hand(k+1, colors[1], 80-p)
        for (k, p) in b_black.to_sparse().indices().transpose(0, 1):
            board.set_piece_at(p, shogi.Piece(k+1, colors[0]))
        for (k, p) in b_white.to_sparse().indices().transpose(0, 1):
            board.set_piece_at(p, shogi.Piece(k+1, colors[1]))
        boards.append(board)
    return boards

def board_2_mat(board, is_white):
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
    if not is_white:
        return board_black, hand_black, board_white, hand_white
    return invert_order(board_black, hand_black, board_white, hand_white)