import torch
import shogi
import numpy
import shogi.KIF

from shogi_rule_constants import *
from io_utils import *
    
def my_einsum(eq, *args):
    return torch.from_numpy(numpy.einsum(eq, *args) > 0)
    
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
    
    new_board_black = after_removing_board_black + my_einsum("BkF,BkFTp,pkK->BKT", S_black, A, a_prom[:,:,:k_dim])
    captured_piece = my_einsum("BKT,BT,Kk->Bk", board_white, BT, a_captured)
    if_captured = my_einsum("Bk->B", captured_piece)
    new_hand_black_if_captured = my_einsum("BKF,Bk,kKFT->BKT", after_removing_hand_black, captured_piece, a_take_piece) + torch.cat((torch.zeros(captured_piece.shape[0], captured_piece.shape[1], 80, dtype=torch.bool), captured_piece.unsqueeze(2)), dim=2)
    new_hand_black = ~if_captured * after_removing_hand_black + if_captured * new_hand_black_if_captured
    new_board_white = board_white ^ my_einsum("BKT,BT->BKT", board_white, BT)
    
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