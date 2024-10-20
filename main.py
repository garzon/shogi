import torch
import shogi
import numpy
import shogi.KIF

from shogi_rule_constants import *
from io_utils import *
    
    

def apply_action_mat(board_black, hand_black, board_white, hand_white, A):
    bougai = my_einsum("BKP,kFTP->BkFT", board_black + board_white, a_bougai)
    bougai_Kdim = torch.cat((bougai, torch.zeros(bougai.shape[0], K_dim-k_dim, bougai.shape[2], bougai.shape[3], dtype=torch.bool)), dim=1)

    # optimizations for tensor A by splitting it into tensors with lower dim
    BkFTp = my_einsum("BkFTp,BkFT->BkFTp", A, ~bougai_Kdim)
    BT = my_einsum("BkFTp->BT", BkFTp)
    BKF = my_einsum("BKFTp->BKF", BkFTp)
    
    S_black = torch.cat((board_black, hand_black), dim=1)
    after_removing = S_black ^ (S_black & BKF)
    after_removing_board_black, after_removing_hand_black = torch.split(after_removing, [k_dim, K_dim-k_dim], dim=1)
    after_removing_hand_black = my_einsum("BkF,KkFT,BKx->BkT", after_removing_hand_black, a_use_piece, BKF)
    
    new_board_black = after_removing_board_black + my_einsum("BkF,BkFTp,pkK->BKT", S_black, BkFTp, a_prom[:,:,:k_dim])
    captured_piece = my_einsum("BKT,BT,Kk->Bk", board_white, BT, a_captured)
    if_captured = my_einsum("Bk->B", captured_piece)
    new_hand_black_if_captured = my_einsum("BKF,Bk,kKFT->BKT", after_removing_hand_black, captured_piece, a_take_piece) + torch.cat((torch.zeros(captured_piece.shape[0], captured_piece.shape[1], 80, dtype=torch.bool), captured_piece.unsqueeze(2)), dim=2)
    new_hand_black = my_einsum("B,BkT->BkT", ~if_captured, after_removing_hand_black) + my_einsum("B,BkT->BkT", if_captured, new_hand_black_if_captured)
    new_board_white = board_white ^ my_einsum("BKT,BT->BKT", board_white, BT)
    
    #TODO: uchifu-tsumi
    
    return invert_order(new_board_black, new_hand_black, new_board_white, hand_white)
    

def calc_legal_moves_mat(board_black, hand_black, board_white):
    S_black = torch.cat((board_black, hand_black), dim=1)
    board_sum = board_black + board_white
    
    ok_to_move = my_einsum("BKF,KFTp->BKFTp", S_black, a_movable)
    
    uchibougai = my_einsum("BKP,kTP->BkT", board_sum, a_uchibougai)
    nifu = my_einsum("BKP,KPkT->BkT", board_black, a_nifu)
    bougai = my_einsum("BKP,kFTP->BkFT", board_sum, a_bougai)
    bougai_Kdim = torch.cat((bougai, torch.zeros(bougai.shape[0], K_dim-k_dim, bougai.shape[2], bougai.shape[3], dtype=torch.bool)), dim=1)
    jibougai = my_einsum("BKP,TP->BT", board_black, a_jibougai)
    
    board_pieces = my_einsum("BKP->BP", board_sum)
    board_if_moved = my_einsum("BP,PF->BPF", board_pieces, ~torch.eye(P_dim, dtype=torch.bool))
    board_if_moved = board_if_moved.unsqueeze(3).expand(-1, -1, -1, T_dim) + a_add_piece.unsqueeze(0)
    
    
    king_pos = board_black[:, KING, :].squeeze(1)
    #white_attack_king = my_einsum("BKF,KFt,Bt,BKFt->BF", board_white, a_oppo_move, king_pos, ~bougai)
    #bougai_pos = my_einsum("BKF,BF,Bt,KFtT->BT", board_white, a_oppo_move, king_pos, a_bougai)
    
    white_if_moved = my_einsum("BKP,PT->BKPT", board_white, ~torch.eye(P_dim, dtype=torch.bool))
    bougai_to_king = my_einsum("Bt,kftP->BkfP", king_pos, a_bougai)
    bougai_to_king = my_einsum("BkfT,BkfP,BPFT->BfFT", white_if_moved, bougai_to_king, board_if_moved)
    white_attack_without_bougai = my_einsum("BKPT,KPt,Bt->BPT", white_if_moved, a_oppo_move, king_pos)
    white_attack_king = my_einsum("BPT,BPFT->BFT", white_attack_without_bougai, ~bougai_to_king)
    
    bougai_to_king_if_king_moved = my_einsum("kfTP,BPFT,BkfT->BfFT", a_bougai, board_if_moved, white_if_moved)
    white_attack_king_if_king_moved = my_einsum("BKPT,KPT,BPFT->BFT", white_if_moved, a_oppo_move, ~bougai_to_king_if_king_moved)
    
    oute = torch.zeros(K_dim, board_black.shape[0], F_dim, T_dim, dtype=torch.bool)
    oute[:] = white_attack_king
    oute[KING] = white_attack_king_if_king_moved
    oute = torch.permute(oute, (1, 0, 2, 3)).contiguous()
    
    total_forbidden = (jibougai.unsqueeze(1) + uchibougai + nifu).unsqueeze(2) + bougai_Kdim + oute

    legal_moves_mat = my_einsum("BKFTp,BKFT->BKFTp", ok_to_move, ~total_forbidden)
    return legal_moves_mat
    
def test_legal_moves(board, mat, is_white):
    ok_mat = calc_legal_moves_mat(mat[0], mat[1], mat[2])

    my = sorted(action_mat_2_usi(ok_mat, is_white, None)[0])
    std = board.legal_moves
    diff = [_ for _ in std if _.usi() not in set(my)]
    diff2 = [m for m in my if m not in set(_.usi() for _ in std)]
    if len(diff) != 0 or len(diff2) != 0:
        print(len(my), len(std), my)
        print(sorted([_.usi()+board.piece_at(_.from_square).japanese_symbol() if _.from_square is not None else (_.usi()+shogi.PIECE_JAPANESE_SYMBOLS[_.drop_piece_type]) for _ in diff]))
        print(sorted(diff2))
        return False
    return True


if __name__ == '__main__':
    kif = shogi.KIF.Parser.parse_file("my.kif")[0]['moves']
    board = shogi.Board()
    step = -1
    for step in range(3):
        board.push(shogi.Move.from_usi(kif[step]))
    step += 1

    board_black, hand_black, board_white, hand_white = board_2_mat(board, step % 2 != 0)
    while step < len(kif):
        is_white = step % 2 != 0
        print('Step', step)
        print(mat_2_boards(board_black, hand_black, board_white, hand_white, is_white)[0].kif_str())
        print('----------------')
        usi_move = kif[step]
        
        A = get_action_mat([usi_2_act_id(usi_move, is_white)])
        debug_usi = action_mat_2_usi(A, is_white, torch.cat((board_black, hand_black), dim=1))
        print(debug_usi, usi_move)
        if len(debug_usi[0]) != 1 or usi_move not in debug_usi[0]:
            print('Step', step, board.kif_str())
            raise 'action mat conversion error'
        
        if not test_legal_moves(board, (board_black, hand_black, board_white), is_white):
            print('Step', step, board.kif_str())
            print(mat_2_boards(board_black, hand_black, board_white, hand_white, is_white)[0].kif_str())
            raise 'legal moves not matched'
        
        board_black, hand_black, board_white, hand_white = apply_action_mat(board_black, hand_black, board_white, hand_white, A)
        board.push(shogi.Move.from_usi(usi_move))
        step += 1