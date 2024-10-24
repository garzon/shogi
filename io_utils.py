import shogi
import torch
import cshogi

from shogi_rule_constants import *

def rotate_180deg(arg):
    return torch.flip(arg, [2])
    
def invert_order(board_black, hand_black, board_white, hand_white):
    return rotate_180deg(board_white), hand_white, rotate_180deg(board_black), hand_black

def usi_2_act_id(usi, is_white):
    t = shogi.SQUARE_NAMES.index(usi[2:4])
    if is_white: t = 80 - t
    if usi[1] == '*':
        hand_piece = shogi.PIECE_SYMBOLS.index(usi[0].lower())-1
        return from_to_2_act_id(None, t, False, hand_piece)
    f = shogi.SQUARE_NAMES.index(usi[:2])
    if is_white: f = 80 - f
    return from_to_2_act_id(f, t, len(usi) == 5)

# From https://github.com/TadaoYamaoka/python-dlshogi/blob/master/pydlshogi/features.py
def from_to_2_act_id(move_from, move_to, move_prom, move_drop_piece_type=None):
    # move direction
    if move_from is not None:
        to_y, to_x = divmod(move_to, 9)
        from_y, from_x = divmod(move_from, 9)
        dir_x = to_x - from_x
        dir_y = to_y - from_y
        if dir_y < 0 and dir_x == 0:
            move_direction = UP
        elif dir_y == -2 and dir_x == -1:
            move_direction = UP2_LEFT
        elif dir_y == -2 and dir_x == 1:
            move_direction = UP2_RIGHT
        elif dir_y < 0 and dir_x < 0:
            move_direction = UP_LEFT
        elif dir_y < 0 and dir_x > 0:
            move_direction = UP_RIGHT
        elif dir_y == 0 and dir_x < 0:
            move_direction = LEFT
        elif dir_y == 0 and dir_x > 0:
            move_direction = RIGHT
        elif dir_y > 0 and dir_x == 0:
            move_direction = DOWN
        elif dir_y > 0 and dir_x < 0:
            move_direction = DOWN_LEFT
        elif dir_y > 0 and dir_x > 0:
            move_direction = DOWN_RIGHT

        # promote
        if move_prom:
            move_direction = MOVE_DIRECTION_PROMOTED[move_direction]
    else:
        move_direction = len(MOVE_DIRECTION) + move_drop_piece_type

    move_label = 9 * 9 * move_direction + move_to
    return move_label
    
def get_action_mat(act_ids):
    A = torch.zeros(len(act_ids), K_dim, F_dim, T_dim, Pr_dim, dtype=torch.bool)
    for ind, act_id in enumerate(act_ids):
        if act_id == -1:
            A[ind, 0, 0, 0, 0] = True # no-op. bug: this results in taking the piece @ 0,0
            continue
        move_direction, t = divmod(act_id, 81)
        if move_direction >= len(MOVE_DIRECTION):
            piece_type = move_direction - len(MOVE_DIRECTION)
            f = 80
            prom = 0
            A[ind, piece_type + k_dim, f, t, prom] = True
        else:
            prom = 1 if (move_direction >= len(MOVE_DIRECTION) - len(MOVE_DIRECTION_PROMOTED)) else 0
            TD = torch.zeros(T_dim, len(MOVE_DIRECTION)-len(MOVE_DIRECTION_PROMOTED)+len(HAND_PIECE_TYPES), dtype=torch.bool)
            d = move_direction if prom == 0 else MOVE_DIRECTION_PROMOTED.index(move_direction)
            TD[t, d] = True
            can_goto = my_einsum("KFTD,TD->KF", a_move_dir, TD)
            A[ind, :, :, t, prom] = can_goto
            A[ind, k_dim:, :, t, prom] = False
    return A
a_id_2_mat = get_action_mat(list(range(-1, 9*9*MOVE_DIRECTION_LABEL_NUM)))
del get_action_mat
get_action_mat = lambda act_ids: a_id_2_mat[[_+1 for _ in act_ids]]

def action_mat_2_usi(A, is_white, is_unique_with_S_black):
    if is_unique_with_S_black is not None:
        A = my_einsum("BKFTp,BKF->BKFTp", A, is_unique_with_S_black)
            
    if is_white:
        A = torch.flip(A, [2, 3])

    ret = []
    for B in range(A.shape[0]):
        temp = set()
        min_d = 99
        for (K, F, T, Pr) in A[B].to_sparse().indices().transpose(0, 1):
            K, F, T, Pr = map(lambda x: x.item(), [K, F, T, Pr])
            if K < k_dim:
                usi = shogi.SQUARE_NAMES[F]+shogi.SQUARE_NAMES[T]+("+" if Pr == 1 else "")
                if is_unique_with_S_black is not None:
                    f_x, f_y = divmod(F, 9)
                    t_x, t_y = divmod(T, 9)
                    d = abs(f_x-t_x)+abs(f_y-t_y)
                    if d < min_d:
                        min_d = d
                        temp = set((usi,))
                else:
                    temp.add(usi)
                continue
            temp.add(shogi.PIECE_SYMBOLS[K-k_dim+1].upper()+"*"+shogi.SQUARE_NAMES[T])
        ret.append(temp)
    return ret
            
def mat_2_boards(board_black, hand_black, board_white, hand_white, is_white):
    boards = []
    colors = [shogi.BLACK, shogi.WHITE]
    if is_white: colors = colors[::-1]
    for b in range(board_black.shape[0]):
        b_black = board_black[b]
        b_white = board_white[b]
        if is_white:
            b_black = rotate_180deg(board_black)[b]
            b_white = rotate_180deg(board_white)[b]
        board = shogi.Board()
        board.clear()
        for (k, p) in b_black.to_sparse().indices().transpose(0, 1):
            board.set_piece_at(p.item(), shogi.Piece(k.item()+1, colors[0]))
        for (k, p) in b_white.to_sparse().indices().transpose(0, 1):
            board.set_piece_at(p.item(), shogi.Piece(k.item()+1, colors[1]))
            
        board.turn = shogi.WHITE if is_white else shogi.BLACK
        
        for k in range(K_dim-k_dim):
            for p in range(80, -1, -1):
                if not hand_black[b, k, p]: break
            if p != 80:
                board.add_piece_into_hand(k+1, colors[0], 80-p)
            for p in range(80, -1, -1):
                if not hand_white[b, k, p]: break
            if p != 80:
                board.add_piece_into_hand(k+1, colors[1], 80-p)
            
        board.pseudo_legal_moves = shogi.PseudoLegalMoveGenerator(board)
        board.legal_moves = shogi.LegalMoveGenerator(board)
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

CPIECE_TYPES = [_,
    BPawn, BLance, BKnight, BSilver, BBishop, BRook, BGold, BKing,
    BProPawn, BProLance, BProKnight, BProSilver, BHorse, BDragon, _, _,
    WPawn, WLance, WKnight, WSilver, WBishop, WRook, WGold, WKing,
    WProPawn, WProLance, WProKnight, WProSilver, WHorse, WDragon] = range(31)
CPIECE_2_PIECES_TYPE = {
    BPawn:PAWN, BLance:LANCE, BKnight:KNIGHT, BSilver:SILVER, BBishop:BISHOP, BRook:ROOK, BGold:GOLD, BKing:KING,
    BProPawn:PROM_PAWN, BProLance:PROM_LANCE, BProKnight:PROM_KNIGHT, BProSilver:PROM_SILVER, BHorse:PROM_BISHOP, BDragon:PROM_ROOK
}
def get_cboard_piece(cboard, f):
    cpiece = cboard.piece(f)
    if cpiece == 0: return None
    color = shogi.BLACK
    if cpiece >= 17:
        color = shogi.WHITE
        cpiece -= 16
    piece = shogi.Piece(CPIECE_2_PIECES_TYPE[cpiece]+1, color)
    return piece

def cboard_2_mat(cboard, is_white):
    board_black = torch.zeros(1, k_dim, F_dim, dtype=torch.bool)
    board_white = torch.zeros(1, k_dim, F_dim, dtype=torch.bool)
    hand_black = torch.zeros(1, K_dim - k_dim, F_dim, dtype=torch.bool)
    hand_white = torch.zeros(1, K_dim - k_dim, F_dim, dtype=torch.bool)
    for f in range(F_dim):
        piece = get_cboard_piece(cboard, f)
        if piece is None: continue
        k = piece.piece_type - 1
        col, row = divmod(f, 9)
        board_f = row * 9 + (8 - col)
        if piece.color == shogi.BLACK:
            board_black[0, k, board_f] = True
        else:
            board_white[0, k, board_f] = True
    for k in HAND_PIECE_TYPES:
        k_id = k - k_dim
        for _ in range(cboard.pieces_in_hand[shogi.BLACK][k_id]):
            hand_black[0, k_id, 80-_] = True
        for _ in range(cboard.pieces_in_hand[shogi.WHITE][k_id]):
            hand_white[0, k_id, 80-_] = True
    if not is_white:
        return board_black, hand_black, board_white, hand_white
    return invert_order(board_black, hand_black, board_white, hand_white)

def board_2_features(board, is_white):
    mat = board_2_mat(board, board.turn == shogi.WHITE)
    mat = torch.cat(mat, dim=1)[0].reshape(-1, 9, 9)
    return mat
    
def calc_attack(board_black, board_white):
    bougai = ~my_einsum2("BKP,kFTP->BkFT", board_black + board_white, a_bougai)
    board_black_attack = my_einsum2("BkF,kFT,BkFT->BkT", board_black, a_move[:k_dim], bougai)
    board_white_attack = my_einsum2("BkF,kFT,BkFT->BkT", board_white, a_oppo_move[:k_dim], bougai)
    return board_black_attack, board_white_attack
    
def mats_2_features2(board_black, hand_black, board_white, hand_white, board_black_attack, board_white_attack):
    return torch.cat((board_black.to('cpu', dtype=torch.bool), hand_black.to('cpu', dtype=torch.bool), board_white.to('cpu', dtype=torch.bool), hand_white.to('cpu', dtype=torch.bool), board_black_attack, board_white_attack), dim=0).reshape(-1, 9, 9)
    
    
def board_2_features2(board, is_white):
    board_black, hand_black, board_white, hand_white = map(lambda _:_.to('cuda'), board_2_mat(board, board.turn == shogi.WHITE))
    
    board_black_attack, board_white_attack = [_.squeeze(0).to('cpu', dtype=torch.bool) for _ in calc_attack(board_black, board_white)]
    board_black, hand_black, board_white, hand_white = map(lambda _:_.squeeze(0), [board_black, hand_black, board_white, hand_white])
    
    return mats_2_features2(board_black, hand_black, board_white, hand_white, board_black_attack, board_white_attack)
    
def cboard_2_features2(board, is_white):
    board_black, hand_black, board_white, hand_white = map(lambda _:_.to('cuda'), cboard_2_mat(board, board.turn == shogi.WHITE))
    
    board_black_attack, board_white_attack = [_.squeeze(0).to('cpu', dtype=torch.bool) for _ in calc_attack(board_black, board_white)]
    board_black, hand_black, board_white, hand_white = map(lambda _:_.squeeze(0), [board_black, hand_black, board_white, hand_white])
    
    return mats_2_features2(board_black, hand_black, board_white, hand_white, board_black_attack, board_white_attack)
    
def boards_2_features(boards, is_white, func=board_2_features2, to_cuda=True):
    feature = func(boards[0], is_white)
    features = torch.zeros(len(boards), feature.shape[0], 9, 9, dtype=torch.bool)
    features[0] = feature
    for f_id in range(1, len(boards)):
        features[f_id] = func(boards[f_id], is_white)
    return features.to('cuda', dtype=torch.float32) if to_cuda else features
    

def boltzmann(logits, temperature):
    logits /= temperature
    logits -= logits.max()
    probabilities = numpy.exp(logits)
    probabilities /= probabilities.sum()
    return numpy.random.choice(len(logits), p=probabilities)

def get_bestmoves_from_logitss(boards, logitss):
    bestmoves = []
    for idx in range(len(boards)):
        board = boards[idx]
        logits = logitss[idx]
        
        legal_moves = []
        legal_logits = []
        for move in board.legal_moves:
            usi = move.usi()
            label = usi_2_act_id(usi, board.turn == shogi.WHITE)
            legal_moves.append(move)
            legal_logits.append(logits[label].item())
            
        selected_index = boltzmann(numpy.array(legal_logits, dtype=numpy.float32), 0.5)
        #selected_index = greedy(legal_logits)
        bestmoves.append(legal_moves[selected_index].usi())
    return bestmoves
    
    
def get_bestmoves_from_legal_mats_and_logitss(legal_mats, logitss, is_white, return_usi=True):
    bestmoves = []
    legal_labelss = []
    usiss = action_mat_2_usi(legal_mats, is_white, None)
    
    for idx in range(legal_mats.shape[0]):
        logits = logitss[idx]
        usis = usiss[idx]
        
        legal_logits = []    # index to logits
        legal_moves = []     # index to usi
        legal_labels = set()
        index_to_labels = []
        for usi in usis:
            label = usi_2_act_id(usi, is_white)
            if label in legal_labels: continue
            legal_labels.add(label)
            
            index_to_labels.append(label)
            legal_moves.append(usi)
            legal_logits.append(logits[label].item())
        
        selected_index = boltzmann(numpy.array(legal_logits, dtype=numpy.float32), 0.5)
        #selected_index = greedy(legal_logits)
        bestmoves.append(legal_moves[selected_index] if return_usi else index_to_labels[selected_index])
        legal_labelss.append(legal_labels)
    return bestmoves, legal_labelss