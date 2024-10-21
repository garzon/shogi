import shogi
import shogi.CSA
import pickle
import random

from io_utils import *
from main import *

init_board_black, init_hand_black, init_board_white, init_hand_white = map(lambda _:_.to('cuda'), board_2_mat(shogi.Board(), False))

SKIP = 5000
TRAIN_PICKLE = 'output/train_list_feature3-{}.ckpt'.format(SKIP)
def save_features(positions):
    print('Saving', len(positions))
    with open(TRAIN_PICKLE, 'wb') as f:
        pickle.dump(positions, f)
    print('Saving done')

BASE_FILE_PATH = "D:\\github\\pydlshogi\\"
def read_kifu(kifu_list_file="kifu.txt", num=15000, save_every=500):
    positions = []
    with open(BASE_FILE_PATH+kifu_list_file, 'r') as f:
        for idx, line in enumerate(f.readlines()):
            if idx <= SKIP: continue
            if idx % 10 == 0: print(idx)
            if idx >= num: return positions
            filepath = BASE_FILE_PATH+line.rstrip('\r\n')
            kifu = shogi.CSA.Parser.parse_file(filepath)[0]
            
            win_color = shogi.BLACK if kifu['win'] == 'b' else shogi.WHITE
            board = shogi.Board()
            for step, move in enumerate(kifu['moves']):
                mat = board_2_features2(board, board.turn == shogi.WHITE)
                
                move_label = usi_2_act_id(move, board.turn == shogi.WHITE)
                
                win = 0.5 + 0.5 * ((step / len(kifu['moves']) * 10) ** 2 / 100)
                if win_color != board.turn:
                    win = 1.0 - win

                positions.append((mat, move_label, win))
                board.push_usi(move)
            
            '''
            win_color_is_black = kifu['win'] == 'b'
            
            mat = (init_board_black, init_hand_black, init_board_white, init_hand_white)
            
            for step, move in enumerate(kifu['moves']):
                is_white = step % 2 != 0
                black_attack, white_attack = [_.to('cpu',dtype=torch.bool) for _ in calc_attack(mat[0], mat[2])]
                features = mats_2_features2(*[_.squeeze(0) for _ in [*mat, black_attack, white_attack]])
                
                move_label = usi_2_act_id(move, is_white)
                
                win = 0.5 + 0.5 * (((step+1) / len(kifu['moves']) * 10) ** 2 / 100)
                if win_color_is_black == is_white:
                    win = 1.0 - win

                positions.append((features, move_label, win))
                
                A = get_action_mat([move_label])
                mat = apply_action_mat(*mat, A)
            '''
            if save_every is not None and idx % save_every == save_every-1:
                save_features(positions)
    return positions
    
def mini_batch(positions, batchsize=15, cons_size=6, device=torch.device("cuda")):
    mini_batch_data = []
    mini_batch_move = []
    mini_batch_win = []
    for _ in range(batchsize):
        ind = random.randint(0, len(positions)-cons_size)
        for i in range(ind, ind+cons_size):
            features, move, win = positions[i]
            mini_batch_data.append(features)
            mini_batch_move.append(move)
            mini_batch_win.append(win)
    return (
        torch.tensor(mini_batch_data, dtype=torch.float32, device=device),
        torch.tensor(mini_batch_move, dtype=torch.long, device=device),
        torch.tensor(mini_batch_win, dtype=torch.float32, device=device).reshape((-1, 1)),
    )
    
    
if __name__ == '__main__':
    with torch.no_grad():
        positions = read_kifu()
        save_features(positions)