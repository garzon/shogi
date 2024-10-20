import shogi
import shogi.CSA
import pickle
import random

from io_utils import *

BASE_FILE_PATH = "D:\\github\\pydlshogi\\"
def read_kifu(kifu_list_file="kifu.txt", num=200):
    positions = []
    with open(BASE_FILE_PATH+kifu_list_file, 'r') as f:
        for idx, line in enumerate(f.readlines()):
            if idx % 100 == 0: print(idx)
            if idx >= num: return positions
            filepath = BASE_FILE_PATH+line.rstrip('\r\n')
            kifu = shogi.CSA.Parser.parse_file(filepath)[0]
            win_color = shogi.BLACK if kifu['win'] == 'b' else shogi.WHITE
            board = shogi.Board()
            for step, move in enumerate(kifu['moves']):
                mat = board_2_features(board, board.turn == shogi.WHITE)
                
                move_label = usi_2_act_id(move, board.turn == shogi.WHITE)
                
                win = 0.5 + 0.5 * ((step / len(kifu['moves']) * 10) ** 2 / 100)
                if win_color != board.turn:
                    win = 1.0 - win

                positions.append((mat.tolist(), move_label, win))
                board.push_usi(move)
    return positions
    
def mini_batch(positions, batchsize=3, cons_size=3, device=torch.device("cuda")):
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
    
TRAIN_PICKLE = 'output/train_list2.ckpt'
if __name__ == '__main__':
    positions = read_kifu()
    print(len(positions))
    with open(TRAIN_PICKLE, 'wb') as f:
        pickle.dump(positions, f)
    