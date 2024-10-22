import cshogi
import shogi.CSA
import pickle
import random
import math

from io_utils import *
from main import *

#init_board_black, init_hand_black, init_board_white, init_hand_white = map(lambda _:_.to('cuda'), board_2_mat(shogi.Board(), False))

SKIP = 0
TRAIN_PICKLE = 'output/train_hao_-{}.ckpt'.format(SKIP)
def save_features(positions):
    print('Saving', len(positions))
    with open(TRAIN_PICKLE, 'wb') as f:
        pickle.dump(positions, f)
    print('Saving done')
    
BASE_FILE_PATH = "D:\\github\\pydlshogi\\"
PSV_FILES=["Suisho10Mn_psv.bin"]
PSV_FILES=['kifus/hao/kifu.tag=train.depth=9.num_positions=1000000000.start_time=1695340981.thread_index=000.bin',
'kifus/hao/kifu.tag=train.depth=9.num_positions=1000000000.start_time=1695340981.thread_index=001.bin',
'kifus/hao/kifu.tag=train.depth=9.num_positions=1000000000.start_time=1695340981.thread_index=002.bin',
'kifus/hao/kifu.tag=train.depth=9.num_positions=1000000000.start_time=1695340981.thread_index=004.bin',
'kifus/hao/kifu.tag=train.depth=9.num_positions=1000000000.start_time=1695340981.thread_index=005.bin',
'kifus/hao/kifu.tag=train.depth=9.num_positions=1000000000.start_time=1695340981.thread_index=007.bin',
'kifus/hao/kifu.tag=train.depth=9.num_positions=1000000000.start_time=1695340981.thread_index=009.bin',
'kifus/hao/kifu.tag=train.depth=9.num_positions=1000000000.start_time=1695340981.thread_index=010.bin',
'kifus/hao/kifu.tag=train.depth=9.num_positions=1000000000.start_time=1695340981.thread_index=014.bin',
'kifus/hao/kifu.tag=train.depth=9.num_positions=1000000000.start_time=1695340981.thread_index=015.bin',
'kifus/hao/kifu.tag=train.depth=9.num_positions=1000000000.start_time=1695340981.thread_index=017.bin',
'kifus/hao/kifu.tag=train.depth=9.num_positions=1000000000.start_time=1695340981.thread_index=019.bin',
'kifus/hao/kifu.tag=train.depth=9.num_positions=1000000000.start_time=1695340981.thread_index=024.bin',
'kifus/hao/kifu.tag=train.depth=9.num_positions=1000000000.start_time=1695340981.thread_index=032.bin']
def read_psv(psv_files=PSV_FILES, num=5000000, save_every=100000, skip=SKIP):
    positions = []
    for psv_file in psv_files:
        print('Reading file:', psv_file)
        psfens = numpy.fromfile(BASE_FILE_PATH+psv_file, dtype=cshogi.PackedSfenValue)
        
        f_idx = skip
        while f_idx < len(psfens):
            if len(positions) % 10000 == 0:
                print('Parsing', f_idx, '/', len(psfens), '. Total', len(positions))
            if len(positions) >= num: return positions
            
            features = gen_features_for_psfen_at(psfens, f_idx)
            if features is not None:
                positions.append(features)
            if save_every is not None and len(positions) % save_every == save_every-1:
                save_features(positions)
            
            f_idx += 1
        if skip != 0:
            skip -= len(psfens)
            skip = max(0, skip)
            print('Remaining skip:', skip)
        print('Finished', psv_file, '@', f_idx, '/', len(psfens))
    return positions
    
def gen_features_for_psfen_at(psfens, idx):
    step = psfens[idx]['gamePly']
    if step <= 30:
        step_percentage = step/100.0
        if random.random() > poss_to_drop(step_percentage):
            return None

    win_color = cshogi.BLACK if psfens[idx]['game_result'] == 0 else shogi.WHITE
    
    board = cshogi.Board()
    board.set_psfen(psfens[idx]['sfen'])
    mat = cboard_2_features2(board, board.turn == shogi.WHITE)
    
    move = cshogi.move_to_usi(cshogi.move16_from_psv(psfens[idx]['move']))
    move_label = usi_2_act_id(move, board.turn == shogi.WHITE)
    
    win = (min(3000, max(-3000, psfens[idx]['score']))+3000.0)/6000.0

    return (mat, move_label, win)


def read_kifu(kifu_list_file="kifu.txt", num=30000, save_every=500):
    positions = []
    with open(BASE_FILE_PATH+kifu_list_file, 'r') as f:
        for idx, line in enumerate(f.readlines()):
            if idx <= SKIP: continue
            if idx % 10 == 0: print(idx)
            if idx >= num: return positions
            filepath = BASE_FILE_PATH+line.rstrip('\r\n')
            kifu = shogi.CSA.Parser.parse_file(filepath)[0]
            
            win_color = cshogi.BLACK if kifu['win'] == 'b' else shogi.WHITE
            board = cshogi.Board()
            # 11s/10kifu
            for step, move in enumerate(kifu['moves']):
                mat = board_2_features2(board, board.turn == shogi.WHITE)
                
                move_label = usi_2_act_id(move, board.turn == shogi.WHITE)
                
                win = 0.5 + 0.5 * ((step / len(kifu['moves']) * 10) ** 2 / 100)
                if win_color != board.turn:
                    win = 1.0 - win

                positions.append((mat, move_label, win))
                board.move_from_usi(move)
            
            ''' slow: 25s/10kifu
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
    
def poss_to_drop(win, max_x=6):
    win = torch.tensor(win)
    min_x = -3
    step_percentage = torch.abs(win-0.5)*2
    poss = step_percentage*3.3333*(max_x-min_x)+min_x
    return torch.nn.functional.sigmoid(poss)
    
def mini_batch(positions, batchsize=100, cons_size=6, device=torch.device("cuda")):
    mini_batch_data = []
    mini_batch_move = []
    mini_batch_win = []
    while len(mini_batch_win) < batchsize:
        ind = random.randint(0, len(positions)-cons_size)
        _, _, win = positions[ind]
        #step_percentage = abs(win-0.5)*2
        #if step_percentage < 0.3:
        #    if random.random() > poss_to_drop(win):
        #        continue
        for i in range(ind, ind+cons_size):
            features, move, win = positions[i]
            mini_batch_data.append(features)
            mini_batch_move.append(move)
            mini_batch_win.append(win)
    return (
        torch.stack(mini_batch_data).to(dtype=torch.float32, device=device),
        torch.tensor(mini_batch_move, dtype=torch.long, device=device),
        torch.tensor(mini_batch_win, dtype=torch.float32, device=device).reshape((-1, 1)),
    )
    
    
if __name__ == '__main__':
    with torch.no_grad():
        positions = read_psv()
        save_features(positions)