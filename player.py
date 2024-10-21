import shogi
import numpy as np
import traceback

from pydlshogi.common import *
from pydlshogi.features import *
from pydlshogi.player.base_player import *

from train import *
from io_utils import *

def greedy(logits):
    return logits.index(max(logits))

def boltzmann(logits, temperature):
    logits /= temperature
    logits -= logits.max()
    probabilities = np.exp(logits)
    probabilities /= probabilities.sum()
    return np.random.choice(len(logits), p=probabilities)

class PolicyPlayer(BasePlayer):
    def __init__(self):
        super().__init__()
        self.modelfile = r"D:\github\gzshogi\output\model2-5-2.ckpt"
        self.model = None

    def usi(self):
        print('id name gzShogi')
        print('id author garzon')
        print('option name modelfile type string default ' + self.modelfile)
        print('usiok')

    def setoption(self, option):
        if option[1] == 'modelfile':
            self.modelfile = option[3]

    def isready(self):
        if self.model is None:
            self.model = PolicyValueResnet(latest_features_dim).cuda()
            self.model.load_state_dict(torch.load(self.modelfile, weights_only=True))
            self.model.eval()
        print('readyok')

    def go(self):
        if self.board.is_game_over():
            print('bestmove resign')
            return

        x = boards_2_features([self.board], self.board.turn == shogi.WHITE)#, func=board_2_features)

        with torch.no_grad():
            policy_outputs, value_outputs = self.model(x)
            logits = policy_outputs[0]
            s = int(torch.nn.functional.sigmoid(value_outputs[0][0])*60000-30000)
            print('info score cp', s)
            print('info string', s)
            probabilities = F.softmax(logits).cpu()

        legal_moves = []
        legal_logits = []
        for move in self.board.legal_moves:
            usi = move.usi()
            label = usi_2_act_id(usi, self.board.turn == shogi.WHITE)
            legal_moves.append(move)
            legal_logits.append(logits[label].item())
            #print('info string {:5} : {:.5f}'.format(usi, probabilities[label]))
            
        selected_index = boltzmann(np.array(legal_logits, dtype=np.float32), 0.5)
        #selected_index = greedy(legal_logits)
        bestmove = legal_moves[selected_index]

        print('bestmove', bestmove.usi())

if __name__=='__main__':
    from pydlshogi.usi.usi import *

    try:
        player = PolicyPlayer()
        usi(player)
    except Exception as e:
        print('info string', traceback.format_exc().replace('\n', '').replace('\r', ''))