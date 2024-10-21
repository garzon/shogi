import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle
import shogi
import os

from constants import *
from main import *
from train_utils import *
from io_utils import *


ch = 192
fcl = 256
latest_features_dim = FEATURES_DIM

e_max_policy_loss = 3.0
e_value = 5.0
e_value_miss = 10.0
#e_policy_illegal = 8.0

MODEL1_PATH = "output/model1-11"
MODEL2_PATH = "output/model2-11"
TRAINING_DATA_PATH = [
    'output/train_list_feature3-5000.ckpt',
    'output/train_list_feature3-4500.ckpt',
    'output/train_list_feature3-3000.ckpt',
    #'output/train_Suisho10Mn_psv_-0.ckpt',
    #'output/train_Suisho10Mn_psv_-450000.ckpt',
]
#TRAINING_DATA_PATH = [
#    'output/train_Suisho10Mn_psv_-0.ckpt',
#]

class Block(nn.Module):
    def __init__(self):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = ch, out_channels = ch, kernel_size = 3, padding = 1, bias=False)
        self.bn1 = nn.BatchNorm2d(ch)
        self.conv2 = nn.Conv2d(in_channels = ch, out_channels = ch, kernel_size = 3, padding = 1, bias=False)
        self.bn2 = nn.BatchNorm2d(ch)

    def forward(self, x):
        h1 = F.relu(self.bn1(self.conv1(x)))
        h2 = self.bn2(self.conv2(h1))
        return F.relu(x + h2)

class PolicyValueResnet(nn.Module):
    def __init__(self, features_dim = K_dim*2, blocks = 10):
        super(PolicyValueResnet, self).__init__()
        self.blocks = blocks
      
        self.l1=nn.Conv2d(in_channels = features_dim, out_channels = ch, kernel_size = 3, padding = 1)
        self.block_layers = nn.ModuleList([Block() for _ in range(1, blocks)])

        # policy network
        self.policy=nn.Conv2d(in_channels = ch, out_channels = MOVE_DIRECTION_LABEL_NUM, kernel_size = 1, bias = False)
        self.policy_bias=nn.Parameter(torch.zeros(OUTPUT_DIM))
 
        # value network
        self.value1=nn.Conv2d(in_channels = ch, out_channels = MOVE_DIRECTION_LABEL_NUM, kernel_size = 1)
        self.value1_bn = nn.BatchNorm2d(MOVE_DIRECTION_LABEL_NUM)
        self.value2=nn.Linear(OUTPUT_DIM, fcl)
        self.value3=nn.Linear(fcl, 1)

    def forward(self, x):
        h = F.relu(self.l1(x))
        for block in self.block_layers:
            h = block(h)

        # policy network
        h_policy = self.policy(h)
        u_policy = self.policy_bias + torch.reshape(h_policy, (-1, OUTPUT_DIM,))

        # value network
        h_value = F.relu(self.value1_bn(self.value1(h)))
        h_value = F.relu(self.value2(h_value.view(-1, OUTPUT_DIM)))
        u_value = self.value3(h_value)

        return u_policy, u_value
        
def my_value_miss_loss(output, target):
    return torch.mean((output-target)*torch.abs(output-target))


if __name__ == '__main__':
    model1 = PolicyValueResnet(latest_features_dim, blocks=10).cuda()
    model2 = PolicyValueResnet(latest_features_dim, blocks=10).cuda()

    optimizer1 = optim.Adam(model1.parameters(), lr=0.001)
    optimizer2 = optim.Adam(model2.parameters(), lr=0.001)

    # Define the loss function
    policy_loss_fn = nn.CrossEntropyLoss()
    policy_loss_fn2 = nn.CrossEntropyLoss(reduction='none')
    value_loss_fn1 = nn.BCEWithLogitsLoss()
    value_loss_fn2 = nn.MSELoss()
    value_loss_fn3 = nn.BCELoss()

    '''
    # Generate some training data
    x = torch.randn(128, 104, 9, 9)  # Example input data
    policy_labels = torch.randint(0, 9 * 9 * MOVE_DIRECTION_LABEL_NUM, (128,))  # Example policy labels
    value_labels = torch.randn(128, 1)  # Example value labels

    model.load_state_dict(torch.load(PATH, weights_only=True))
    model.eval()
    '''
    
    if os.path.isfile(MODEL2_PATH):
        model1.load_state_dict(torch.load(MODEL1_PATH, weights_only=True))
        model2.load_state_dict(torch.load(MODEL2_PATH, weights_only=True))
        print("Trained model loaded.")
    
    print('Loading training data.')
    positions = []
    for p in TRAINING_DATA_PATH:
        print('Loading training data from', p)
        with open(p, 'rb') as f:
            positions += pickle.load(f)
    print('Loaded.')
    
    for epoch in range(1000000):
        x, policy_labels, value_labels = mini_batch(positions)
    
        
        policy_outputs, value_outputs = model1(x)
        policy_loss = policy_loss_fn(policy_outputs, policy_labels)
        value_loss = value_loss_fn1(value_outputs, (value_labels >= 0.5).to(dtype=torch.float32))
        loss1 = policy_loss + value_loss

        optimizer1.zero_grad()
        loss1.backward()
        optimizer1.step()
        
        # ==========================
        
        e_policy_loss = e_max_policy_loss * poss_to_drop(torch.abs(value_labels.squeeze(1)-0.5)*6.0, 3.0)
        
        policy_outputs, value_outputs = model2(x)
        policy_loss = torch.mean(policy_loss_fn2(policy_outputs, policy_labels) * e_policy_loss)
        value_loss = e_value * value_loss_fn1(value_outputs, value_labels)
        value_outputs = F.sigmoid(value_outputs)
        

        with torch.no_grad():
            board_black, hand_black, board_white, hand_white, _, _2 = map(lambda _:(_!=0).to('cpu', dtype=torch.bool), torch.split(x.reshape(x.shape[0], x.shape[1], 9*9), [k_dim, K_dim-k_dim, k_dim, K_dim-k_dim, k_dim, k_dim], dim=1))
            fake_is_white = False
            
            legal_mats = calc_legal_moves_mat(board_black, hand_black, board_white)
            bestmoves_label, legal_labelss = get_bestmoves_from_legal_mats_and_logitss(legal_mats, policy_outputs, fake_is_white, return_usi=False)

            '''
            policy_outputs = F.softmax(policy_outputs, dim=1)
            legal_policy = torch.zeros(policy_outputs.shape[0], policy_outputs.shape[1], dtype=torch.float32)
            for b in range(policy_outputs.shape[0]):
                labels = legal_labelss[b]
                for label in labels:
                    legal_policy[b, label] = policy_outputs[b, label]
            illegal_policy_loss = e_policy_illegal * value_loss_fn2(policy_outputs, legal_policy.to('cuda'))
            '''
            
            A = get_action_mat(bestmoves_label)
            board_black, hand_black, board_white, hand_white = apply_action_mat(board_black, hand_black, board_white, hand_white, A, to_cpu=False)
            black_attack, white_attack = calc_attack(board_black, board_white)
            x2 = torch.cat(to_gpu_float32(board_black, hand_black, board_white, hand_white, black_attack, white_attack), dim=1).reshape(-1, latest_features_dim, 9, 9).contiguous()
        
        '''
        boards = mat_2_boards(board_black, hand_black, board_white, hand_white, fake_is_white)
        # bestmoves_usi = get_bestmoves_from_logitss(boards, policy_outputs)
        bestmoves_usi, legal_labelss = get_bestmoves_from_legal_mats_and_logitss(legal_mats, policy_outputs, False)
        x2 = torch.zeros(x.shape[0], K_dim*2, 9, 9, dtype=torch.bool)
        for i in range(len(boards)):
            boards[i].push_usi(bestmoves_usi[i])
        x2 = boards_2_features(boards, True)
        '''
        
        _, value_outputs2 = model2(x2)
        value_outputs2 = F.sigmoid(value_outputs2)
        value_miss_loss = e_value_miss * value_loss_fn3(1.0-value_outputs2, value_outputs)
        loss2 = policy_loss + value_loss + value_miss_loss #+ illegal_policy_loss
        
        optimizer2.zero_grad()
        loss2.backward()
        optimizer2.step()

        # Print the loss
        if epoch % 500 == 0:
            #print(f"Epoch {epoch + 1}, Loss2: {loss2.item()}")
            print(f"Epoch {epoch + 1}, Loss1: {loss1.item()}, Loss2: {loss2.item()}")
        
        if epoch % 10000 == 9999:
            print('saving @' + str(epoch))
            torch.save(model1.state_dict(), MODEL1_PATH+"."+str(epoch))
            torch.save(model2.state_dict(), MODEL2_PATH+"."+str(epoch))
        
    torch.save(model1.state_dict(), MODEL1_PATH)
    torch.save(model2.state_dict(), MODEL2_PATH)