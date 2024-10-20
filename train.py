import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle
import shogi

from constants import *
from main import *
from train_utils import *
from io_utils import *

ch = 192
fcl = 256

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
    def __init__(self, blocks = 5):
        super(PolicyValueResnet, self).__init__()
        self.blocks = blocks
      
        self.l1=nn.Conv2d(in_channels = K_dim*2, out_channels = ch, kernel_size = 3, padding = 1)
        self.block_layers = nn.ModuleList([Block() for _ in range(1, blocks)])

        # policy network
        self.policy=nn.Conv2d(in_channels = ch, out_channels = MOVE_DIRECTION_LABEL_NUM, kernel_size = 1, bias = False)
        self.policy_bias=nn.Parameter(torch.zeros(9*9*MOVE_DIRECTION_LABEL_NUM))
 
        # value network
        self.value1=nn.Conv2d(in_channels = ch, out_channels = MOVE_DIRECTION_LABEL_NUM, kernel_size = 1)
        self.value1_bn = nn.BatchNorm2d(MOVE_DIRECTION_LABEL_NUM)
        self.value2=nn.Linear(9*9*MOVE_DIRECTION_LABEL_NUM, fcl)
        self.value3=nn.Linear(fcl, 1)

    def forward(self, x):
        h = F.relu(self.l1(x))
        for block in self.block_layers:
            h = block(h)

        # policy network
        h_policy = self.policy(h)
        u_policy = self.policy_bias + torch.reshape(h_policy, (-1, 9*9*MOVE_DIRECTION_LABEL_NUM,))

        # value network
        h_value = F.relu(self.value1_bn(self.value1(h)))
        h_value = F.relu(self.value2(h_value.view(-1, 9*9*MOVE_DIRECTION_LABEL_NUM)))
        u_value = self.value3(h_value)

        return u_policy, u_value
        
def my_value_miss_loss(output, target):
    return torch.mean(output-target)


BATCH_SIZE = 128
if __name__ == '__main__':
    model1 = PolicyValueResnet(blocks=5).cuda()
    model2 = PolicyValueResnet(blocks=5).cuda()

    optimizer1 = optim.Adam(model1.parameters(), lr=0.001)
    optimizer2 = optim.Adam(model2.parameters(), lr=0.001)

    # Define the loss function
    policy_loss_fn = nn.CrossEntropyLoss()
    value_loss_fn1 = nn.BCEWithLogitsLoss()
    value_loss_fn2 = nn.MSELoss()

    '''
    # Generate some training data
    x = torch.randn(128, 104, 9, 9)  # Example input data
    policy_labels = torch.randint(0, 9 * 9 * MOVE_DIRECTION_LABEL_NUM, (128,))  # Example policy labels
    value_labels = torch.randn(128, 1)  # Example value labels

    model.load_state_dict(torch.load(PATH, weights_only=True))
    model.eval()
    '''
    
    with open(TRAIN_PICKLE, 'rb') as f:
        positions = pickle.load(f)
    print('Loaded.')
    
    e_value = 5.0
    e_value_miss = 2.0

    for epoch in range(10000):
        x, policy_labels, value_labels = mini_batch(positions)
    
        policy_outputs, value_outputs = model1(x)
        policy_loss = policy_loss_fn(policy_outputs, policy_labels)
        value_loss = value_loss_fn1(value_outputs, (value_labels >= 0.5).to(dtype=torch.float32))
        loss1 = policy_loss + value_loss

        optimizer1.zero_grad()
        loss1.backward()
        optimizer1.step()
        
        # ==========================
        
        policy_outputs, value_outputs = model2(x)
        policy_loss = policy_loss_fn(policy_outputs, policy_labels)
        value_loss = e_value * value_loss_fn2(value_outputs, value_labels)

        board_black, hand_black, board_white, hand_white = map(lambda _:_.to('cpu', dtype=torch.bool), torch.split(x.reshape(x.shape[0], x.shape[1], 9*9), [k_dim, K_dim-k_dim, k_dim, K_dim-k_dim], dim=1))
        boards = mat_2_boards(board_black, hand_black, board_white, hand_white, False)
        bestmoves_usi = get_bestmoves_from_logitss(boards, policy_outputs)
        
        x2 = torch.zeros(x.shape[0], K_dim*2, 9, 9, dtype=torch.bool)
        for i in range(len(boards)):
            boards[i].push_usi(bestmoves_usi[i])
            x2[i] = board_2_features(boards[i], True)
        x2 = x2.to('cuda', dtype=torch.float32)
        
        _, value_outputs2 = model2(x2)
        value_miss_loss = e_value_miss * my_value_miss_loss(1.0-value_outputs2, value_outputs)
        loss2 = policy_loss + value_loss + value_miss_loss
        
        optimizer2.zero_grad()
        loss2.backward()
        optimizer2.step()

        # Print the loss
        if epoch % 50 == 0:
            print(f"Epoch {epoch + 1}, Loss1: {loss1.item()}, Loss2: {loss2.item()}")
        
    torch.save(model1.state_dict(), "output/model1.ckpt")
    torch.save(model2.state_dict(), "output/model2-5-2.ckpt")