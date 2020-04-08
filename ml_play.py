"""
The template of the main script of the machine learning process
"""
import pickle
import numpy as np
import games.arkanoid.communication as comm
from games.arkanoid.communication import ( \
    SceneInfo, GameStatus, PlatformAction
)
import os.path as path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as DATA

class hw_model(nn.Module):
    def __init__(self):
        super(hw_model, self).__init__()
        self.emb = nn.Embedding(5, 5)
        
        self.data_fc1 = nn.Linear(48, 64)
        self.data_fc2 = nn.Linear(64, 128)
        

        self.fc1 = nn.Linear(128+5, 512)
        self.fc2 = nn.Linear(512,256)
        
        self.data_out1 = nn.Linear(256,64)
        self.data_out2 = nn.Linear(64,8)

        self.rnn = nn.LSTM(64,128, 1,batch_first=True)
        self.label_out1 = nn.Linear(128,32)
        self.label_out2 = nn.Linear(32,3)
        
    def forward(self, input_data, input_direct, bs):
        x = input_data
        x = F.relu(self.data_fc1(x))
        x = F.sigmoid(self.data_fc2(x))

        y = input_direct
        y = F.softmax(self.emb(y))

        cat = torch.cat([x, y], 1)

        out = F.relu(self.fc1(cat))
        out_feature = F.relu(self.fc2(out))

        data_out = F.relu(self.data_out1(out_feature))
        data_out = F.tanh(self.data_out2(data_out))
        
        r_in = out_feature.view(-1,4,64)
        
        r_out ,(h_n, hc) =  self.rnn(r_in)
        label_out = F.relu(self.label_out1(r_out[:, -1, :]))

        label_out = F.softmax(self.label_out2(label_out))

        return label_out, data_out

# class hw_model(nn.Module):
#     def __init__(self):
#         super(hw_model, self).__init__()
#         self.emb = nn.Embedding(5, 5)
#         self.data_fc1 = nn.Linear(6, 16)
#         self.data_fc2 = nn.Linear(16, 32)
        

#         self.fc1 = nn.Linear(32+5, 64)
#         self.fc2 = nn.Linear(64,32)
#         self.fc3 = nn.Linear(32,16)
#         self.label_out = nn.Linear(16,3)
#         self.data_out = nn.Linear(32,6)

#     def forward(self, input_data, input_direct, bs):
#         x = input_data
#         x = F.relu(self.data_fc1(x))
#         x = F.sigmoid(self.data_fc2(x))

#         y = input_direct
#         y = F.softmax(self.emb(y))

#         cat = torch.cat([x, y], 1)

#         out = F.sigmoid(self.fc1(cat))
#         out = F.relu(self.fc2(out))
#         data_out = F.tanh(self.data_out(out))
#         out = F.relu(self.fc3(out))
#         label_out = F.softmax(self.label_out(out))
#         return label_out, data_out
    #--------------------------------------------------------------
# class hw_model(nn.Module):
#     def __init__(self):
#         super(hw_model, self).__init__()
#         self.emb = nn.Embedding(5, 5)
#         self.data_fc1 = nn.Linear(3, 16)
#         self.data_fc2 = nn.Linear(16, 32)
        
#         self.fc00 = nn.Linear(21,64)
#         self.fc00_deconv = nn.ConvTranspose2d(64,32,3,1)
#         self.fc00_deconv_bn = nn.BatchNorm2d(32)

#         self.fc1 = nn.Linear(32*3*3, 128)
#         self.fc2 = nn.Linear(128,64)
#         self.fc3 = nn.Linear(64,32)
#         self.label_out1 = nn.Linear(32,16)
#         self.label_out2 = nn.Linear(16,3)
#         self.data_out = nn.Linear(32,3)

#     def forward(self, input_data, input_direct, bs):
#         x = input_data
#         x = F.relu(self.data_fc1(x))
#         # x = F.sigmoid(self.data_fc2(x))

#         y = input_direct
#         y = F.softmax(self.emb(y))

#         cat = torch.cat([x, y], 1)
#         cat = F.relu(self.fc00(cat))
#         cat = cat.unsqueeze(2).unsqueeze(3)
#         cat = F.leaky_relu(self.fc00_deconv_bn(self.fc00_deconv(cat)),0.2)
#         cat = cat.view(cat.size(0), -1)
#         cat = F.relu(self.fc1(cat))
#         cat = F.relu(self.fc2(cat))
#         feature = F.relu(self.fc3(cat))

#         data_out = torch.tanh(self.data_out(feature))

#         label_out = F.relu(self.label_out1(feature))
#         label_out = F.softmax(self.label_out2(label_out))

#         return label_out, data_out
        
def label_to_move(input_label):
    out_data = []
    
    for n in input_label:
        if np.max(n) == n[0]:
            out_data.append(0)
        elif np.max(n) == n[1]:
            out_data.append(1)
        elif np.max(n) == n[2]:
            out_data.append(2)
        else:
            print('>>lable to move Error!')
    print(out_data)
    return out_data

def ml_loop():
    """
    The main loop of the machine learning process

    This loop is run in a separate process, and communicates with the game process.

    Note that the game process won't wait for the ml process to generate the
    GameInstruction. It is possible that the frame of the GameInstruction
    is behind of the current frame in the game process. Try to decrease the fps
    to avoid this situation.
    """

    # === Here is the execution order of the loop === #
    # 1. Put the initialization code here.
    ball_served = False
	# 讀取模型資料
    # filename = path.join(path.dirname(__file__),"save\clf_KMeans_BallAndDirection.pickle")
    filename = path.join(path.dirname(__file__),"save\\test_model_02.pickle")
    # filename = path.join(path.dirname(__file__),"\games\arkanoid\ml\save\clf_KMeans_BallAndDirection.pickle")
    # with open(filename, 'rb') as file:
        
    #     clf = pickle.load(file)
    model = hw_model()
    model.load_state_dict(torch.load(filename))
    model.eval()
    # 2. Inform the game process that ml process is ready before start the loop.
    comm.ml_ready()
    
    s = [93,93]
    def get_direction(ball_x,ball_y,ball_pre_x,ball_pre_y):
        
        VectorX = ball_x - ball_pre_x
        VectorY = ball_y - ball_pre_y
        
        if(VectorX>=0 and VectorY>=0):
            return 0,VectorX,VectorY
        elif(VectorX>0 and VectorY<0):
            return 1,VectorX,VectorY
        elif(VectorX<0 and VectorY>0):
            return 2,VectorX,VectorY
        elif(VectorX<0 and VectorY<0):
            return 3,VectorX,VectorY
        else:
            return 4,VectorX,VectorY
    
    kk =[]
    order = 6
    kk = np.zeros((order, 8))
    # 3. Start an endless loop.
    
    while True:
        # 3.1. Receive the scene information sent from the game process.
        scene_info = comm.get_scene_info()#0326 取得資料
        feature = []
        feature.append(scene_info.ball[0])
        feature.append(scene_info.ball[1])
        feature.append(scene_info.ball[0]+5) #球x+5
        feature.append(scene_info.ball[1]+5) #球y+5
        feature.append(scene_info.platform[0])
        feature.append(scene_info.platform[0]+40) #板子x+40

        direct, velx,vely = get_direction(feature[0],feature[1],s[0],s[1])
        feature.append(velx)
        feature.append(vely)
        feature.append(direct)
        s = [feature[0], feature[1]]
        feature = np.array(feature)
        feature = feature.reshape((-1,9))
        # 3.2. If the game is over or passed, the game process will reset
        #      the scene and wait for ml process doing resetting job.
        if scene_info.status == GameStatus.GAME_OVER or \
            scene_info.status == GameStatus.GAME_PASS:
            # Do some stuff if needed
            ball_served = False

            # 3.2.1. Inform the game process that ml process is ready
            comm.ml_ready()
            continue

        # 3.3. Put the code here to handle the scene information

        # 3.4. Send the instruction for this frame to the game process
        if not ball_served:
            comm.send_instruction(scene_info.frame, PlatformAction.SERVE_TO_LEFT)
            ball_served = True
        else:
            input_data = feature[:,:6]
            input_vel = feature[ : , 6:8]
            input_data_norm = (input_data - 149.37214)/97.815793
            input_vel_norm = (input_vel - (-0.00629))/7.5278
            input_data_now = np.hstack((input_data_norm,input_vel_norm))
            
            input_data_plus = []
            kk[5, :] = kk[4, :]
            kk[4, :] = kk[3, :]
            kk[3, :] = kk[2, :]
            kk[2, :] = kk[1, :]
            kk[1, :] = kk[0, :]
            kk[0, :] = input_data_now

            input_data_plus.append(np.array(kk).flatten())
            input_data_plus = np.array(input_data_plus)
            
            input_direct = feature[:, -1]
            input_data = torch.from_numpy(input_data_plus).type(torch.FloatTensor)
            input_direct = torch.from_numpy(input_direct).type(torch.LongTensor)
            # feature = torch.from_numpy(feature).type(torch.FloatTensor)
            
            out ,_= model(input_data, input_direct, len(feature))
            parms = model.state_dict()
            # print('check parameter >> ',parms['label_out.weight'])
            out = out.data.numpy()
            print(out)
            y = label_to_move(out)
            y = np.array(y)
            # y = clf.predict(feature)#取得預測
            if y == 0:
                comm.send_instruction(scene_info.frame, PlatformAction.NONE)
                print('NONE')
            elif y == 1:
                comm.send_instruction(scene_info.frame, PlatformAction.MOVE_LEFT)
                print('LEFT')
            elif y == 2:
                comm.send_instruction(scene_info.frame, PlatformAction.MOVE_RIGHT)
                print('RIGHT')
