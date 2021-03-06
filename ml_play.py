
import pickle
import numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn import metrics
import os.path as path
import games.arkanoid.communication as comm
from games.arkanoid.communication import ( \
    SceneInfo, GameStatus, PlatformAction
)


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
    filename = path.join(path.dirname(__file__),"save/clf_KMeans_BallAndDirection.pickle")
    # filename = path.join(path.dirname(__file__),"save\test_model_01.pickle")
    # filename = path.join(path.dirname(__file__),"\games\arkanoid\ml\save\clf_KMeans_BallAndDirection.pickle")
    with open(filename, 'rb') as file:
        
        clf = pickle.load(file)

    # 2. Inform the game process that ml process is ready before start the loop.
    comm.ml_ready()
    
    s = [93,93]
    def get_direction(ball_x,ball_y,ball_pre_x,ball_pre_y):
        VectorX = ball_x - ball_pre_x
        VectorY = ball_y - ball_pre_y
        if(VectorX>=0 and VectorY>=0):
            return 0
        elif(VectorX>0 and VectorY<0):
            return 1
        elif(VectorX<0 and VectorY>0):
            return 2
        elif(VectorX<0 and VectorY<0):
            return 3
        

    # 3. Start an endless loop.
    while True:
        # 3.1. Receive the scene information sent from the game process.
        scene_info = comm.get_scene_info()#0326 取得資料
        feature = []
        feature.append(scene_info.ball[0])
        feature.append(scene_info.ball[1])
        feature.append(scene_info.platform[0])
        
        feature.append(get_direction(feature[0],feature[1],s[0],s[1]))
        s = [feature[0], feature[1]]
        feature = np.array(feature)
        feature = feature.reshape((-1,4))
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
            
            out = clf(feature, len(feature))
            y = label_to_move(out)
            y = np.array(y)
            # y = clf.predict(feature)#取得預測
            print('movement :', y)
            if y == 0:
                comm.send_instruction(scene_info.frame, PlatformAction.NONE)
                print('NONE')
            elif y == 1:
                comm.send_instruction(scene_info.frame, PlatformAction.MOVE_LEFT)
                print('LEFT')
            elif y == 2:
                comm.send_instruction(scene_info.frame, PlatformAction.MOVE_RIGHT)
                print('RIGHT')
