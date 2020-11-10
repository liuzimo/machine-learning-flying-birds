#coding:utf-8
import random
import PATH
import cv2
from itertools import cycle

FPS = 30
ScreenWidth  = 288
ScreenHeight = 512

Image = PATH.load()
Screen = Image['background']
PipGapSize = 100 # 管道障碍物中间间隔的长度设置为100
Basey = 400 # 基底线所在的位置

index = random.randint(50, 250) #在gapYs中随机选取
PlayerWidth = Image['player'][0].shape[1]
PlayerHeight = Image['player'][0].shape[0]
PipeWidth = Image['pipe'][0].shape[1]
PipeHeight = Image['pipe'][0].shape[0]
BackgroundHeight = Image['background'].shape[0]

PlayerIndex = cycle([0, 1, 2, 1])

class Bird:
    def __init__(self):
        self.playerx = int(ScreenWidth * 0.2)
        self.playery = int((ScreenHeight - PlayerHeight) / 2)
        self.playerVy = 0
        self.playerMaxVy = 8 # 小鸟最快的下落速度是每帧8格
        self.playerMinVy = -6 # 小鸟最快的上升速度是每帧6格
        self.playerAccY = 1 # 小鸟的下落加速度
        self.playerFlapAcc = -8 #在小鸟扇动翅膀时的速度
        self.playerFlapped = False # 标识符，小鸟扇动翅膀时为True
        self.score = self.playerIndex = self.loopIter = 0
        self.action = [0,0]
        self.reward = 0.1
        self.endpoint = False
        self.screen = None
        self.los=0

class GameState:
    def __init__(self):
        self.basex = 0
        self.baseUpline = Image['base'].shape[0] - BackgroundHeight
        self.pipeVx = -4 # 管道障碍物在x轴方向上的移动速度是每帧4格，向x轴负方向移动。就是向左
        self.interval_n =0 #间隔加速度
        self.interval =30 #间隔
        self.baseground=None #基础画面
        self.showground=None #显示画面
        self.score = 0#通用分数
        self.playerx = int(ScreenWidth * 0.2)
        newPipe1 = GetPipe()
        self.upPipes = [
            {'x': ScreenWidth, 'y': newPipe1[0]['y']},
        ]
        self.downPipes = [
            {'x': ScreenWidth, 'y': newPipe1[1]['y']},
        ]

    def pipeline(self):
        self.baseground=Screen.copy()
        # 管道障碍物向左移动时候的速度 
        for uPipe, lPipe in zip(self.upPipes, self.downPipes):
            uPipe['x'] += self.pipeVx
            lPipe['x'] += self.pipeVx

        # 生成新的管道
        self.interval_n +=1
        if self.interval_n ==self.interval:
            newPipe = GetPipe()
            self.upPipes.append({'x': ScreenWidth+PipeWidth-1, 'y': newPipe[0]['y']})
            self.downPipes.append({'x': ScreenWidth+PipeWidth-1, 'y': newPipe[1]['y']})
            self.interval_n=0

        # 当旧的管道移动到屏幕最左边的时候，将其从upPips与downPips中删除
        if self.upPipes[0]['x'] <0:
            self.upPipes.pop(0)
            self.downPipes.pop(0)

        #显示管子
        for uPipe, lPipe in zip(self.upPipes, self.downPipes):
            self.baseground[0:uPipe['y']+PipeHeight,uPipe['x']-PipeWidth if uPipe['x']-PipeWidth > 0 else 0: ScreenWidth if uPipe['x']>ScreenWidth else uPipe['x']]=Image['pipe'][0][abs(uPipe['y']):320,PipeWidth-uPipe['x'] if PipeWidth-uPipe['x'] > 0 else 0:ScreenWidth+PipeWidth-uPipe['x'] if ScreenWidth+PipeWidth-uPipe['x']>0 else PipeWidth]
            self.baseground[lPipe['y']:400,lPipe['x']-PipeWidth if lPipe['x']-PipeWidth > 0 else 0:lPipe['x'] if ScreenWidth >ScreenWidth else lPipe['x']]=Image['pipe'][1][0:400-lPipe['y'],PipeWidth-lPipe['x'] if PipeWidth-lPipe['x'] > 0 else 0:ScreenWidth+PipeWidth-lPipe['x'] if ScreenWidth+PipeWidth-lPipe['x']>0 else PipeWidth]
        
        self.showground = self.baseground.copy()

    def frame_step(self, birds):


        self.pipeline()
      
        # 确定分数，在小鸟的x超过管道右边界的x时，获得的分数+1
        if self.upPipes[0]['x'] <self.playerx  < self.upPipes[0]['x'] + 4:
            self.score += 1

        for bird in birds:
            if bird.endpoint:
                continue
            bird.screen =  self.baseground.copy()
            bird.reward = 0.1
            bird.endpoint = False
            # 0：小鸟不做任务事    1：小鸟扇动翅膀
            if bird.action[1] == 1:
                if bird.playery > -2 * PlayerHeight:
                    bird.playerVy = bird.playerFlapAcc
                    bird.playerFlapped = True


            if (bird.loopIter + 1) % 3 == 0:
                bird.playerIndex = next(PlayerIndex)
            bird.loopIter = (bird.loopIter + 1) % 30
            self.basex = -((-self.basex + 100) % self.baseUpline)

            # 小鸟在y轴上的移动方式
            if bird.playerVy < bird.playerMaxVy and not bird.playerFlapped:
                bird.playerVy += bird.playerAccY
            if bird.playerFlapped:
                bird.playerFlapped = False
            bird.playery += min(bird.playerVy, Basey - bird.playery - PlayerHeight)
            if bird.playery < 0:
                bird.playery = 0



            #显示小鸟
            bird.playery = int(bird.playery)
            self.showground[bird.playery:PlayerHeight+bird.playery,bird.playerx:bird.playerx+PlayerWidth]  = addpng(self.showground[bird.playery:PlayerHeight+bird.playery,bird.playerx:bird.playerx+PlayerWidth],Image['player'][bird.playerIndex]) 
            bird.screen[bird.playery:PlayerHeight+bird.playery,bird.playerx:bird.playerx+PlayerWidth]  = addpng(bird.screen[bird.playery:PlayerHeight+bird.playery,bird.playerx:bird.playerx+PlayerWidth],Image['player'][bird.playerIndex]) 
            cv2.putText(self.showground, str(bird.flag), (bird.playerx+6, bird.playery), cv2.FONT_HERSHEY_SIMPLEX, 1, bird.color, 2)

            # cv2.imshow("1",bird.screen)

            # 碰撞检测。发生碰撞时结束游戏，并马上开始新一局的游戏
            Crash = CrashHappen({'x': bird.playerx, 'y': bird.playery,
                                'index': bird.playerIndex},
                                self.upPipes, self.downPipes)
            if Crash:
                bird.score = self.score
                bird.endpoint = True
                print(str(bird.flag)+"号小鸟死亡 得分 "+str(bird.score)+"， 对应模型："+bird.path)
                # self.__init__()
                bird.reward = -1

                #判断获胜
                birds[0].los +=1
                if birds[0].los==len(birds):
                    print("----------竞赛结束----------")
                    print(str(bird.flag)+"号小鸟获胜 得分 "+str(self.score)+"， 对应模型："+bird.path)
                    birds[0].end=True


        self.showground = showScore(self.score,self.showground)
        # 游戏界面的显示
        cv2.imshow("Screen",self.showground)
        # 手动结束
        if cv2.waitKey(1) & 0xFF == 27:
            print("----------竞赛结束----------")
            for bird in birds:
                if not bird.endpoint:
                    print(str(bird.flag)+"号小鸟得分 "+str(self.score)+"， 对应模型："+bird.path)
            birds[0].end=True
        return birds

def addpng(img,png):
    img = cv2.split(img)
    png = cv2.split(png)
    for i in range(3):
        img[i] = img[i]*(255.0  - png[3])/255 + png[i]*(png[3]/255)
    return cv2.merge(img)

def showScore(score,sc):
    scorenum = [int(x) for x in list(str(score))]
    Total = 0

    for num in scorenum:
        Total += Image['numbers'][num].shape[1]

    Xoffset = int((ScreenWidth - Total) / 2)

    for num in scorenum:
        # Screen.blit(Image['numbers'][num], (Xoffset, ScreenHeight * 0.1))
        sc[int(ScreenHeight * 0.1):int(ScreenHeight * 0.1)+Image['numbers'][num].shape[0],Xoffset:Xoffset+Image['numbers'][num].shape[1]]=addpng (sc[int(ScreenHeight * 0.1):int(ScreenHeight * 0.1)+Image['numbers'][num].shape[0],Xoffset:Xoffset+Image['numbers'][num].shape[1]],Image['numbers'][num])
        Xoffset += Image['numbers'][num].shape[1]
    return sc
def GetPipe():
    gapYs = [20, 30, 40, 50, 60, 70, 80, 90] # 管道间隙的上边所在的y轴的值
    index = random.randint(0, len(gapYs)-1) #在gapYs中随机选取
    gapY = gapYs[index]

    gapY += int(Basey * 0.2)
    pipeX = ScreenWidth + 10

    return [
        {'x': pipeX, 'y': gapY - PipeHeight},  # 上管道的起始坐标值
        {'x': pipeX, 'y': gapY + PipGapSize},  # 下管道的起始坐标值
    ]

def CrashHappen(player, upPipes, downPipes):
    pi = player['index']
    player['w'] = Image['player'][0].shape[1]
    player['h'] = Image['player'][0].shape[0]

    # 判断小鸟是否与地面相碰撞
    if player['y'] + player['h'] >= Basey - 1:
        return True
    else:

        for uPipe, lPipe in zip(upPipes, downPipes):
            # 上下管道的矩形位置和长宽数据
            if uPipe['x']-PipeWidth < player['x']+PlayerWidth/4*3 and player['x']+PlayerWidth/4<uPipe['x']:
                if player['y']+PlayerHeight/4 <uPipe['y']+PipeHeight or lPipe['y'] <player['y'] +PlayerHeight/4*3:
                    return True

    return False

if __name__ == '__main__':
    game = GameState()
    birds = []
    for i in range(2):
       birds.append(Bird())

    for _ in range(100):
        birds[1].action[1]=1

        birds = game.frame_step(birds)