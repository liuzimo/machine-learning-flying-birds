#coding:utf-8
import random
import pygame
import PATH
import string
import numpy as np
from itertools import cycle

FPS = 30
ScreenWidth  = 288
ScreenHeight = 512

pygame.init()
FPSClock = pygame.time.Clock()
Screen = pygame.display.set_mode((ScreenWidth, ScreenHeight))
pygame.display.set_caption('Flappy Bird')

Image, Sounds, Hitmasks = PATH.load()
PipGapSize = 100 # 管道障碍物中间间隔的长度设置为100
Basey = ScreenHeight * 0.79 # 基底线所在的位置

BirdWidth = Image['player'][0].get_width()
BirdHeight = Image['player'][0].get_height()
PipeWidth = Image['pipe'][0].get_width()
PipeHeight = Image['pipe'][0].get_height()
BackgroundWidth = Image['background'].get_width()

PlayerIndex = cycle([0, 1, 2, 1])

player_num = 100


class BirdAgent:
    def __init__(self):
        self.id = random.sample(string.ascii_letters + string.digits, 8)
        self.x = int(ScreenWidth * 0.2)
        self.y = int((ScreenHeight - BirdHeight) / 2)
        self.v_y = 0
        self.max_falling_speed = 10 # 小鸟最快的下落速度是每帧10格
        self.min_rising_speed = -8 # 小鸟最快的上升速度是每帧8格
        self.falling_acc = 6 # 小鸟的下落加速度
        self.flap_speed = -9 #在小鸟扇动翅膀时的速度
        self.isFlapped = False # 标识符，小鸟扇动翅膀时为True
        self.reward = 0
        self.is_alive = True

    def action(self):
        # TODO : select action from DQN
        return random.randint(0, 1)

    def replay_buffer(self):
        # TODO : add replay buffer
        return NotImplementedError

    def network(self):
        #TODO : add DQN
        return NotImplementedError

class Arena:
    def __init__(self):

        self.basex = 0
        self.baseUpline = Image['base'].get_width() - BackgroundWidth
        self.score = self.playerIndex = self.loopIter = 0

        newPipe1 = GetPipe()
        newPipe2 = GetPipe()
        self.upPipes = [
            {'x': ScreenWidth, 'y': newPipe1[0]['y']},
            {'x': ScreenWidth + (ScreenWidth / 2), 'y': newPipe2[0]['y']},
        ]
        self.downPipes = [
            {'x': ScreenWidth, 'y': newPipe1[1]['y']},
            {'x': ScreenWidth + (ScreenWidth / 2), 'y': newPipe2[1]['y']},
        ]

        self.pipeVx = -4 # 管道障碍物在x轴方向上的移动速度是每帧4格，向x轴负方向移动。就是向左

        self.all_birds = []
        for _ in range(player_num):
            bird = BirdAgent()
            bird.x = int(ScreenWidth * random.uniform(0.05,0.4))
            bird.y = int((ScreenHeight - BirdHeight) / 2)
            self.all_birds.append(bird)
        # bird.v_y = 0
        # bird.max_falling_speed = 10 # 小鸟最快的下落速度是每帧10格
        # bird.min_rising_speed = -8 # 小鸟最快的上升速度是每帧8格
        # bird.falling_acc = 1 # 小鸟的下落加速度
        # bird.flap_speed = -9 #在小鸟扇动翅膀时的速度
        # bird.isFlapped = False # 标识符，小鸟扇动翅膀时为True

    def compete(self):
        pygame.event.pump()

        reward = 0.1
        endpoint = False

        # if sum(inputAction) != 1:
        #     raise ValueError('输入错误')

        # 游戏界面的显示
        Screen.blit(Image['background'], (0, 0))

        # 管道障碍物向左移动时候的速度
        for uPipe, lPipe in zip(self.upPipes, self.downPipes):
            uPipe['x'] += self.pipeVx
            lPipe['x'] += self.pipeVx

        # 生成新的管道
        if 0 < self.upPipes[0]['x'] < 5:
            newPipe = GetPipe()
            self.upPipes.append(newPipe[0])
            self.downPipes.append(newPipe[1])

        # 当旧的管道移动到屏幕最左边的时候，将其从upPips与downPips中删除
        if self.upPipes[0]['x'] < -PipeWidth:
            self.upPipes.pop(0)
            self.downPipes.pop(0)

        for uPipe, lPipe in zip(self.upPipes, self.downPipes):
            Screen.blit(Image['pipe'][0], (uPipe['x'], uPipe['y']))
            Screen.blit(Image['pipe'][1], (lPipe['x'], lPipe['y']))

        Screen.blit(Image['base'], (self.basex, Basey))

        alive_birds = []
        for _, bird in enumerate(self.all_birds):
            if bird.is_alive:
                alive_birds.append(bird)

        if len(alive_birds) == 0:
            endpoint = True

        for _,bird in enumerate(alive_birds):
            # 0：小鸟不做任务事    1：小鸟扇动翅膀
            if bird.action() == 1:
                if bird.y > -2 * BirdHeight:
                    bird.v_y = bird.flap_speed
                    bird.isFlapped = True

            # 确定分数，在小鸟的x超过管道右边界的x时，获得的分数+1
            playerplace = bird.x + BirdWidth / 2
            for pipe in self.upPipes:
                pipeplace = pipe['x'] + PipeWidth / 2
                if pipeplace <= playerplace < pipeplace + 4:
                    self.score += 1
                    reward = 1

            if (self.loopIter + 1) % 3 == 0:
                self.playerIndex = next(PlayerIndex)
            self.loopIter = (self.loopIter + 1) % 30
            self.basex = -((-self.basex + 100) % self.baseUpline)

            # 小鸟在y轴上的移动方式
            if bird.v_y < bird.max_falling_speed and not bird.isFlapped:
                bird.v_y += bird.falling_acc
            if bird.isFlapped:
                bird.isFlapped = False
            bird.y += min(bird.v_y, Basey - bird.y - BirdHeight)
            if bird.y < 0:
                bird.y = 0

            Screen.blit(Image['player'][self.playerIndex], (bird.x, bird.y))

            Crash = CrashHappen({'x': bird.x, 'y': bird.y, 'index': self.playerIndex}, self.upPipes, self.downPipes)
            if Crash:
                bird.is_alive = False
                bird.reward = self.score


        # # 碰撞检测。发生碰撞时结束游戏，并马上开始新一局的游戏
        # Crash = CrashHappen({'x': bird.x, 'y': bird.y,
        #                      'index': self.playerIndex},
        #                     self.upPipes, self.downPipes)
        # if Crash:
        #     endpoint = True
        #     self.__init__()
        #     reward = -1

        # # 游戏界面的显示
        # Screen.blit(Image['background'], (0,0))
        #
        # for uPipe, lPipe in zip(self.upPipes, self.downPipes):
        #     Screen.blit(Image['pipe'][0], (uPipe['x'], uPipe['y']))
        #     Screen.blit(Image['pipe'][1], (lPipe['x'], lPipe['y']))
        #
        # Screen.blit(Image['base'], (self.basex, Basey))

        showScore(self.score)

        ImageData = pygame.surfarray.array3d(pygame.display.get_surface())
        pygame.display.update()
        FPSClock.tick(FPS)
        return ImageData, reward, endpoint

    def reset(self):
        self.__init__()

def showScore(score):
    scorenum = [int(x) for x in list(str(score))]
    Total = 0

    for num in scorenum:
        Total += Image['numbers'][num].get_width()

    Xoffset = (ScreenWidth - Total) / 2

    for num in scorenum:
        Screen.blit(Image['numbers'][num], (Xoffset, ScreenHeight * 0.1))
        Xoffset += Image['numbers'][num].get_width()

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
    player['w'] = Image['player'][0].get_width()
    player['h'] = Image['player'][0].get_height()

    # 判断小鸟是否与地面相碰撞
    if player['y'] + player['h'] >= Basey - 1:
        return True
    else:

        playerRect = pygame.Rect(player['x'], player['y'],
                      player['w'], player['h'])

        for uPipe, lPipe in zip(upPipes, downPipes):
            # 上下管道的矩形位置和长宽数据
            uPipeRect = pygame.Rect(uPipe['x'], uPipe['y'], PipeWidth, PipeHeight)
            lPipeRect = pygame.Rect(lPipe['x'], lPipe['y'], PipeWidth, PipeHeight)

            pHitMask = Hitmasks['player'][pi]
            uHitmask = Hitmasks['pipe'][0]
            lHitmask = Hitmasks['pipe'][1]

            # 判断小鸟是否回合上下管道发生碰撞
            uCollide = pixelCollision(playerRect, uPipeRect, pHitMask, uHitmask)
            lCollide = pixelCollision(playerRect, lPipeRect, pHitMask, lHitmask)

            if uCollide or lCollide:
                return True

    return False

def pixelCollision(rect1, rect2, hitmask1, hitmask2):
    rect = rect1.clip(rect2)

    if rect.width == 0 or rect.height == 0:
        return False

    x1, y1 = rect.x - rect1.x, rect.y - rect1.y
    x2, y2 = rect.x - rect2.x, rect.y - rect2.y

    for x in range(rect.width):
        for y in range(rect.height):
            if hitmask1[x1+x][y1+y] and hitmask2[x2+x][y2+y]:
                return True
    return False

game = Arena()

for _ in range(1000):
    # action = np.random.randint(0, 1, player_num)
    # print(action)
    state, reward, terminal = game.compete()
    if terminal:
        game.reset()
