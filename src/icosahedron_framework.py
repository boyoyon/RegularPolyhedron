import cv2, os, sys
import glfw
import numpy as np
from PIL import Image
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
from hsv2rgb import *

#画像幅をALIGNピクセルの倍数にcropする
ALIGN = 4

# マウスドラッグ中かどうか
isDragging = False

# マウスのクリック位置
oldPos = [0, 0]
newPos = [0, 0]

# 操作の種類
MODE_NONE = 0x00
MODE_TRANSLATE = 0x01
MODE_ROTATE = 0x02
MODE_SCALE = 0x04

# マウス移動量と回転、平行移動の倍率
ROTATE_SCALE = 10.0
TRANSLATE_SCALE = 500.0

# 座標変換のための変数
Mode = MODE_NONE
Scale = 5.0

#MODEL_SIZE = 1.0
MODEL_SIZE = 0.1


# スキャンコード定義
SCANCODE_LEFT  = 331
SCANCODE_RIGHT = 333
SCANCODE_UP    = 328
SCANCODE_DOWN  = 336

# キーコード定義
KEY_R = 82
KEY_S = 83
KEY_I = 73

KEY_STATE_NONE = 0
KEY_STATE_PRESS_R = 4

KeyState = KEY_STATE_NONE
PrevKeyState = KEY_STATE_NONE

# 方位角、仰角
#AZIMUTH = 0.0
AZIMUTH = 315
#ELEVATION = 0.0
ELEVATION = -135
ROLL = 0.0

"""
PrevAzimuth = -1
PrevElevation = -1
PrevRoll = -1
"""

dAZIMUTH = 0.0
dELEVATION = 0.0

# モデル位置
ModelPos = [0.0, 0.0]

# テクスチャー画像
textureImages = []

WIN_WIDTH = 600  # ウィンドウの幅 / Window width
WIN_HEIGHT = 800  # ウィンドウの高さ / Window height
WIN_TITLE = "Icosahedron"  # ウィンドウのタイトル / Window title

textureIds = []

idxModels = []

frameNo = 1

points = np.empty((12, 3), np.float32)

fInertia = False

def createTexture(c, prefix):

    W = H = 256
    fontFace = cv2.FONT_HERSHEY_SIMPLEX

    texture = np.ones((H, W, 3), np.uint8)
    #texture *= 255

    cv2.rectangle(texture, (0, 0), (W-1, H-1), c, 5)
    cv2.line(texture, (W//2, 0), (W//2, H-1), c, 3)
    cv2.putText(texture, text = '%s0' % prefix, org = (10,30), fontFace = fontFace, fontScale = 1.0, 
            color = c, thickness = 2, lineType = cv2.LINE_4) 

    cv2.putText(texture, text = '%s1' % prefix, org = (10,240), fontFace = fontFace, fontScale = 1.0, 
            color = c, thickness = 2, lineType = cv2.LINE_4) 

    cv2.putText(texture, text = '%s2' % prefix, org = (210,240), fontFace = fontFace, fontScale = 1.0, 
            color = c, thickness = 2, lineType = cv2.LINE_4) 
    
    cv2.putText(texture, text = '%s3' % prefix, org = (210,30), fontFace = fontFace, fontScale = 1.0, 
            color = c, thickness = 2, lineType = cv2.LINE_4) 

    return texture

def createIcosahedron(size):

    global points

    t = (1+np.sqrt(5)) / 2 * size

    points[0] = ( t, 0,  size)
    points[1] = ( t, 0, -size)
    points[2] = (-t, 0, -size)
    points[3] = (-t, 0,  size)
    
    points[4] = ( size,  t, 0)
    points[5] = (-size,  t, 0)
    points[6] = (-size, -t, 0)
    points[7] = ( size, -t, 0)
    
    points[8] = (0,  size,  t)
    points[9] = (0, -size,  t)
    points[10] = (0, -size, -t)
    points[11] = (0,  size, -t)

def createFace(idx):

    glBegin(GL_TRIANGLES)

    if idx == 0:
    
        # x-z plane (upper)
        glTexCoord2f(0.0, 0.0)
        glVertex3f(points[0][0], points[0][1], points[0][2])
        
        glTexCoord2f(1.0, 0.0)
        glVertex3f(points[3][0], points[3][1], points[3][2])
    
        glTexCoord2f(0.0, 1.0)
        glVertex3f(points[1][0], points[1][1], points[1][2])
    
        # x-z plane (lower)
        glTexCoord2f(1.0, 0.0)
        glVertex3f(points[3][0], points[3][1], points[3][2])
        
        glTexCoord2f(1.0, 1.0)
        glVertex3f(points[2][0], points[2][1], points[2][2])
    
        glTexCoord2f(0.0, 1.0)
        glVertex3f(points[1][0], points[1][1], points[1][2])
        
    elif idx == 1:

        # y-x plane (upper)
        glTexCoord2f(0.0, 0.0)
        glVertex3f(points[4][0], points[4][1], points[4][2])
        
        glTexCoord2f(1.0, 0.0)
        glVertex3f(points[7][0], points[7][1], points[7][2])
    
        glTexCoord2f(0.0, 1.0)
        glVertex3f(points[5][0], points[5][1], points[5][2])
    
        # y-x plane (lower)
        glTexCoord2f(1.0, 0.0)
        glVertex3f(points[7][0], points[7][1], points[7][2])
        
        glTexCoord2f(1.0, 1.0)
        glVertex3f(points[6][0], points[6][1], points[6][2])
    
        glTexCoord2f(0.0, 1.0)
        glVertex3f(points[5][0], points[5][1], points[5][2])
        
    else:

        # z-y plane (upper)
        glTexCoord2f(0.0, 0.0)
        glVertex3f(points[8][0], points[8][1], points[8][2])
        
        glTexCoord2f(1.0, 0.0)
        glVertex3f(points[11][0], points[11][1], points[11][2])
    
        glTexCoord2f(0.0, 1.0)
        glVertex3f(points[9][0], points[9][1], points[9][2])
    
        glTexCoord2f(1.0, 0.0)
        glVertex3f(points[11][0], points[11][1], points[11][2])
        
        glTexCoord2f(1.0, 1.0)
        glVertex3f(points[10][0], points[10][1], points[10][2])
    
        glTexCoord2f(0.0, 1.0)
        glVertex3f(points[9][0], points[9][1], points[9][2])
    
    glEnd()

# OpenGLの初期化関数
def initializeGL():
    
    global textureIds, idxModels

    # 背景色の設定 (黒)
    glClearColor(0.0, 0.0, 0.0, 1.0)

    # 深度テストの有効化
    glEnable(GL_DEPTH_TEST)

    #glEnable(GL_CULL_FACE)

    #glEnable(GL_LIGHT0)
    #glLightfv(GL_LIGHT0, GL_POSITION, (1.0, 1.0, 1.0, 1.0))

    # テクスチャの有効化
    glEnable(GL_TEXTURE_2D)

    createIcosahedron(MODEL_SIZE)

    for i in range(len(textureImages)):

        # テクスチャの設定
        image = textureImages[i]
    
        texHeight, texWidth, _ = image.shape
    
        # テクスチャの生成と有効化
        textureIds.append(glGenTextures(1))
        glBindTexture(GL_TEXTURE_2D, textureIds[i])
    
        gluBuild2DMipmaps(GL_TEXTURE_2D, GL_RGB8, texWidth, texHeight, GL_RGB, GL_UNSIGNED_BYTE, image.tobytes())
    
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
    
        # テクスチャ境界の折り返し設定
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
    
        # テクスチャの無効化
        glBindTexture(GL_TEXTURE_2D, 0)
    
        idxModels.append(glGenLists(1))
    
        glNewList(idxModels[i], GL_COMPILE)
        glBindTexture(GL_TEXTURE_2D, textureIds[i])  # テクスチャの有効化
        createFace(i)
        glEndList()

# OpenGLの描画関数
def paintGL():

    global idxModels, PrevKeyState, PrevAzimuth, PrevElevation, PrevRoll

    if WIN_HEIGHT > 0:

        # 背景色と深度値のクリア
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    
        # 投影変換行列
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        #gluPerspective(10.0, WIN_WIDTH / WIN_HEIGHT, 1.0, 100.0)
        gluPerspective(45.0, WIN_WIDTH / WIN_HEIGHT, 1.0, 100.0)
    
    
        # モデルビュー行列
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        
        gluLookAt(0.0, 0.0, -5.0,   # 視点の位置
            0.0, 0.0, 0.0,   # 見ている先
            0.0, -1.0, 0.0)  # 視界の上方向
      
        #gluLookAt(0.0, 0.0, 0.0,   # 視点の位置
        #    0.0, 0.0, -1.0,   # 見ている先
        #    0.0, -1.0, 0.0)  # 視界の上方向
    
        PrevKeyState = KeyState
    
        glPushMatrix()
        glScalef(Scale, Scale, Scale)
        glTranslatef(ModelPos[0], ModelPos[1], 0.0)
        glRotatef(ELEVATION, 1.0, 0.0, 0.0)
        glRotatef(AZIMUTH, 0.0, 1.0, 0.0)
        glRotatef(ROLL, 0.0, 0.0, 1.0)

        """
        if AZIMUTH != PrevAzimuth or ELEVATION != PrevElevation or ROLL != PrevRoll:
            print(AZIMUTH, ELEVATION, ROLL)
            PrevAzimuth = AZIMUTH
            PrevElevation = ELEVATION
            PrevRoll = ROLL
        """

        if len(idxModels) > 0:
            for i in range(len(idxModels)):
                glCallList(idxModels[i])
    
        glPopMatrix()
    
        glBindTexture(GL_TEXTURE_2D, 0)  # テクスチャの無効化


# ウィンドウサイズ変更のコールバック関数
def resizeGL(window, width, height):
    global WIN_WIDTH, WIN_HEIGHT

    # ユーザ管理のウィンドウサイズを変更
    WIN_WIDTH = width
    WIN_HEIGHT = height

    # GLFW管理のウィンドウサイズを変更
    glfw.set_window_size(window, WIN_WIDTH, WIN_HEIGHT)

    # 実際のウィンドウサイズ (ピクセル数) を取得
    renderBufferWidth, renderBufferHeight = glfw.get_framebuffer_size(window)

    # ビューポート変換の更新
    glViewport(0, 0, renderBufferWidth, renderBufferHeight)

# アニメーションのためのアップデート
def animate():
    global AZIMUTH, ELEVATION

    # 慣性モード中は回転し続ける
    if fInertia and not isDragging:
        AZIMUTH += dAZIMUTH
        ELEVATION += dELEVATION

def save_screen():
    global frameNo

    width = WIN_WIDTH
    height = WIN_HEIGHT

    glReadBuffer(GL_FRONT)
    screen_shot = np.zeros((height, width, 3), np.uint8)
    glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, screen_shot.data)
    screen_shot = cv2.cvtColor(screen_shot, cv2.COLOR_RGB2BGR)
    screen_shot = cv2.flip(screen_shot, 0)
    filename = 'screenshot_%04d.png' % frameNo
    cv2.imwrite(filename, screen_shot)
    print('saved %s' % filename)
    frameNo += 1

# キーボードの押し離しを扱うコールバック関数
def keyboardEvent(window, key, scancode, action, mods):
    global AZIMUTH, ELEVATION, dAZIMUTH, dELEVATION, KeyState, idxModel, fInertia 

    # 矢印キー操作

    if scancode == SCANCODE_LEFT:
        dAZIMUTH = 0.1
        AZIMUTH += dAZIMUTH * 10

    if scancode == SCANCODE_RIGHT:
        dAZIMUTH = -0.1
        AZIMUTH += dAZIMUTH * 10

    if scancode == SCANCODE_DOWN:
        dELEVATION = 0.1
        ELEVATION += dELEVATION * 10

    if scancode == SCANCODE_UP:
        dELEVATION = -0.1
        ELEVATION += dELEVATION * 10

    # sキー押下でスクリーンショット
    if key == KEY_S and action == 1: # press, releaseで2回キャプチャーしないように
        save_screen()

    # ホィールモードの選択

    if key == KEY_R:
        if action == glfw.PRESS:
            KeyState = KEY_STATE_PRESS_R
        elif action == 0:
            KeyState = KEY_STATE_NONE

    # 慣性モードのトグル

    if key == KEY_I and action == 1:
        fInertia = not fInertia

# マウスのクリックを処理するコールバック関数
def mouseEvent(window, button, action, mods):
    global isDragging, newPos, oldPos, Mode, fInertia

    # クリックしたボタンで処理を切り替える
    if button == glfw.MOUSE_BUTTON_LEFT:
        Mode = MODE_ROTATE
    
    elif button == glfw.MOUSE_BUTTON_MIDDLE:
        if action == 1:
            fInertia = not fInertia

    elif button == glfw.MOUSE_BUTTON_RIGHT:
        Mode = MODE_TRANSLATE

    # クリックされた位置を取得
    px, py = glfw.get_cursor_pos(window)

    # マウスドラッグの状態を更新
    if action == glfw.PRESS:
        if not isDragging:
            isDragging = True
            oldPos = [px, py]
            newPos = [px, py]
    else:
        isDragging = False
        oldPos = [0, 0]
        newPos = [0, 0]

# マウスの動きを処理するコールバック関数
def motionEvent(window, xpos, ypos):
    global isDragging, newPos, oldPos, AZIMUTH, dAZIMUTH, ELEVATION, dELEVATION, ModelPos

    if isDragging:
        # マウスの現在位置を更新
        newPos = [xpos, ypos]

        dx = newPos[0] - oldPos[0]
        dy = newPos[1] - oldPos[1]
        
        # マウスがあまり動いていない時は処理をしない
        #length = dx * dx + dy * dy
        #if length < 2.0 * 2.0:
        #    return
        #else:
        if Mode == MODE_ROTATE:
            dAZIMUTH = (xpos - oldPos[0]) / ROTATE_SCALE
            dELEVATION = (ypos - oldPos[1]) / ROTATE_SCALE
            AZIMUTH += dAZIMUTH
            ELEVATION += dELEVATION
        elif Mode == MODE_TRANSLATE:
            ModelPos[0] += (xpos - oldPos[0]) / TRANSLATE_SCALE
            ModelPos[1] += (ypos - oldPos[1]) / TRANSLATE_SCALE

        oldPos = [xpos, ypos]

# マウスホイールを処理するコールバック関数
def wheelEvent(window, xoffset, yoffset):
    global Scale, idxModel, ROLL

    if KeyState == KEY_STATE_NONE:
        Scale += yoffset / 10.0

    elif KeyState == KEY_STATE_PRESS_R:
        ROLL += yoffset

# 画像の横幅がALIGNピクセルの倍数になるようにクロップする
# そうしないとうまくテクスチャーマッピングされないため
def prescale(image):
    height, width = image.shape[:2]

    if width % ALIGN != 0:
        WIDTH = width // ALIGN * ALIGN
        startX = (width - WIDTH) // 2
        endX = startX + WIDTH

        dst = np.empty((height, WIDTH, 3), np.uint8)
        dst = image[:, startX:endX]
        return dst

    else:
        return image

def main():

    global textureImages

    argv = sys.argv
    argc = len(argv)

    if argc == 1:
        textureImages.append(createTexture((255, 0, 0), 'x'))
        textureImages.append(createTexture((0, 255, 0), 'y'))
        textureImages.append(createTexture((0, 0, 255), 'z'))

    elif argc == 2:
        img = cv2.imread(argv[1], cv2.IMREAD_COLOR)
        H, W = img.shape[:2]
        if H > W:
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        img = prescale(img)
        
        textureImages.append(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        textureImages.append(createTexture((0, 255, 0), 'y'))
        textureImages.append(createTexture((0, 0, 255), 'z'))
    elif argc == 3:
        img = cv2.imread(argv[1], cv2.IMREAD_COLOR)
        H, W = img.shape[:2]
        if H > W:
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        img = prescale(img)
        textureImages.append(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

        img = cv2.imread(argv[2], cv2.IMREAD_COLOR)
        H, W = img.shape[:2]
        if H > W:
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        img = prescale(img)
            
        textureImages.append(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        
        textureImages.append(createTexture((0, 0, 255), 'z'))
    else:
        img = cv2.imread(argv[1], cv2.IMREAD_COLOR)
        H, W = img.shape[:2]
        if H > W:
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        img = prescale(img)
            
            
        textureImages.append(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

        img = cv2.imread(argv[2], cv2.IMREAD_COLOR)
        H, W = img.shape[:2]
        if H > W:
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        img = prescale(img)
            
            
        textureImages.append(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        
        img = cv2.imread(argv[3], cv2.IMREAD_COLOR)
        H, W = img.shape[:2]
        if H > W:
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        img = prescale(img)
            
        textureImages.append(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        

    # OpenGLを初期化する
    if glfw.init() == glfw.FALSE:
        raise Exception("Failed to initialize OpenGL")

    # Windowの作成
    window = glfw.create_window(WIN_WIDTH, WIN_HEIGHT, WIN_TITLE, None, None)
    if window is None:
        glfw.terminate()
        raise Exception("Failed to create Window")

    # OpenGLの描画対象にWindowを追加
    glfw.make_context_current(window)

    # ウィンドウのリサイズを扱う関数の登録
    glfw.set_window_size_callback(window, resizeGL)

    # キーボードのイベントを処理する関数を登録
    glfw.set_key_callback(window, keyboardEvent)

    # マウスのイベントを処理する関数を登録
    glfw.set_mouse_button_callback(window, mouseEvent)

    # マウスの動きを処理する関数を登録
    glfw.set_cursor_pos_callback(window, motionEvent)

    # マウスホイールを処理する関数を登録
    glfw.set_scroll_callback(window, wheelEvent)
    
    # ユーザ指定の初期化
    initializeGL()

    # メインループ
    while glfw.window_should_close(window) == glfw.FALSE:
        # 描画
        paintGL()

        # アニメーション
        animate()

        # 描画用バッファの切り替え
        glfw.swap_buffers(window)
        glfw.poll_events()

    # 後処理
    glfw.destroy_window(window)
    glfw.terminate()


if __name__ == "__main__":
    main()
