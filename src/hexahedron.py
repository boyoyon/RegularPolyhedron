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
MODEL_SIZE = 0.3

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
KEY_STATE_PRESS_R = 1

KeyState = KEY_STATE_NONE
PrevKeyState = KEY_STATE_NONE

# 方位角、仰角
AZIMUTH = 0.0
ELEVATION = 0.0
ROLL = 0.0

dAZIMUTH = 0.0
dELEVATION = 0.0

# モデル位置
ModelPos = [0.0, 0.0]

# テクスチャー画像
textureImages = []

WIN_WIDTH = 600  # ウィンドウの幅 / Window width
WIN_HEIGHT = 800  # ウィンドウの高さ / Window height
WIN_TITLE = "Hexahedron"  # ウィンドウのタイトル / Window title

TEX_FILE = None
textureIds = []

idxModels = []

frameNo = 1

points = np.empty((8, 3), np.float32)

faces = np.empty((6, 4), np.int32)

colors = None

fInertia = False

textureScale = 256

TexCoords = np.empty((4,2), np.float32)

def rotDeg2D(p, degree):

    rad = np.deg2rad(degree)
    x = p[0]
    y = p[1]

    X = np.cos(rad) * x - np.sin(rad) * y
    Y = np.sin(rad) * x + np.sin(rad) * y

    return (X, Y)

def createColors(num):

    global colors

    colors = np.empty((num, 3), np.int32)

    delta = 360 // num

    s = 200
    v = 200

    for i in range(num):

        h = i * delta

        r, g, b = hsv2rgb(h, s, v)

        colors[i] = (b, g, r)

def createTexture(c, moji):

    c = c.tolist()
    H = W = textureScale
    texture = np.ones((H, W, 3), np.uint8)
    #texture *= 255

    cv2.rectangle(texture, (0, 0), (W - 1, H - 1), c, 3)

    fontFace = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(texture, text = '0', org = (10, 30), fontFace = fontFace, fontScale = 1.0, 
            color = c, thickness = 2, lineType = cv2.LINE_4) 

    cv2.putText(texture, text = '1', org = (220, 30), fontFace = fontFace, fontScale = 1.0, 
            color = c, thickness = 2, lineType = cv2.LINE_4) 

    cv2.putText(texture, text = '2', org = (220, 240), fontFace = fontFace, fontScale = 1.0, 
            color = c, thickness = 2, lineType = cv2.LINE_4) 

    cv2.putText(texture, text = '3', org = (10, 240), fontFace = fontFace, fontScale = 1.0, 
            color = c, thickness = 2, lineType = cv2.LINE_4) 

    cv2.putText(texture, text = moji, org = (105, 140), fontFace = fontFace, fontScale = 2.0, 
            color = c, thickness = 4, lineType = cv2.LINE_4) 

    #cv2.imshow('texture', texture)
    #cv2.waitKey(0)
    #cv2.destroyWindow('texture')

    return texture

def createHexahedron(size):

    global points

    points[0] = ( size/2, size/2, size/2)
    points[1] = ( size/2, size/2,-size/2)
    points[2] = (-size/2, size/2,-size/2)
    points[3] = (-size/2, size/2, size/2)
    points[4] = (-size/2,-size/2, size/2)
    points[5] = ( size/2,-size/2, size/2)
    points[6] = ( size/2,-size/2,-size/2)
    points[7] = (-size/2,-size/2,-size/2)

    TexCoords[0] = (1.0, 0.0)
    TexCoords[1] = (0.0, 0.0)
    TexCoords[2] = (0.0, 1.0)
    TexCoords[3] = (1.0, 1.0)

    faces[0] = ( 0, 5, 6, 1)
    faces[1] = ( 1, 6, 7, 2)
    faces[2] = ( 4, 7, 6, 5)
    faces[3] = ( 3, 4, 5, 0)
    faces[4] = ( 2, 3, 0, 1)
    faces[5] = ( 2, 7, 4, 3)

def createFace(i):
    

    glBegin(GL_POLYGON)

    idx0 = faces[i][0]
    idx1 = faces[i][1]
    idx2 = faces[i][2]
    idx3 = faces[i][3]

    glVertex3fv(points[idx0])
    glTexCoord2fv(TexCoords[0])

    glVertex3fv(points[idx1])
    glTexCoord2fv(TexCoords[1])

    glVertex3fv(points[idx2])
    glTexCoord2fv(TexCoords[2])

    glVertex3fv(points[idx3])
    glTexCoord2fv(TexCoords[3])
    
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

    createHexahedron(MODEL_SIZE)

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
        AZIMUTH -= dAZIMUTH
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
            AZIMUTH -= dAZIMUTH
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

    global textureImage

    argv = sys.argv
    argc = len(argv)

    print('python %s' % argv[0])
    print('python %s <texture image_1>' % argv[0])
    print('python %s <texture image_1> <texture image_2>' % argv[0])
    print(':')
    print('python %s <texture image_1> <texture image_2> ... <texture image_6>' % argv[0])
   
    createColors(6)

    for i in range(6):

        if argc > 1:

            nrImages = argc - 1
            idx = (i % nrImages) + 1
            texture = cv2.imread(argv[idx])
            texture = cv2.cvtColor(texture, cv2.COLOR_BGR2RGB)
            texture = prescale(texture)

        else:

            moji = '%d' % i
            texture = createTexture(colors[i], moji)
        
    
        texture = cv2.flip(texture, -1)
        textureImages.append(texture)

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
