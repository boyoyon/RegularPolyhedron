import copy, cv2, os, sys
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
KEY_H = 72
KEY_V = 86
KEY_I = 73
KEY_T = 84

KEY_0 = 48
KEY_1 = 49
KEY_2 = 50
KEY_3 = 51
KEY_4 = 52

KEY_ESC = 1

LOD = 0 # Level of Detail
PrevLOD = -1

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
textureImage = None

TEXTURE_MODE_KALEIDO = 0
TEXTURE_MODE_GLOBE = 1

TextureMode = TEXTURE_MODE_KALEIDO
PrevTextureMode = -1

WIN_WIDTH = 600  # ウィンドウの幅 / Window width
WIN_HEIGHT = 800  # ウィンドウの高さ / Window height
WIN_TITLE = "Icosahedron_Subdivision"  # ウィンドウのタイトル / Window title

textureId = 0

idxModel = None

frameNo = 1

points = np.empty((12+2, 3), np.float32)

fInertia = False

running_state = True 

def createTexture():

    color = (255, 255, 255)
    thickness = 20

    W = 512
    H = int(np.sqrt(3) * W / 2)

    texture = np.zeros((H, W, 3), np.uint8)
    #texture = np.ones((H, W, 3), np.uint8)
    #texture *= 255

    texture = cv2.line(texture, (W//2, 0), (W-1, H-1), color, thickness)
    texture = cv2.line(texture, (W//2, 0), (0, H-1), color, thickness)
    texture = cv2.line(texture, (0, H-1), (W-1, H-1), color, thickness)

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

    # 正二十面体の頂点だけだと、テクスチャー座標がラップアラウンドするので
    # あらかじめ分割しておく
   
    R = np.linalg.norm(points[0])

    points[12] = (points[0] + points[1]) / 2
    points[12] = scale_vector(points[12], R)

    points[13] = (points[2] + points[3]) / 2
    points[13] = scale_vector(points[13], R)

def scale_vector(p, R):

    r = np.linalg.norm(p) + 0.000001
    P = [p[0] * R / r, p[1] * R / r, p[2] * R /r]
    return P

def _subdivision(p0, p1, p2):

    R = np.linalg.norm(p0)

    p01 = ((p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2, (p0[2] + p1[2]) / 2)
    p01 = scale_vector(p01, R)

    p12 = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2, (p1[2] + p2[2]) / 2)
    p12 = scale_vector(p12, R)

    p20 = ((p2[0] + p0[0]) / 2, (p2[1] + p0[1]) / 2, (p2[2] + p0[2]) / 2)
    p20 = scale_vector(p20, R)

    return p01, p12, p20

def subdivision(TrianglesIn, TrianglesOut):

    for Triangle in TrianglesIn:

        p0 = Triangle[0]
        p1 = Triangle[1]
        p2 = Triangle[2]

        p01, p12, p20 = _subdivision(p0, p1, p2)

        TrianglesOut.append((p01, p12, p20))
        TrianglesOut.append((p0, p01, p20))
        TrianglesOut.append((p01, p1, p12))
        TrianglesOut.append((p20, p12, p2))

def createFace():

    faces = []

    if TextureMode == TEXTURE_MODE_KALEIDO:
        faces.append(( 0, 1, 4)) # 1: 0-1-4
        faces.append(( 0, 7, 1)) # 2: 0-7-1
        faces.append(( 2, 3, 5)) # 3: 2-3-5
        faces.append(( 2, 6, 3)) # 4: 2-6-3
    
    else:

        faces.append(( 0,12, 4)) # 1-1:
        faces.append((12, 1, 4)) # 1-2:

        faces.append(( 0, 7,12)) # 2-1:
        faces.append((12, 7, 1)) # 2-2:

        faces.append(( 2,13, 5)) # 3-1
        faces.append((13, 3, 5)) # 3-2

        faces.append(( 2, 6, 13)) # 4-1
        faces.append((13, 6,  3)) # 4-2 

    faces.append(( 4, 5, 8)) # 5: 4-5-8
    faces.append(( 4,11, 5)) # 6: 4-11-5
    faces.append(( 6,10, 7)) # 7: 6-10-7
    faces.append(( 6, 7, 9)) # 8: 6-7-9

    faces.append(( 0, 8, 9)) # 9: 0-8-9
    faces.append(( 3, 9, 8)) #10: 3-9-8
    faces.append(( 2,11,10)) #11: 2-11-10
    faces.append(( 1,10,11)) #12: 1-10-11

    faces.append(( 0, 4, 8)) #13: 0-4-8
    faces.append(( 0, 9, 7)) #14: 0-9-7
    faces.append(( 1,11, 4)) #15: 1-11-4
    faces.append(( 1, 7,10)) #16: 1-7-10

    faces.append(( 2, 5,11)) #17: 2-5-11
    faces.append(( 2,10, 6)) #18: 2-10-6
    faces.append(( 3, 8, 5)) #19: 3-8-5
    faces.append(( 3, 6, 9)) #20: 3-6-9

    TrianglesIn = []
    TrianglesOut0 = []
    TrianglesOut1 = []
    TrianglesOut2 = []
    TrianglesOut3 = []

    for i in range(len(faces)):
    
        idx0 = faces[i][0]
        idx1 = faces[i][1]
        idx2 = faces[i][2]

        p0 = points[idx0]
        p1 = points[idx1]
        p2 = points[idx2]

        TrianglesIn.append((p0, p1, p2))

    subdivision(TrianglesIn, TrianglesOut0)
    subdivision(TrianglesOut0, TrianglesOut1)
    subdivision(TrianglesOut1, TrianglesOut2)
    subdivision(TrianglesOut2, TrianglesOut3)

    if LOD == 0:
        TrianglesList = TrianglesIn
    elif LOD == 1:
        TrianglesList = TrianglesOut0
    elif LOD == 2:
        TrianglesList = TrianglesOut1
    elif LOD == 3:
        TrianglesList = TrianglesOut2
    else:
        TrianglesList = TrianglesOut3

    glBegin(GL_TRIANGLES)

    if TextureMode == TEXTURE_MODE_KALEIDO:

        for Triangle in TrianglesList:

            p0 = Triangle[0]
            p1 = Triangle[1]
            p2 = Triangle[2]
          
            glTexCoord2f(0.5, 0.0)
            glVertex3f(p0[0], p0[1], p0[2])
            
            glTexCoord2f(0.0, 1.0)
            glVertex3f(p1[0], p1[1], p1[2])
            
            glTexCoord2f(1.0, 1.0)
            glVertex3f(p2[0], p2[1], p2[2])

    else: # TEXTURE_MODE_GLOBE

        axisX = (1.0, 0.0, 0.0)
        axisY = (0.0, 1.0, 0.0)
    
        for Triangle in TrianglesList:
    
            p0 = Triangle[0]
            p1 = Triangle[1]
            p2 = Triangle[2]
          
            px = copy.copy(p0)
            px[1] = 0.0
    
            unitV = scale_vector(p0, 1.0)
            elevation = np.arccos(np.dot(unitV, axisY))
            unitV = scale_vector(px, 1.0)
            azimuth = np.arccos(np.dot(unitV, axisX))
            
            if p0[2] < 0:
                texCoordX0 = 0.5 + azimuth / (np.pi * 2)
            else:
                texCoordX0 = 0.5 - azimuth / (np.pi * 2)
           
            texCoordY0 = elevation / np.pi
           
            px = copy.copy(p1)
            px[1] = 0.0
    
            unitV = scale_vector(p1, 1.0)
            elevation = np.arccos(np.dot(unitV, axisY))
            unitV = scale_vector(px, 1.0)
            azimuth = np.arccos(np.dot(unitV, axisX))
            
            if p1[2] < 0:
                texCoordX1 = 0.5 + azimuth / (np.pi * 2)
            else:
                texCoordX1 = 0.5 - azimuth / (np.pi * 2)
           
            texCoordY1 = elevation / np.pi
           
            px = copy.copy(p2)
            px[1] = 0.0
    
            unitV = scale_vector(p2, 1.0)
            elevation = np.arccos(np.dot(unitV, axisY))
            unitV = scale_vector(px, 1.0)
            azimuth = np.arccos(np.dot(unitV, axisX))
            
            if p2[2] < 0:
                texCoordX2 = 0.5 + azimuth / (np.pi * 2)
            else:
                texCoordX2 = 0.5 - azimuth / (np.pi * 2)
    
            texCoordY2 = elevation / np.pi
                
            if texCoordX2 < 0.5 and texCoordX1 < 0.5 and texCoordX0 >= 0.5:
                texCoordX2 = 1.0 - texCoordX2
                texCoordX1 = 1.0 - texCoordX1
            elif texCoordX2 < 0.5 and texCoordX1 >= 0.5 and texCoordX0 < 0.5:
                texCoordX2 = 1.0 - texCoordX2
                texCoordX0 = 1.0 - texCoordX0
            elif texCoordX2 < 0.5 and texCoordX1 >= 0.5 and texCoordX0 >= 0.5:
                texCoordX2 = 1.0 - texCoordX2
            elif texCoordX2 >= 0.5 and texCoordX1 < 0.5 and texCoordX0 < 0.5:
                texCoordX1 = 1.0 - texCoordX1
                texCoordX0 = 1.0 - texCoordX0
            elif texCoordX2 >= 0.5 and texCoordX1 < 0.5 and texCoordX0 >= 0.5:
                texCoordX1 = 1.0 - texCoordX1
            elif texCoordX2 >= 0.5 and texCoordX1 >= 0.5 and texCoordX0 < 0.5:
                texCooedX0 = 1.0 - texCoordX0
                
            glTexCoord2f(texCoordX0, texCoordY0)
            glVertex3f(p0[0], p0[1], p0[2])
            
            glTexCoord2f(texCoordX1, texCoordY1)
            glVertex3f(p1[0], p1[1], p1[2])
            
            glTexCoord2f(texCoordX2, texCoordY2)
            glVertex3f(p2[0], p2[1], p2[2])

    glEnd()

# OpenGLの初期化関数
def initializeGL():
    global textureId, idxModel

    # 背景色の設定 (黒)
    glClearColor(0.0, 0.0, 0.0, 1.0)

    # 深度テストの有効化
    glEnable(GL_DEPTH_TEST)

    #glEnable(GL_CULL_FACE)

    #glEnable(GL_LIGHT0)
    #glLightfv(GL_LIGHT0, GL_POSITION, (1.0, 1.0, 1.0, 1.0))

    # テクスチャの有効化
    glEnable(GL_TEXTURE_2D)

    # テクスチャの設定

    image = textureImage

    texHeight, texWidth, _ = image.shape

    # テクスチャの生成と有効化
    textureId = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, textureId)

    gluBuild2DMipmaps(GL_TEXTURE_2D, GL_RGB8, texWidth, texHeight, GL_RGB, GL_UNSIGNED_BYTE, image.tobytes())

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)

    # テクスチャ境界の折り返し設定
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)

    # テクスチャの無効化
    glBindTexture(GL_TEXTURE_2D, 0)

    createIcosahedron(MODEL_SIZE)

    idxModel = glGenLists(1)
    glNewList(idxModel, GL_COMPILE)
    createFace()
    glEndList()

# OpenGLの描画関数
def paintGL():

    global PrevLOD, PrevTextureMode

    if WIN_HEIGHT > 0:

        global PrevKeyState, idxModel
    
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
    
        # 平面の描画
        glBindTexture(GL_TEXTURE_2D, textureId)  # テクスチャの有効化
    
        if LOD != PrevLOD or TextureMode != PrevTextureMode:

            createIcosahedron(MODEL_SIZE)
            glDeleteLists(idxModel, 1)
            idxModel = glGenLists(1)
            glNewList(idxModel, GL_COMPILE)
            createFace()
            glEndList()
    
            PrevLOD = LOD
            PrevTextureMode = TextureMode

        PrevKeyState = KeyState
    
        glPushMatrix()
        glScalef(Scale, Scale, Scale)
        glTranslatef(ModelPos[0], ModelPos[1], 0.0)
        glRotatef(ELEVATION, 1.0, 0.0, 0.0)
        glRotatef(AZIMUTH, 0.0, 1.0, 0.0)
        glRotatef(ROLL, 0.0, 0.0, 1.0)
    
        glCallList(idxModel)
    
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
    global AZIMUTH, ELEVATION, dAZIMUTH, dELEVATION, KeyState, idxModel, fInertia, LOD, TextureMode
    global running_state

    if scancode == KEY_ESC:
        running_state = False

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

    if key == KEY_0:
        LOD = 0

    if key == KEY_1:
        LOD = 1

    if key == KEY_2:
        LOD = 2

    if key == KEY_3:
        LOD = 3

    if key == KEY_4:
        LOD = 4

    # 慣性モードのトグル

    if key == KEY_I and action == 1:
        fInertia = not fInertia

    if key == KEY_T and action == 1:
        if TextureMode == TEXTURE_MODE_KALEIDO:
            TextureMode = TEXTURE_MODE_GLOBE
        else:
            TextureMode = TEXTURE_MODE_KALEIDO

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
    global Scale, ROLL

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

    global textureImage, TextureMode

    argv = sys.argv
    argc = len(argv)

    if argc > 1:
        img = cv2.imread(argv[1], cv2.IMREAD_COLOR)
        img = prescale(img)
        img = cv2.flip(img, -1)
        textureImage = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        TextureMode = TEXTURE_MODE_GLOBE

    else:
        textureImage = createTexture()

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
    while glfw.window_should_close(window) == glfw.FALSE and running_state == True:
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
