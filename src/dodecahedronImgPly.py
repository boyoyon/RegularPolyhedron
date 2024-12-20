import cv2, os, sys
import glfw
import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
from plyfile import PlyData, PlyElement

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
Scale = 1.0

#MODEL_SIZE = 1.0
MODEL_SIZE = 0.05

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
textureImage = None

WIN_WIDTH = 600  # ウィンドウの幅 / Window width
WIN_HEIGHT = 800  # ウィンドウの高さ / Window height
WIN_TITLE = "Dodecahedron"  # ウィンドウのタイトル / Window title

TEX_FILE = None
textureIds = []
textureId = -1

idxModels = []
idxModel = -1

frameNo = 1

points = np.empty((20, 3), np.float32)
faces = np.empty((12, 5), np.int32)


positions = None
texcoords = None
_3d_faces = None

fInertia = False

textureScale = 256

TexCoords = None

colors = None

surfaces = np.empty((12, 5), np.int32)

def createColors(num):

    global colors

    colors = np.empty((num, 3), np.int32)

    step = 256 * 6 // (num + 1)

    idx = 0
    for i in range((num // 6) + 1):

        val = 255 - step * i

        colors[idx] = (val, 0, 0)

        idx += 1
        if idx >= num:
            break

        colors[idx] = (0, val, 0)

        idx += 1
        if idx >= num:
            break

        colors[idx] = (0, 0, val)

        idx += 1
        if idx >= num:
            brak

        colors[idx] = (val * 80 // 100, val * 80 // 100, 0)

        idx += 1
        if idx >= num:
            break

        colors[idx] = (0, val * 80 // 100, val * 80 // 100)

        idx += 1
        if idx >= num:
            break

        colors[idx] = (val * 80 // 100, 0, val * 80 // 100)

        idx += 1
        if idx >= num:
            break

def rotDeg2D(p, degree):

    rad = np.deg2rad(degree)
    x = p[0]
    y = p[1]

    X = np.cos(rad) * x - np.sin(rad) * y
    Y = np.sin(rad) * x + np.sin(rad) * y

    return (X, Y)

def createTexture(c, moji):

    global TexCoords

    color = (int(c[0]), int(c[1]), int(c[2]))

    P = (1.0, 0.0)

    V = []

    for i in range(5):
        V.append(rotDeg2D(P, 72 * i - 18))

    V = np.array(V)

    V[:,0] -= np.min(V[:,0])
    V[:,1] -= np.min(V[:,1])

    V[:,0] *= textureScale / np.max(V[:,0])
    V[:,1] *= textureScale / np.max(V[:,1])

    WIDTH = int(np.max(V[:,0]))
    HEIGHT = int(np.max(V[:,1]))

    texture = np.ones((HEIGHT, WIDTH, 3), np.uint8)
    texture *= 255

    for i in range(5):

        j = (i + 1) % 5

        x0 = int(V[i][0])
        y0 = int(V[i][1])
        x1 = int(V[j][0])
        y1 = int(V[j][1])

        cv2.line(texture, (x0, y0), (x1, y1), color, 3)

    fontFace = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(texture, text = '0', org = (215,120), fontFace = fontFace, fontScale = 1.0, 
            color = color, thickness = 2, lineType = cv2.LINE_4) 

    cv2.putText(texture, text = '1', org = (120,40), fontFace = fontFace, fontScale = 1.0, 
            color = color, thickness = 2, lineType = cv2.LINE_4) 

    cv2.putText(texture, text = '2', org = (15,120), fontFace = fontFace, fontScale = 1.0, 
            color = color, thickness = 2, lineType = cv2.LINE_4) 

    cv2.putText(texture, text = '3', org = (50,240), fontFace = fontFace, fontScale = 1.0, 
            color = color, thickness = 2, lineType = cv2.LINE_4) 

    cv2.putText(texture, text = '4', org = (180,240), fontFace = fontFace, fontScale = 1.0, 
            color = color, thickness = 2, lineType = cv2.LINE_4) 

    cv2.putText(texture, text = moji, org = (105 - (len(moji) - 1) * 20,160), fontFace = fontFace, fontScale = 2.0, 
            color = color, thickness = 6, lineType = cv2.LINE_4) 
    
    #cv2.imshow('texture', texture)
    #cv2.waitKey(0)
    #cv2.destroyWindow('texture')

    TexCoords = V / textureScale
    TexCoords[:,1] = 1.0 - TexCoords[:,1]

    return texture

def createDodecahedron(size):

    global points

    P = (size, 0.0)
    z = (np.sqrt(5) + 3) * size / 4
    
    for i in range(5):
        Q = rotDeg2D(P, 18 + 72 * i)
        points[i] = (Q[0], Q[1], z)

    P = ((np.sqrt(5) + 1) * size / 2, 0.0)
    z = (np.sqrt(5) - 1) * size / 4

    for i in range(5, 10):
        Q = rotDeg2D(P, 18 + 72 * i)
        points[i] = (Q[0], Q[1], z)

    z = -(np.sqrt(5) - 1) * size / 4

    for i in range(10, 15):
        Q = rotDeg2D(P, 54 + 72 * i)
        points[i] = (Q[0], Q[1], z)

    P = (size, 0.0)
    z = -(np.sqrt(5) + 3) * size / 4

    for i in range(15, 20):
        Q = rotDeg2D(P, 54 + 72 * i)
        points[i] = (Q[0], Q[1], z)

    surfaces[0] = ( 5,10, 6, 1, 0)
    surfaces[1] = ( 3, 4, 0, 1, 2)
    surfaces[2] = ( 9,14, 5, 0, 4)
    surfaces[3] = (19,15,10, 5,14)
    surfaces[4] = (16,11, 6,10,15)
    surfaces[5] = ( 7, 2, 1,6,11)
    surfaces[6] = (17,18,13, 8,12)
    surfaces[7] = (11,16,17,12, 7)
    surfaces[8] = (15,19,18,17,16)
    surfaces[9] = (14, 9,13,18,19)
    surfaces[10] = ( 4, 3, 8,13, 9)
    surfaces[11] = ( 2, 7,12, 8, 3)

def createFace(i):
    
    glBegin(GL_POLYGON)

    idx0 = surfaces[i][0]
    idx1 = surfaces[i][1]
    idx2 = surfaces[i][2]
    idx3 = surfaces[i][3]
    idx4 = surfaces[i][4]

    glVertex3fv(points[idx0])
    glTexCoord2fv(TexCoords[0])

    glVertex3fv(points[idx1])
    glTexCoord2fv(TexCoords[1])

    glVertex3fv(points[idx2])
    glTexCoord2fv(TexCoords[2])

    glVertex3fv(points[idx3])
    glTexCoord2fv(TexCoords[3])
    
    glVertex3fv(points[idx4])
    glTexCoord2fv(TexCoords[4])
     
    glEnd()

# OpenGLの初期化関数
# OpenGLの初期化関数
def initializeGL():

    global textureIds, idxModels, idxModel

    # 背景色の設定 (黒)
    glClearColor(0.0, 0.0, 0.0, 1.0)

    # 深度テストの有効化
    glEnable(GL_DEPTH_TEST)

    glEnable(GL_CULL_FACE)

    #glEnable(GL_LIGHT0)
    #glLightfv(GL_LIGHT0, GL_POSITION, (1.0, 1.0, 1.0, 1.0))

    #createDodecahedron(MODEL_SIZE)

    #"""
    # テクスチャの有効化
    glEnable(GL_TEXTURE_2D)

    createDodecahedron(MODEL_SIZE)

    #for i in range(len(textureImages)):
    for i in range(12):

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
    
    #"""

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
    
    idxModel = glGenLists(1)
    glNewList(idxModel, GL_COMPILE)
    glBindTexture(GL_TEXTURE_2D, textureId)  # テクスチャの有効化
    glBegin(GL_TRIANGLES)

    for i in range(len(_3d_faces)):

        idx0 = _3d_faces[i][0]
        idx1 = _3d_faces[i][1]
        idx2 = _3d_faces[i][2]

        glTexCoord2fv(texcoords[idx0])
        glVertex3fv(positions[idx0])

        glTexCoord2fv(texcoords[idx1])
        glVertex3fv(positions[idx1])
        
        glTexCoord2fv(texcoords[idx2])
        glVertex3fv(positions[idx2])

    glEnd()

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
        gluPerspective(45.0, WIN_WIDTH / WIN_HEIGHT, 0.1, 100.0)
    
    
        # モデルビュー行列
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        
        gluLookAt(0.0, 0.0, -0.1,   # 視点の位置
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
        if len(idxModels) > 0:
            for i in range(len(idxModels)):
                glCallList(idxModels[i])
    
        """

        axisY = (0.0, 1.0, 0.0)
        axisX = (1.0, 0.0, 0.0)

        for i in range(12):

            idx0 = surfaces[i][0]
            idx1 = surfaces[i][1]
            idx2 = surfaces[i][2]
            idx3 = surfaces[i][3]
            idx4 = surfaces[i][4]
           
            v0 = points[idx0] + points[idx1] + points[idx2] + points[idx3] + points[idx4]
            l0 = np.linalg.norm(v0)
            v1 =v0 / l0

            alpha = np.arccos(np.dot(v1, axisY)) #- np.pi / 2
            if v1[2] < 0:
                alpha *= -1

            v1[1] = 0.0
            l1 = np.linalg.norm(v1)
            v2 = v1 / l1

            beta = np.arccos(np.dot(v2, axisX)) - np.pi/2
            
            if v2[2] >= 0:
                beta *= -1
            
            glPushMatrix()
            #glTranslatef(-v0[0] / 5.2, -v0[1] / 5.2, -v0[2] / 5.2)
            glTranslatef(-v0[0] / 5.0, -v0[1] / 5.0, -v0[2] / 5.0)
            glRotatef(np.rad2deg(beta), 0.0, 1.0, 0.0)
            glRotatef(np.rad2deg(alpha) - 90, 1.0, 0.0, 0.0)
            #glTranslatef(0.0, 0.01, 0.0)
            glCallList(idxModel)
            glPopMatrix()

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

def loadPly(ply_filename, zScale = 1.0):

    positions = []
    texcoords = []
    _3d_faces = []

    plydata = PlyData.read(ply_filename)
   
    x = plydata.elements[0].data['x']
    y = plydata.elements[0].data['y']
    z = plydata.elements[0].data['z'] * zScale * MODEL_SIZE
    s = plydata.elements[0].data['s']
    t = plydata.elements[0].data['t']
    
    meanX = np.mean(x)
    meanY = np.mean(y)
    meanZ = np.mean(z)
    
    for i in range(x.shape[0]):
        positions.append((-(x[i]-meanX), y[i]-meanY, -(z[i]-meanZ)))
        texcoords.append((s[i], t[i]))

    for i in range(plydata['face'].data['vertex_indices'].shape[0]):
        _3d_faces.append(plydata['face'].data['vertex_indices'][i].tolist())

    return positions, texcoords, _3d_faces

def normalizePos(pos, scale = 1.0):

    pos = np.array(pos)

    minX = np.min(pos[:,0])
    maxX = np.max(pos[:,0])
    width = maxX - minX
    pos[:,0] -= (minX + maxX) / 2

    minY = np.min(pos[:,1])
    maxY = np.max(pos[:,1])
    height = maxY - minY
    pos[:,1] -= (minY + maxY) / 2

    aspect = height / width

    pos[:,0] /= width
    pos[:,1] *= aspect
    pos[:,0] *= scale * MODEL_SIZE
    pos[:,1] *= scale * MODEL_SIZE

    return pos 

def main():

    global positions, texcoords, _3d_faces, textureImage

    createColors(12)

    argv = sys.argv
    argc = len(argv)

    if argc < 3:
        print('python %s' % argv[0])
        print('python %s <image> <ply> [<zScale> <Scale>]' % argv[0])
        quit()

    #"""
    # オール画像の場合、TexCoords生成のため呼び出す...
    createTexture((0, 0, 0), '0')

    for i in range(12):

        moji = '%d' % i
        texture = createTexture(colors[i], moji)
    
        texture = cv2.flip(texture, -1)
        textureImages.append(texture)
    #"""

    img = cv2.imread(argv[1], cv2.IMREAD_COLOR)
    img = prescale(img)
    WIN_HEIGHT, WIN_WIDTH = img.shape[:2]
    textureImage = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    textureImage = cv2.flip(textureImage, 0)

    zScale = 1.0
    if argc > 3:
        zScale = float(argv[3])

    scale = 1.0
    if argc > 4:
        scale = float(argv[4])

    positions, texcoords, _3d_faces = loadPly(argv[2], zScale)

    positions = normalizePos(positions, scale)


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
