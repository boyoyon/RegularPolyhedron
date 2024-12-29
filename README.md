<html lang="ja">
    <head>
        <meta charset="utf-8" />
    </head>
    <body>
        <h1><center>Regular Polyhedron</center></h1>
        <h2>なにものか？</h2>
        <p>
            正多面体を表示するだけのプログラムです。<br>
        </p>
        <h3>正４面体</h3>
        <img src="images/tetrahedron.gif"><br>
        <h3>正６面体</h3>
        <img src="images/hexahedron.gif"><br>
        <h3>正８面体</h3>
        <img src="images/octahedron.gif"><br>
        <h3>正１２面体</h3>
        <img src="images/dodecahedron.gif"><br>
        <h3>正１２面体(骨格)</h3>
        ４枚の五角形の頂点を結んで面を張ると正１２面体になります。<br>
        <img src="images/dodecahedron_framework.gif"><br>
        <h3>正２０面体</h3>
        <img src="images/icosahedron.gif"><br>
        <h3>正２０面体(骨格)</h3>
        黄金比の長方形３枚を組み合わせた骨格の頂点を結んで面を張ると正２０面体になります。<br>
        <img src="images/icosahedron_framework.gif"><br>
        <h3>正２０面体(細分１)</h3>
        各辺の中点が外接球に接するように持ち上げて細分します。正多面体ではなくなります。<br>
        <img src="images/how_to_subdivide.gif"><br>
        <img src="images/icosahedron_subdiv1.gif"><br>
        <h3>正２０面体(細分２)</h3>
        <img src="images/icosahedron_subdiv2.gif"><br>
        <h3>正２０面体(細分３)</h3>
        <img src="images/icosahedron_subdiv3.gif"><br>
        <h3>正２０面体(細分４)</h3>
        <img src="images/icosahedron_subdiv4.gif"><br>
        <h3>サッカーボール</h3>
        正２０面体のでっぱり部分１／３を切り落とすとサッカーボールになります。<br>
        <img src="images/soccerball.gif"><br>
        <h2>環境構築方法</h2>
        <p>
            pip install opencv-python PyOpenGL glfw<br>
        </p>
        <h2>使い方</h2>
        <h3>正４面体</h3>
        <p>
            python tetrahedron.py<br>
            python tetrahedron.py (画像ファイル名１) ・・・ (画像ファイル名４)<br>
        </p>
        <h3>正６面体</h3>
        <p>
            python hexahedron.py<br>
            python hexahedron.py (画像ファイル名１) ・・・ (画像ファイル名６)<br>
        </p>
        <h3>正８面体</h3>
        <p>
            python octahedron.py<br>
            python octahedron.py (画像ファイル名１) ・・・ (画像ファイル名８)<br>
        </p>
        <h3>正１２面体</h3>
        <p>
            python dodecahedron.py<br>
            python dodecahedron.py (画像ファイル名１) ・・・ (画像ファイル名１２)<br>
            <br>
            python dodecahedron_framework.py<br>
            <br>
            おまけ<br>
            python dodecahedronImgPly.py (RGB画像ファイル名) (PLYファイル名) [(zスケール) (xyスケール]<br>
            正１２面体の各面にテクスチャーつき3Dを貼り付けてみました。<br>
            <img src="images/dodecahedronImgPly.gif"><br>
        </p>
        <h3>正２０面体</h3>
        <p>
            python icosahedron.py<br>
            python icosahedron.py (画像ファイル名１) ・・・ (画像ファイル名２０)<br>
            <br>
            python icosahedron_framework.py<br>
            <br>
            python icosahedron_subdivision.py<br>
            python icosahedron_subdivision.py (画像ファイル)<br>
            <br>
            ・0キー押下：正２０面体<br>
            ・1キー押下：正２０面体(細分１)<br>
            ・2キー押下：正２０面体(細分２)<br>
            ・3キー押下：正２０面体(細分３)<br>
            ・4キー押下：正２０面体(細分４)<br>
            <br>
            python soccerball.py<br>
            python soccerball.py (画像ファイル名１) ・・・ (画像ファイル名２０)<br>
            <br>
        </p>
        <table border="1">
            <tr><th>操作</th><th>機能</th></tr>
                <tr><td>左ボタン押下＋ドラッグ</td><td>3Dモデルの回転(yaw,pitch)</td></tr>
                <tr><td>rキー押下＋ホイール回転</td><td>3Dモデルの回転(roll)</td></tr>
            <tr><td>右ボタン押下＋ドラッグ</td><td>3Dモデルの移動</td></tr>
            <tr><td>ホイール回転</td><td>3Dモデルの拡大・縮小</td></tr>
            <tr><td>ホイールボタン押下</td><td>慣性モードのトグル(on⇔off)</td></tr>
            <tr><td>iキー押下</td><td>(同上)</td></tr>
            <tr><td>sキー押下</td><td>スクリーンショット保存</td></tr>
            <tr><td>ウィンドウ閉じるボタン押下　</td><td>プログラム終了</td></tr>
        </table>
    </body>
</html>
