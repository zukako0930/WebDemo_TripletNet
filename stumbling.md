# flask
- api宣言時、引数の'static_folder'で指定しないと'render_template()'先のhtmlから外部jsを参照できない
- render_template()の引数にlist等のパラメータを渡すことが可能
- html上でpythonのコードを書く場合'jinja'を利用する

# htmlからflaskへの渡し方
- ボタンを押した時の動作としてjQueryでのsubmitからflask側でrequest.forms[]で受け取ることを考えた
- 結局formのactionに'/predict'を指定することでflask側からrequest.file[]の形で受け取れることがわかった。jQueryを書く必要ないのでこの方が楽

# jinja
- html上でflaskから渡されたpythonのオブジェクトを表示する時に利用。
- 値の代入は予め.pyファイル上で行って置くべき
- for文で書く場合'<img>'等のタグはその文生成される
