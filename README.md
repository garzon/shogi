## はじめに

２年前、僕は`einsum`にハマってました（？）。そこで、競プロの経験から「将棋のルールや物理法則をすべて巨大な定数テンソルとして事前定義し、現在の盤面テンソルとPyTorchの `einsum`（アインシュタインの縮約記法）で掛け合わせることで、for文やif文を使わずに完全GPU駆動のルールエンジンを作れるのではないか？」と考え、自己満で実験的なフルスクラッチ実装を行いました。

本記事では、その数学的なモデリングと、複雑なルールをテンソル空間の射影として解く原理について、実際のコードを交えながら解説します。

> **※ 注意事項**
> 本記事で解説しているテンソルによるルールエンジンのアイデアおよびPyTorchのコードは、筆者自身が考案・実装したものです……が、それを分かりやすく解説するブログ記事を書くのが面倒くさかったためｗ、この記事のテキスト自体は僕のコードをもとにAI（Gemini）に投げて生成してもらっています。めっちゃ編集してますけど...

---

## 1. テンソル空間と次元（インデックス）の定義

将棋のルールをテンソル演算に落とし込むためには、計算グラフ内で用いる添字（インデックス）の定義が極めて重要です。後の `einsum` における縮約（どの次元で和をとるか）の挙動を厳密に決定づけます。

| 添字 | 定義名 | サイズ | 意味・役割 |
| --- | --- | --- | --- |
| B | `Batch` | 任意 | バッチサイズ。並列にシミュレーションする局面の数。 |
| K | `K_dim` | 21 | **全駒種**。盤上の駒14種（歩=0〜龍=13）＋ 持ち駒7種（歩=14〜飛=20）。 |
| k | `k_dim` | 14 | **盤上の駒種のみ**。大文字の$K$に内包される先頭の14要素。主に盤上の遮蔽や利きの計算に使用します。 |
| F | `F_dim` | 81 | 移動元（**F**rom）のマス。 |
| T | `T_dim` | 81 | 移動先（**T**o）のマス。 |
| P | `P_dim` | 81 | 盤面上の任意のマス（**P**osition）。主に「経路上の障害物」を判定する中継次元として用います。 |
| 小文字p | `Pr_dim` | 2 | 成りフラグ（**p**romote）。`0`: 成らず, `1`: 成り。 |

たとえば、現在の先手の盤面状態は S_black [B, K, F] というテンソルで表現されます。これは「バッチ $B$ において、駒種 $K$ が、マス $F$ に存在するか？」を保持する真偽値Boolテンソルです。また、盤上の全駒（敵味方すべて）の配置は `board_sum [B, k, P]` として保持しておきます。

> 💡 実装上の工夫： `my_einsum2`
> 実際は所々エラーが出るため、自作ラッパー関数`my_einsum2`を使用していますが機能上`torch.einsum`とは大差はないです。
> ```python
> def my_einsum2(eq, *args):
>     args = list(map(lambda _: _.to('cuda', dtype=torch.float16), args))
>     return torch.einsum(eq, *args) != 0
> ```

---

## 2. 合法手生成の原理：定数テンソルと現在場面の中間変数の計算

合法手の生成にあたり、毎回の局面で `if` 文を用いてルールを判定するのではなく、あらゆる条件を **「現在場面によらず定数ルールテンソル（以下`a_*`変数たち）」としてGPUメモリ上に初期化時に事前構築**します。
推論時には、この定数テンソルと現在の局面を表すテンソルを `einsum` で掛け合わせることで一斉に計算します。

各ルールの中間変数がどのように計算されるのか、定数テンソルの生成コードとともに見ていきましょう。

### 2.1 基礎移動候補 (`ok_to_move`の計算)

まずは障害物を無視し（現在場面によらず）「ルール上だけ物理的に到達可能なマス」を算出します。駒の移動ルール、成りの条件、行き所のない駒（1段目の歩など）の制約を組み合わせます。

#### 基礎移動と成りに関する定数テンソル

「盤上に他の駒（障害物）が一切ない」という前提のもと、駒の物理的なポテンシャルを定義するテンソルです。

`a_move_dir [K, F, T, Dir]`: 駒 $K$ が $F$ から $T$ に移動する際の「方向（全20方向＋手駒の種別）」を紐づけるテンソル。

`a_move [K, F, T]`: 上記から方向次元を縮約した、純粋な「移動可能なマス」の判定マスク。【例】桂馬（$K=2$）が５五（$F=40$）にいる場合、$T$ が４三（$22$）と６三（$24$）の要素のみが True となり、それ以外の$T$は False となります。

`a_move_k_dim [k, F, T] / a_oppo_move [k, F, T]`: a_move を盤上の駒種 $k$ に絞ったもの。後者は盤面を反転させた後手視点の移動定義です。

`a_prom [p, K, K_{after}]`: 成りフラグ $p$ に伴う駒の変化行列。
【例】$p=1$（成る）のとき、元の駒が「歩（$K=0$）」であれば、$K_{after}$ が「と金（$8$）」の箇所が True になります。$p=0$ （不成）の場合は単位行列（$K \to K$）として働きます。

`a_prom_able [p, F, T]`: マス $F$ から $T$ への移動において、成りフラグ $p=1$ がルール上許可されるか（敵陣1〜3段目への進入・退出）を定義する空間マスク。

`a_left [K, T]`: 行き所のない駒の禁止ルール。
【例】歩（$K=0$）や香車は最奥の1段目に移動できないため、その領域が False（または禁止フラグ）として設定されています。

**【定数テンソルの構築コード】**

詳しいコードは[ここ](https://github.com/garzon/shogi/blob/main/shogi_rule_constants.py)に参照

`a_movable [K, F, T, p]`: 以上によって、これらを事前に einsum で統合した、「ルール上物理的な移動候補テンソル」の最終形です。

```python
# 以上の定数テンソルを einsum で統合し、最終的な移動候補マトリクスを作る
a_movable = my_einsum("KFT,pKk,pFT,kT->KFTp", a_move, a_prom, a_prom_able, a_left)
```

これで定数`a_movable [K, F, T, p]`得られます。

**【障害物を無視し現在場面の駒の移動可能のマスの候補`ok_to_move`の計算】**
この定数テンソル `a_movable` と現在の盤面 `S_black` を掛け合わせるだけで完了。

```python
# 存在する駒が、障害物をすべて無視する場合、移動可能なマスの候補を一斉に取得
ok_to_move = my_einsum2("BKF,KFTp->BKFTp", S_black, a_movable)
```

### 2.2 自駒への衝突と打ち込み制限 (`jibougai`, `uchibougai`)

ルール上、自分の駒があるマスには移動できず、また何らかの駒があるマスに持ち駒を打つことはできません。

`a_jibougai [T, P]`: 移動先 $T$ と、自駒の存在する位置 $P$ が一致する移動（自駒を取る手）を禁止します。

`a_uchibougai [K, T, P]`: 手駒 $K$ を打つマス $T$ に、既に何らかの駒が$P$に存在する場合の打ち込みを禁止します。

**【定数テンソルの構築】**

```python
# 自駒との衝突: 移動先 T と 盤面マス P が一致する箇所を True とする単位行列
a_jibougai = torch.eye(T_dim, dtype=torch.bool)

# 打ち込み制限: 手駒 K を打つ際、移動先 T と 盤面マス P が一致する箇所を True
a_uchibougai = torch.zeros(K_dim, T_dim, P_dim, dtype=torch.bool)
for k in HAND_PIECE_TYPES:
    for t in range(T_dim):
        a_uchibougai[k, t, t] = True

```

**【現在場面による中間変数の計算】**

自駒の配置 `board_black` および 全駒の配置 `board_sum` と掛け合わせ、反則フラグを立てます。

```python
# 移動先 T に自駒 P が存在するか？
jibougai = my_einsum2("BKP,TP->BT", board_black, a_jibougai)

# 打ち込み先 T に何らかの駒 P が存在するか？
uchibougai = my_einsum2("BKP,kTP->BkT", board_sum, a_uchibougai)
```

### 2.3 二歩判定 (`nifu`)

「同じ筋に自駒の歩がいる場合、手駒の歩を打てない」というルールです。

`a_nifu [k, P, K, T]`: 二歩判定マスク。
【例】マス $P$ に自分の盤上の歩（$k=0$）が存在する場合、同じ筋に属する任意のマス $T$ に対して、手駒の歩（$K=14$）を打つ行動が True（反則）となります。

**【定数テンソルの構築】**

```python
a_nifu = torch.zeros(k_dim, P_dim, K_dim, T_dim, dtype=torch.bool)
for p_x in range(9):        # 筋 (X座標)
    for p_y in range(9):    # 段 (Y座標)
        for t_x in range(9):
            # 盤上の歩(PAWN) が (p_x, p_y) にあるとき、
            # 同じ筋 (p_yが同じ) の任意のマス T に手駒の歩(HAND_PAWN) を打つことを禁じる
            a_nifu[PAWN, p_x*9+p_y, HAND_PAWN, t_x*9+p_y] = True

```

**【現在場面による中間変数の計算】**

```python
# 盤面に歩(K)が位置(P)にあるとき、手駒の歩(k)を(T)に打つフラグを立てる
nifu = my_einsum2("BKP,KPkT->BkT", board_black, a_nifu)
```

### 2.4 飛び駒の遮蔽判定 (`bougai`)

将棋エンジンにおいて面倒い「飛車・角・香車の経路上に他の駒があったら止まる」という遮蔽判定です。

`a_bougai [k, F, T, P]`: 盤上の飛び駒 $k$ がマス $F$ から $T$ へ直線移動する際、中間の経路上にマス $P$ に駒が存在する場合、Trueで移動を禁じる。

**【定数テンソルの構築】**

```python
a_bougai = torch.zeros(k_dim, F_dim, T_dim, P_dim, dtype=torch.bool)
# 例: 飛車の遮蔽定義
for p_x in range(9):
    for p_y in range(9):
        for f_x in range(p_x):
            for t_x in range(p_x+1, 9):
                # f_x (移動元) と t_x (移動先) の間に p_x (障害物) がある場合
                a_bougai[ROOK, f_x*9+p_y, t_x*9+p_y, p_x*9+p_y] = True
                a_bougai[ROOK, t_x*9+p_y, f_x*9+p_y, p_x*9+p_y] = True
        for f_y in range(p_y):
            for t_y in range(p_y+1, 9):
                # f_y (移動元) と t_y (移動先) の間に p_y (障害物) がある場合
                a_bougai[ROOK, p_x*9+f_y, p_x*9+t_y, p_x*9+p_y] = True
                a_bougai[ROOK, p_x*9+t_y, p_x*9+f_y, p_x*9+p_y] = True
# (※角や香車についても同様の直線を定義)

```

**【現在場面による中間変数の計算】**
全駒の配置 `board_sum [B, K, P]` と掛け合わせます。

```python
# 文字列の指定に注意: "BKP"の"K"は board_sum の次元、"k"は a_bougai の次元
bougai = my_einsum2("BKP,kFTP->BkFT", board_sum, a_bougai)

# 後続の計算のため、持ち駒の次元を 0 (False) でパディングして K_dim に拡張する
bougai_Kdim = torch.cat((
    bougai, 
    torch.zeros(bougai.shape[0], K_dim-k_dim, F_dim, T_dim, dtype=torch.bool).to('cuda')
), dim=1)

```

式 `"BKP,kFTP->BkFT"` では、出力次元から `board_sum` の駒種インデックス$K$が欠落しています。これは、`einsum` の縮約により「マス$P$に**駒の種類$K$を問わず何らかの駒が存在するか**」が自動的に総和（sum）されることを意味します。
経路上に一つでも駒が存在すれば「妨害あり（True）」のフラグが立ちます。

### 2.5 難関：王手放置判定 (`oute`)

合法手生成における最難関が「動かすと自玉に敵の駒の利きが通ってしまう（ピンや自殺手）」の除外です。「もし自分がマス$F$から$T$に動いた仮想の盤面において、敵の攻撃が玉に届くか」を計算します。

**【定数テンソルの構築（補助）】**
仮想盤面を作るための加算マスクと、敵視点の移動マスクを用意します。

```python
# a_add_piece [F, P, T]: 移動先 T に駒が追加されたことを表現する (T=PのときTrue)
a_add_piece = torch.eye(P_dim, dtype=torch.bool).unsqueeze(0).expand(F_dim, -1, -1)
a_add_piece = torch.permute(a_add_piece, (0, 1, 2)).contiguous()

# a_oppo_move [k, F, T]: 敵駒(後手)から見た駒の移動ポテンシャル (盤面反転)
a_oppo_move = torch.flip(a_move[:k_dim, :, :], [1, 2])

```

**【中間変数`oute`の計算】**

```python
# 1. 仮想盤面 (board_if_moved) の構築
board_pieces = my_einsum2("BKP->BP", board_sum)
# Fマスが空く
board_if_moved = my_einsum2("BP,PF->BPF", board_pieces, ~torch.eye(P_dim, dtype=torch.bool))
# Tマスが埋まる (a_add_pieceを利用)
board_if_moved = board_if_moved.unsqueeze(3).expand(-1, -1, -1, T_dim) + a_add_piece.unsqueeze(0).to('cuda')

# 2. 自玉の位置を特定
king_pos = board_black[:, KING, :].squeeze(1) # [B, t]

# 3. 仮想盤面上における敵からの遮蔽判定
white_if_moved = my_einsum2("BKP,PT->BKPT", board_white, ~torch.eye(P_dim, dtype=torch.bool))
bougai_to_king = my_einsum2("Bt,kftP->BkfP", king_pos, a_bougai)
# 仮想盤面 board_if_moved を用いて遮蔽を再計算
bougai_to_king = my_einsum2("BkfT,BkfP,BPFT->BfFT", white_if_moved, bougai_to_king, board_if_moved)

# 4. 敵の最終的な利きの計算
# 敵の純粋な移動範囲
white_attack_without_bougai = my_einsum2("BKPT,KPt,Bt->BPT", white_if_moved, a_oppo_move, king_pos)
# 敵の攻撃ライン上に遮蔽物がない（~bougai_to_king）場合、王手(oute)となる
white_attack_king = my_einsum2("BPT,BPFT->BFT", white_attack_without_bougai, ~bougai_to_king)

# 結果をテンソルに格納 (玉自身の移動による自殺手も別途計算し統合)
oute = torch.zeros(K_dim, board_black.shape[0], F_dim, T_dim, dtype=torch.bool)
oute[:] = white_attack_king
# oute[KING] = white_attack_king_if_king_moved != 0 (省略)
oute = torch.permute(oute, (1, 0, 2, 3)).contiguous().to('cuda')

```

詳しいコードは[ここ](https://github.com/garzon/shogi/blob/main/main.py#L76)に参照

### 2.6 合法手の最終決定（`calc_legal_moves_mat()`）

これまで求めたすべての反則フラグ（`jibougai`、`uchibougai`、`nifu`、`bougai`、`oute`）を `total_forbidden` として論理和（加算）で統合し、初期の基礎移動候補`ok_to_move`から除外します。

```python
# 反則フラグの統合（次元ブロードキャストによる加算）
total_forbidden = (jibougai.unsqueeze(1) + uchibougai + nifu).unsqueeze(2) + bougai_Kdim + oute

# 移動候補 AND (NOT 反則手)
legal_moves_mat = my_einsum2("BKFTp,BKFT->BKFTp", ok_to_move, ~total_forbidden)
```

これで、再帰探索や `while` ループを一切用いることなく、バッチ内の全局面におけるすべての合法手`legal_moves_mat[B,K,F,T,p]`がテンソル演算のみで算出されました。つまり、`legal_moves_mat`の要素はTrueの場合、駒$K$がマス$F$から$T$まで、成か不成か（$p$）で移動するのが一つの合法手だと表す。

---

## 3. 局面の推演（状態遷移）の処理

選択した行動（指し手）テンソル `A [B, K, F, T, p]` ($A$の中ただ一つの要素がTrue)をもとに盤面を更新する処理も、配列要素への破壊的代入は使わず、ビットマスクを用いた論理演算（XOR `^` と AND `&`）空間の足し引きで表現します。

ここで、手駒の増減を管理するための定数テンソルを使用します。

* `a_captured [k, K_hand]`：取った盤上の駒 `k` が、手駒の何 `K_hand` に変換されるか（例: と金を取ると歩になる）。
* `a_use_piece` / `a_take_piece`：手駒を使った際、または得た際に、手駒枚数を表す空間次元を前後にシフトさせるための変換行列。

**【定数テンソルの構築（補助）】**

```python
# constant tensors which update hand by shifting if needed
a_use_piece = torch.zeros(K_dim, K_dim-k_dim, F_dim, T_dim, dtype=torch.bool)
a_take_piece = torch.zeros(K_dim-k_dim, K_dim-k_dim, F_dim, T_dim, dtype=torch.bool)
a_captured = torch.zeros(k_dim, K_dim-k_dim, dtype=torch.bool)
for k in range(k_dim):
    a_use_piece[k, :] = torch.eye(F_dim, dtype=torch.bool)
for k in HAND_PIECE_TYPES:
    for k_p in range(K_dim-k_dim):
        if k-k_dim != k_p:
            a_use_piece[k, k_p] = torch.eye(F_dim, dtype=torch.bool)
        else:
            for p in range(1, T_dim):
                a_use_piece[k, k_p, p-1, p] = True
for k in PIECE_TYPES:
    if k != KING:
        a_captured[k, k] = True
a_captured[PROM_BISHOP, BISHOP] = True
a_captured[PROM_ROOK, ROOK] = True
a_captured[PROM_PAWN, PAWN] = True
a_captured[PROM_LANCE, LANCE] = True
a_captured[PROM_KNIGHT, KNIGHT] = True
a_captured[PROM_SILVER, SILVER] = True
for k_t in range(K_dim-k_dim):
    for k_p in range(K_dim-k_dim):
        if k_t != k_p:
            a_take_piece[k_t, k_p] = torch.eye(F_dim, dtype=torch.bool)
        else:
            for p in range(1, T_dim):
                a_take_piece[k_p, k_p, p, p-1] = True

a_add_piece = torch.zeros(F_dim, P_dim, T_dim, dtype=torch.bool)
a_add_piece[:] = torch.eye(P_dim, dtype=torch.bool)
a_add_piece = torch.permute(a_add_piece, (1, 0, 2)).contiguous()
```

**【局面の推演（状態遷移）`apply_action_mat()`】**

```python
# 選択された行動 A からマスクを作成(一応非合法手は~bougai_Kdimで削除)
BkFTp = my_einsum2("BkFTp,BkFT->BkFTp", A, ~bougai_Kdim)
BKF = my_einsum2("BkFTp->BKF", BkFTp) 
BT  = my_einsum2("BkFTp->BT", BkFTp)

# 1. 移動元の駒の消去（XOR演算）
# 移動元マスク(BKF)と自分の盤面(S_black)のANDを取り、対象駒を抽出し全体とXORを取る
after_removing_board = S_black ^ (S_black & BKF)
# (※手駒を使用した場合の消費処理も a_use_piece を用いて同様に行う)

# 2. 移動先への駒の配置（成りフラグの処理）
# 成りの変換定義テンソル a_prom を掛け合わせ、指定マス T へ加算
new_board_black = after_removing_board + my_einsum2("BkF,BkFTp,pkK->BKT", S_black, BkFTp, a_prom[:,:,:k_dim])

# 3. 敵駒の捕獲と盤面からの消去
# 移動先 BT と敵盤面 board_white が重なる駒を特定し、a_captured で手駒(K_hand)に変換
captured_piece = my_einsum2("BKT,BT,Kk->Bk", board_white, BT, a_captured)
new_board_white = board_white ^ my_einsum2("BKT,BT->BKT", board_white, BT)

# (※得た captured_piece を a_take_piece を用いて手駒テンソルに加算する)
# 盤面の反転など...
```
詳しいコードは[ここ](https://github.com/garzon/shogi/blob/main/main.py#L33)に参照

この処理により、バッチ内の千差万別な局面が、完全に並列かつ同一の計算グラフ上で、一斉に次の状態へと遷移します。

---

## おわりに

現状の工学的な課題は「スパース性（Sparsity）の呪い」です。将棋盤の大半は空きマスですが、密テンソル（Dense Tensor）による一括計算では無数の「0 * 0」という無意味な積和演算が発生し、バッチサイズを上げるとVRAMのメモリ帯域を猛烈に消費します。PyTorch の `torch.compile` を用いたカーネル自動最適化やSparse Tensor の導入によるメモリ効率化が不可欠です。

複雑なボードゲームのロジックを、純粋な線形代数とテンソル空間の射影問題として解く試みは、エンジニアリングとして非常に奥深く面白いアプローチだと感じています（これはAIの感想ですね）。

最後までお読みいただきありがとうございました。
