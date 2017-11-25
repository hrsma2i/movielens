
# MovieLens 100K recommendation
MovieLens 100K のデータセットに対し、 
recommendation のアルゴリズムの1つ、 SVD
を実装した。

## 1. SVDアルゴリズムの説明

### recommedation アルゴリズムの概要

- 背景: 最近は情報過多社会。ユーザーにとって最適な商品（アイテム）の情報を探すのは大変。
- 目的: そこで各ユーザーが好む最適な商品を、全ての商品から選出する必要がある。
- 入力: ユーザーの行動履歴・評価 | アイテムの特徴 etc
- 出力: ユーザーの評価が高い・買うであろうアイテム | 似たようなアイテム

### recommedation アルゴリズムの分類

- **CF Collaborative Filetering**
    - memoery-based | Nearest Neighbor
        - user-user | user-based
        - item-item | item-based
    - **model-based**
        - **MF Matrix Factorization**
            - SVD (only  explicit)
            - SVD++ (also implicit)
            - NMF
        - clustering | regression | 確率モデル | topic model | Bayesean Network model | restricted Boltzmann machine | time series model
- Content-based
- Hybrid = CF + Content

今回は、 CF > model-based > MF > SVD を実装した。

### CF vs Content-based
- **CF**
    - 入力: あるユーザーによる 商品のrating, 商品の購買・閲覧履歴 (e.g. 購入した服の5段階☆評価、ある服のページの閲覧、購入入したかしないか）
    - 出力: ユーザーの評価が高い・買うであろうアイテム (e.g. )
- Content-based
    - 入力: アイテムの特徴 (e.g. 服の画像) 
    - 出力: 似たようなアイテム (e.g. 見た目が似たような服)

### memory-based vs model-based

CF では、下のFig.1のような　**(ユーザー数) x (アイテム数) の形の user-item matrix** をデータとして扱う。

![](https://i.imgur.com/bmW79NS.png)

Fig.1: user-item matrix rating

行がuser, 列がitem　を表し、
行列の各要素が、ある user による、ある item の 評価値である。

行列の要素は **1~5段階評価** などがよく使われるが、
評価ではなく、そのitemページの閲覧や、そのitemの購入をしたかしないか
の **バイナリ（0/1）** でもよい。

#### memory-based

memory-based は、各行|列を user|item を表すベクトルとしてとらえる。
そして、**ベクトル間の内積**などで、 user|item 間の類似度を計算し、
特定の user に対し、オススメの item を recommend する。

**user-based** だと、user ベクトルの類似度を計算し、
推薦をしたい user A が、まだ購入していない item のうち、
A と似ている user B, C が高評価しているアイテムを推薦する。

**item-based** では、 item ベクトルの類似度を計算し、
A が高評価している item と似ている item を推薦する。

欠点: user-item matrix はスパース性であることが多く、本来ならば似てる user や item なのに、未評価というだけで、似ているとされない問題がある。

#### model-based

上述のスパース性の問題に対応する手法であり、 
user-item matrix をそのまま推薦に用いるのではなく、
一度、統計モデルを施してから、推薦する手法。

### SVD

#### MF Matrix Factorization
user-item matrix における各 user|item ベクトルを
潜在的な特徴量での表現に変換し、欠損している評価値を推定する手法。

具体的には、 user-item matrixの形が $U \times I$ だとすると、
各行 = user ベクトルの特徴量は、各 item の評価値 $r_i$ である。

$\boldsymbol{u} = (r_1, r_2, ..., r_I)$

一方、各列 = item ベクトルの特徴量は、各 user の評価値 $r_u$ である。

$\boldsymbol{i} = (r_1, r_2, ..., r_U)$

これらのベクトルを、$k$ （任意）次元の特徴量で表現する。

$\boldsymbol{u} = (f_1, f_2, ..., f_k)^T$

$\boldsymbol{i} = (g_1, g_2, ..., g_k)^T$

これらを user|item ごとに並べた行列を

- $P: K \times U$
- $Q: K \times I$

とし、それぞれ **user|item factor** と呼ぶことにする。
そして、真の user-matrix（欠損値のない user-matrix）は、
これらの積で表されると仮定する。
すなわち、真の user-matrix の予測行列を、 $\hat{R}$ とすると

$\hat{R} = P^TQ$

となる。

このような仮定を置く理由は、以下の直観的な背景からである。

userおよびitemは、k個の潜在的な特徴量を持っていると考える。
たとえば、

- user ベクトルの視点: この user が SF をどれくらい好きかを表す数値
- item ベクトルの視点: この movie がどれくらい、SFに当てはまるかの数値

などが潜在的な特徴量として考えられる。

userの好みのジャンルとitemのジャンルが近ければ、
そのuserによる、そのitemの評価は高くなることが直観的に考えられる。

また、userの潜在特徴ベクトルと、itemの潜在特徴ベクトルの内積は
両ベクトルが似ているほど高い値を示す。
つまり、先の、ジャンルと評価値の関係は、
この潜在特徴ベクトルとその内積の値の関係ににている。
なので、評価値は、各user、itemの潜在特徴ベクトルの内積で表せると仮定する。

$r_{u,j} = \boldsymbol{u}^T\boldsymbol{i}$

これを行列でまとめて表現すると、先ほどの仮定に一致する。

$\hat{R} = P^TQ$

また、この仮定は、user-item matrix $R$ を
2つの行列 $P, Q$ に分解しているので、行列分解 Matrix Factorizationとよぶ。

#### SVD

MF のアルゴリズムの代表の1つとして、
**SVD Singular Value Decomposition** があり、
今回はこれを実装した。

user|item factor どうしの積 $P^TQ$ が、
真の user-item matrix $R$ に近づけばいいので、
これらの誤差を目的関数とする最小化問題を解けば良い。
式で表すと、

$
min. \sum_{u,i}(R-P^TQ)_{u,i}^2 
+ \lambda(\|P\|^2 + \|Q\|^2)
$

なお、学習パラメータは $P,Q$ であり、 
$\lambda$ は過学習を防ぐ正則化項の係数である。

最適化手法として、今回は SGD Stochastic Gradient Descent を用いる。
すなわち、上記の目的関数の勾配を用いて、パラメータ $P, Q$ を更新していく。
更新式は以下。

$
e_{u,i} = (R-P^TQ)_{u,i}\\
P_u \leftarrow P_u - \eta (- e_{u,i}Q_i + \lambda P_u)\\
Q_i \leftarrow Q_i - \eta (- e_{u,i}P_u + \lambda Q_i)
$

$\eta$ はlearning rateであり、後の括弧の中が目的関数の勾配である。

注意として、与えられた user-item matrix において、
**評価値の存在する user, item の組 $(u,i)$ のみ**について更新を行う。
そうでないと、欠損値を0とすると、その0が未評価という意味ではなく、
0という評価値として扱われ、良い精度が出ないためである。

なお、すべての、 user, item の組 $(u,i)$ について iteration していき、
1組の $(u,i)$ につき、$P_u, Q_i$ をそれぞれ1回 update してゆく。
また、すべての組についての更新が終わったら、これを1epochとし、
学習時の引数として、指定epoch回数、これを繰り返す。

また、今回は、学習パラメタに bias を追加した。

$$
\hat{R} = P^TQ 
+ \left[
    \begin{array}{rrr}
        b_{u} & 
        b_{u} &
        \cdots &
        b_{u}
    \end{array}
\right]
+ \left[
    \begin{array}{rrr}
        b_{i} \\
        b_{i} \\
        \vdots \\
        b_{i}
    \end{array}
\right]
$$

で近似する。
bias を導入することで、
辛口|甘口 user や、大人気|不人気 item による評価の偏りを考慮できる。
辛口でれば、user-bias のその user の要素は負の値をとり、
その user の評価が全 item において低くなるように調整できる。

上述の説明では、biasを加えると、SVDの本質がわかりにくくなるため、あえて省いた。
目的関数も、勾配の式も、導出の方法は上述同様。

#### 評価
MovieLensの trainting data u1.base、test data u1.test において、
u1.baseで学習させ、u1.testで評価した。

評価指標は直観的にわかりやすいRMSE, MAEを用いた。
評価値が1であれば、各予測評価値が真の観測評価値よりも平均して1ほどずれている
と考えればよい。


## SVDの改善点

### SVD++
