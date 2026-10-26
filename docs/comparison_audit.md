> 公開用転記：初回4粒子実験の監査記録。行番号は当時の元スクリプトを指し、ポータブル版の行番号ではない。元実験のローカル絶対パスを除去し、リンクを配置に合わせた。32粒子再実験は後日実施したため、当時の「未実行」は現在の状況ではない。

# 実装比較監査：論文・公式デモ・今回の小規模実験

監査日：2026-10-26。追加学習・元コード変更なし。今回変更したファイルは本書だけ。保存済みコード、checkpointの係数・mask、標本、metrics、lossログを読み取り、本文と付録を比較した。

## 結論と重要な訂正

**今回の実験は、論文の手法を使う小規模な分布学習検証であり、論文のharmonic-chain図3の再現ではない。** 公式コードが無変更でも、32→4粒子、RNVP 8→4層、MLP幅128→16、CLI既定5000→600更新、batch200→128、float32→float64という設定変更がある。論文の周波数・低速モード評価も行っていない。

学習で実行されたHamiltonian・log-density・logdet符号・KL正規化・共分散の式について、保存済み結果を無効にする明白な誤りは見つからなかった。ただし数値的往復テストだけで勾配・密度全体の厳密検証まで済んだとはしない。

重大な解釈上の注意は、**45%の残差を「600更新が短かっただけ」とは説明できない**こと。終端の損失はほぼ横ばいで、今回の有限層ネットワーク・mask・初期化・最適化・正則化を分離比較していない。一方、point transformation一般と今回の学習可能な座標分散priorの組合せに、harmonic chainを原理的に表せない欠陥はない（後述の解析的構成）。今回の特定4層RNVPの表現可能性まで証明したわけではない。

記録上の訂正：README.md L24の「公式Gaussian priorがfloat32で生成された後doubleになる」は、乱数標本の精度の説明としては誤り。Gaussianのsigmaパラメータはconstructorでfloat32だが、flowBuilder内で先にdouble化され、sampleはsigma.dtypeで乱数を生成するため、今回の潜在標本は生成時点からfloat64（根拠：source/gaussian.py L9–15、utils/flowBuilder.py L14、experiment.py L16,L29,L33）。依頼に従いREADME自体は変更しない。

## 根拠の扱い

論文は [arXiv v3 PDF](https://arxiv.org/pdf/1910.00024v3) を使用。ページはPDF先頭を1とした印字ページ番号。本文pp.1–11、補足に相当するAppendix A/Bはpp.12–13。Aは別の正規写像、BはMNIST圧縮の説明で、harmonic chainの詳細なハイパーパラメータ表は見つからない。

公式コードは [commit f5673f185a2b787f748cb72721eadebf907413a4](https://github.com/li012589/neuralCT/tree/f5673f185a2b787f748cb72721eadebf907413a4)。`git rev-parse HEAD`で照合し、`git status --short`は空だった。以下の公式ファイル行番号は `upstream/` 以下のこのcommit。notebookのL番号はipynbのJSONファイル行番号で、cell番号は0始まり。

公式デモは `2_HarmonicChain.ipynb` の解析ロジック。実際の学習済みdemo/parameter.hdf5・checkpointは取得していない。このため「CLI既定」「notebookに記されたパス」「実際の論文実行値」は同一視しない。今回は `experiment.py`（全80行）と `results/metrics.json` を実際の実行の根拠とする。

## 三者の比較

| 項目 | 原論文 | 公式コード／デモ | 今回の実行 |
|---|---|---|---|
| 物理系 | IV.B、pp.5–6：固定端harmonic chain、正常モードを解析解と比較 | source/harmonicChain.py L10–22：下記の三重対角行列 | 同じクラス、kappa=1、n=4（experiment.py L17,L28） |
| 粒子数 | Fig.3、p.5は32モード | notebook L78、variation.py L89：32 | 4、位相空間8次元 |
| 境界・zero mode | IV.B、p.5：両端固定、周波数は正 | 座標精度行列diag=2、隣接=-1。周期鎖ではない | 同じ構造。並進zero modeなし |
| 逆温度 | IV.B、p.6：beta=1 | HarmonicChainのK既定1。CLIの-K=300はtargetに渡されない | target.K=1を使用。温度300の実験ではない |
| 正規写像 | III.A.1、p.3、式(2)(3)：座標の可逆写像と対応する運動量写像。real NVPを使用 | PointTransformation＋RNVP。flowBuilderのtypeLayer=1 | 同じ種類を1個使用（L25–29）。一般のq,p混合層は追加しない |
| 潜在分布 | III.A.2、pp.3–4、式(4)：学習可能な座標周波数、単位運動量分散 | 標準Gaussian→座標のみDiagScaling→正規写像 | 同じ。relax=False、shift=False |
| RNVP層数・MLP | harmonic-chain固有の深さ・幅・活性化の値は本文／付録では不明 | notebookパスL59はdepth1/l8/M2/H128。L65–74で実際はHDF5から読む。CLI既定は1/8/2/128（variation.py L32–35） | 1 point map、4 coupling層。各t/sに4→16→16→4（L20,L26–27） |
| 活性化 | real NVPという種類以外のharmonic-chain固有値は不明 | t/sのhidden=Softplus。t出力linear、s出力ScalableTanh（variation.py L110、notebook L95） | 同じ種類。ただし小容量 |
| mask | 詳細・seed不明 | 偶数層にランダム半分mask、次の層に補集合（variation.py L98–109） | 同じ規則、4次元で半分2個。seed42（L15,L18–24） |
| 初期化 | harmonic-chain固有の詳細・seed不明 | Linearの既定初期化、ScalableTanh.scale=0、DiagScalingに0.1×NumPy標準正規。CLI seedはコメントアウト | 同じ初期化規則だがfloat64でLinearを構築しseed固定。公式と同じ初期値ではない |
| 学習目的 | III.B.1、p.4、式(5)：変分自由エネルギー | reversedKLDはlog rho−未正規化log target（train/learn.py L43–48） | その関数を直接使用（L47） |
| optimizer/LR | harmonic-chain固有の値は本文・補足では不明 | CLI/train既定Adam、lr=.001、weight_decay=.001 | 同じ値（L44）。AdamWではない |
| scheduler | harmonic-chain固有値は不明。IV.Cのvalidation plateauによる減衰は分子の説明 | learnにはoptional StepLR(500,.7)、既定OFF。variation.pyは有効化しない | なし。LR一定 |
| batch/更新 | harmonic-chain固有値は本文・補足では不明 | CLI既定batch200、epochs5000。各epochは新規標本1 minibatch=1更新。demoの実学習数はHDF5未取得で不明 | batch128、600更新、early stoppingなし（L43–50） |
| precision/device | harmonic-chain固有値は不明 | notebookはCPU/float32（L60–61）。CLI既定float32、-doubleで64 | Mac CPU、float64、Torch各1 thread（L13–16,L28–29） |
| 評価 | IV.BとFig.3、pp.5–6：周波数・2低速モードのJacobian | notebook cell4：周波数、10生成標本でモードJacobian | 周波数／mode評価なし。KL、全q,p共分散、往復、正規写像の局所条件を追加 |

論文のAdam/batch200/lr=.001/plateau減衰の具体的記載（IV.C、p.6）を、harmonic chainにも適用したと推測してはいけない。今回の値は公式CLI/trainから選んだもので、論文実行条件の確認にはならない。

## Hamiltonian・境界条件・正規化の点検

公式 source/harmonicChain.py L16–17 は、n>=2について

\[
A=\tfrac12\operatorname{diag}(\kappa L,I),\qquad
L=\begin{pmatrix}2&-1&&\\-1&2&-1&\\&\ddots&\ddots&\ddots\\&&-1&2\end{pmatrix},
\quad H=\tfrac12p^Tp+\tfrac\kappa2q^TLq.
\]

これは両端を固定した n+1 本のばね、すなわち \(\sum_{i=1}^{n+1}(q_i-q_{i-1})^2\)、q0=q(n+1)=0 に対応。固有値 \(4\sin^2(k\pi/(2(n+1)))\) は全て正。4粒子で周波数は約[0.6180,1.1756,1.6180,1.9021]。zero mode除去やpinningの追加処理は不要。

**論文の印字式には注意**：IV.B p.5に書かれたばね項の和の上限はnで、その直後の「両端固定」および記載された周波数と、文字どおりには整合しない（上限nでは右端のばねが抜ける）。公式実装と今回コードは、両端固定・記載された正常モード周波数に整合するLを用いている。今回独自に別の境界条件へ変えたという問題ではなく、本文の省略／誤記と意図されたモデルを区別する必要がある。

beta=1で \(\Sigma_*=(2A)^{-1}=\operatorname{diag}(L^{-1},I)\)。今回experiment.py L34–35の計算は正しい。保存済みsamples.npzの真値も次と一致する：

```text
Sigma_qq = [[.8,.6,.4,.2], [.6,1.2,.8,.4], [.4,.8,1.2,.6], [.2,.4,.6,.8]]
Sigma_pp = I4; Sigma_qp = 0
log Z = 4 log(2 pi) - (1/2) log det(2A) = 6.5467893094203315
```

公式SourceのKは名前が紛らわしいがenergyに掛ける係数（source/source.py L29–33）。variation.py L28の説明はTemperature、既定300だがL63–73で保存するだけで、L89のHarmonicChainには渡さない。notebookもL74で読むだけでL78には渡さない。今回のbeta=1は論文に一致し、パス中T_300の値に合わせる必要はない。

## canonical mapとlatent baseの分解

公式utils/flowBuilder.py L14–36、utils/layerList.py L3–13より、生成方向は

\[
u\sim N(0,I_8),\quad (Q,P)=D(u),\quad (q,p)=S^{-1}(Q,P).
\]

Dは座標前半だけをexp(a)倍、運動量後半はそのままにする。`fix=[0]*n+[1]*n`をDiagScaling内で反転するので、実際の有効maskは[1,1,1,1,0,0,0,0]（flowBuilder L21、diagScaling.py L15–21）。D後の座標分散はexp(2a)、運動量分散は1。したがってomega=exp(-a)。これは論文式(4)のpriorを標準正規から実装したもので、priorの座標分散を固定してしまった実験ではない。

SはPointTransformationで、座標Fに対応して運動量に逆Jacobianを作用させる。実装はvector-Jacobian productをautogradで計算しcreate_graph=Trueでパラメータ学習に必要な高階微分を残す（pointTransformation.py L11–25）。コード内のQ/qという局所変数名はforward/inverseで物理的呼び方と紛らわしいが、式(2)(3)の対応自体に逆転バグは見つからない。

Dは一般には非シンプレクティックで、Sはシンプレクティック。したがって**標準正規uからの全サンプラーを正規変換だと主張するのは誤り**だが、論文もpriorからのスケーリングと正規写像を分けており、この分解は不一致ではない。今回の局所probeの結果（metrics L19–21）はSの誤差5.55e-16、全体は0.2228。後者は実装破損の証拠ではない。

座標順は論文の(p,q)に対して実装は(q,p)。順列とJの符号を一貫して扱えば正規条件は同等。今回L56–62のM^T J M=Jと論文式(1)のM J M^T=Jは正方可逆なsymplectic行列に対する等価な条件であり、表記の違いだけで失敗とはならない。ただしprobe8点の局所検証と往復16点だけで全空間の数値安定性は保証しない。

## logdet・loss・regularization

RNVP座標写像単体のlogdetは一般に非ゼロ（flow/rnvp.py L19–41）。しかし正規liftで運動量側が逆因子を持つため、PointTransformationの位相空間logdetは0（同クラスL13,L21）。ここで座標RNVPのlogdetをさらに足すと二重計上になる。今回その誤りはない。

全サンプラーG=S^{-1}Dについて、log|det DG|=sum(a)をDiagScaling L27–33が返す。flow/flow.py L17–24が `log p0(u)-logdet` を返し、learn.py L43–48はそこにbeta Hを足す。今回の学習L47と評価L37–41はこの符号に整合。真のtarget.logProbabilityは未正規化だが、その定数は学習勾配に不要。評価のみlogZを足して正規化済み D_KL(rho||pi) としており、逆向きKLやGaussian近似KLを計算しているわけではない。

Adamのweight_decay=.001は公式train既定を引き継いでいる（learn.py L7,L15、experiment.py L44）。したがって実際の最適化には、論文の表示された変分目的以外のL2寄与がある。loss.csvはそのL2を加えたtotalではなく、正則化前の変分loss。正則化が45%の主因という証拠はない。内部的にはDiagScalingのelements後半も単一のtrainable Parameterの一部で、物理写像にはfix=0で寄与しなくてもweight decayで更新される点に注意。

## 表現制限と45%の解釈

**解析的な区別（追加学習の結果ではない）**：L=O diag(omega^2) O^Tと直交対角化すれば、Q=O^T q、P=O^T pはpoint transformationかつ正規変換。Q~N(0,diag(omega^-2))、P~N(0,I)を使えば元の固定端鎖のGaussianを厳密に表せる。このため「運動量priorを単位分散に固定したこと」「point mapを選んだこと」一般に起因する不可避な45%誤差、とは説明できない。Appendix A.1 p.12にも直交モード変換との関係が記載される。

ただし、**今回の特定4 coupling層・幅16・mask順序が必要な直交写像を厳密に表せるかは未証明**。一般のpoint mapの表現可能性と、有限RNVPの表現可能性・学習しやすさは別。Softplusは線形関数の構成を妨げるだけの活性化ではないが、4層で任意の直交行列を表せるとの断定はしない。shift=Falseも、原点中心のこのtargetでは直ちに不適合を意味しない。一般のq,p混合Symplectic層を追加しなかったことも、今回の鎖の厳密表現に必要な構造を必ず欠くことにはならない。

保存済みtrained.ptのmaskは次のとおり。全変数が少なくとも一度更新・条件付けに参加しており、単純な「ある座標が全層で固定」という欠落はない。ただしこの事実だけで十分な混合能力は証明しない。

```text
0011
1100
0110
1001
```

観測値：KL 2.04085→0.354309、共分散相対Frobenius誤差0.595222→0.452404。loss.csvの100更新平均は400–499で−6.175368、500–599で−6.174673、厳密自由エネルギーは−6.546789。終端は明瞭な下降を続けているとは言えない。600更新は低負荷の動作検証用予算であり収束判定の結果ではない。単に回数を増やせば解決すると結論できない。

保存済み標本8192点の読み取り解析では、最終全共分散誤差の二乗ノルムは2.37416。その内訳はqq=1.73975、pp=0.628894、qp/pq合計=0.00552121。座標相関が弱く、運動量にも非対角相関が残る：例えば真のCov(q1,q2)=0.6に対して標本値0.23456、真のCov(p1,p2)=0に対して−0.29779。これはpoint mapで座標と運動量を同時に合わせる学習が不十分だったという観測であり、その原因が容量・局所解・正則化のどれかを決める証拠ではない。共分散だけから生成分布が完全なGaussianだとも推定しない。

評価式はexperiment.py L41の \(\|\widehat\Sigma-\Sigma_*\|_F/\|\Sigma_*\|_F\)。全q,pの8×8行列で、初期・最終各8192標本からnp.covの標本平均中心化・分母8191で算出。45%は個々の変数の一律誤差でも「論文の相対誤差」でもない。有限標本誤差も含むが、その寄与の反復推定や信頼区間は実施していない。KLのMC標準誤差0.00916を共分散誤差の誤差棒として使えない。

未検証仮説：600更新の予算、4層・幅16への縮小、maskによるパラメータ化、初期値seed42、一定LR、Adam＋weight decay、その他の最適化停滞。これらの比較実験・追加学習はしていない。今回の監査で確定できるのは設定差、計算式の整合、および保存済みの残差である。

## 公式notebook自体の注意点（今回には未使用）

1. L18の`variationMain.py -source 1`はこのcommitのCLIと一致しない。現行はvariation.pyと文字列HarmonicChain（variation.py L88–91）。説明が古い。
2. notebook L71とvariation.py L81はロードしたlrをintにするため、.001なら0になる。今回の新規学習ではL44でfloat .001を直接指定し、この経路を使っていない。notebook自体もここで再学習しないため即座に図を壊す箇所ではない。
3. notebook L136–138のomega=exp(-(d0+d1))は、実際の有効scale=elements*fixを使っていない。現在のrelax=Falseではd1は無効なので、一般にはexp(-d0)が正しい。保存済みdemoのfix・係数を取得していないため、配布済み図への実害の大小は不明。今回checkpointの無効d1はweight decayでほぼ0になっているが、今回そもそもこの周波数評価を実行していない。
4. DiagScaling.transMatrix L35–40は実際のexp(elements*fix)の行列を返しておらず、inverse側も1/elementsで、一般には不整合。今回はこのメソッドを使わずautograd Jacobianを使った（experiment.py L61）。今回の幾何学指標への混入はない。

以上は対象commitのコードから確認した潜在的問題。論文図の誤りを証明するものではなく、今回の45%を生んだ直接の経路でもない。

## 今回の評価範囲と再実行

公式notebook cell4（JSON L134–177）はomegaの解析解比較と、生成10標本上で低速2モードのJacobianを比較する。今回L36–41,L53–78はKL・全共分散・往復・局所symplectic defect・q1/q2散布図のみ。**密度が改善したことと、正常モード／周波数を論文どおり獲得したことは別**であり、後者は未確認。

今回の実行条件：モデル／mask／NumPy seed42、学習seed43、評価seed2026。学習前に独立な標準Gaussian8192点を用意し、学習前後で同じ潜在点を使用する対応比較。batch128×600=76800訓練生成標本。metricsの3.158秒は学習loopだけで、import・評価・描画は含まない。

既存環境での再実行コマンド（今回の監査では実行していない。実行するとresultsとrun.logを上書き）：

```sh
cd <repository>
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python experiments/harmonic4/run.py
```

## 参照箇所一覧

- 論文PDF：II.A p.2 式(1)；III.A.1 p.3 式(2)(3)；III.A.2 pp.3–4 式(4)；III.B.1 p.4 式(5)；IV.B pp.5–6 Fig.3；IV.C p.6（分子の学習設定）；Appendix A p.12、B pp.12–13。
- [公式variation.py](../vendor/neuralCT/variation.py)：L16、L21–37、L63–89、L98–120。
- [公式notebook](../vendor/neuralCT/2_HarmonicChain.ipynb)：cell1/2/4、JSON L18、L59–98、L134–177。
- [flowBuilder](../vendor/neuralCT/utils/flowBuilder.py)：L12–36；[DiagScaling](../vendor/neuralCT/flow/diagScaling.py)：L8–40；[PointTransformation](../vendor/neuralCT/flow/pointTransformation.py)：L11–25。
- [HarmonicChain](../vendor/neuralCT/source/harmonicChain.py)：L6–22；[Source](../vendor/neuralCT/source/source.py)：L26–33；[Gaussian](../vendor/neuralCT/source/gaussian.py)：L7–18。
- [RNVP](../vendor/neuralCT/flow/rnvp.py)：L13–41；[MLP](../vendor/neuralCT/utils/layers/mlp.py)：L6–35；[ScalableTanh](../vendor/neuralCT/utils/layers/scalableTanh.py)：L5–10。
- [learn/reversedKLD](../vendor/neuralCT/train/learn.py)：L7–33、L43–48；[sample/logProbability](../vendor/neuralCT/flow/flow.py)：L17–30；[layer order](../vendor/neuralCT/utils/layerList.py)：L3–13。
- [今回コード](../experiments/harmonic4/run.py)：L13–29設定、L32–51学習評価、L53–78幾何学と描画；[metrics](../results/harmonic4/metrics.json)：L2–27；[loss](../results/harmonic4/loss.csv)；保存済み標本（公開パッケージには含めない）。

本監査では原コード・README・metrics・画像・モデルを変更せず、新しい監査文書だけを追加した。新しい学習・公開・外部共有はない。
