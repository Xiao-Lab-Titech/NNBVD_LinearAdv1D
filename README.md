
1次元線形移流方程式を解くソルバー．
MLBasednW5BVDをオブジェクト化することでNNBVDとして解くことができます（main.cpp参照）．

# 動作環境例
- Ubuntu20.04
- Python 3.11.9
- Pytorch 2.5.1
- ONNX runtime for GPU 1.18.0
- CUDA 11.8
- Nvidia driver 560.35.03

特にPytorch，CUDA，ONNX runtimeのバージョン同士の互換性がかなりややこしいので，以上の動作環境にDockerで再現することをお勧めします．

# 使い方
クイックスタートなら基本的にmain.cppをいじるだけで使うことができます．
再構築関数はNNBVDだけでなく，従来手法であるBVD法やWENO法なども設定して解くことができます．
使いたい再構築関数/近似リーマンソルバ/時間積分法をオブジェクト化してください．（例：new W3TBVD(slv);）

現在実装された再構築関数/近似リーマンソルバ/時間積分法は各サブディレクトリReconstruction, Flux, TimeIntegralの中身を確認してください．
再構築関数/近似リーマンソルバ/時間積分法を追加したい場合は，親ディレクトリにある各ヘッダーファイルが基底クラスとして記述されているため，そのルールに従って，ファイルの追加を行ってください．

## 生データ作成方法
main.cpp内に記述された，AnalyzerクラスのsetGeneratePreProcessedDataOption関数によりデータセット作成モードの切り替えを行うことができます．

例）計算領域全体で7ステンシルのデータを1000タイムステップまで保存する．
```
Analyzer* analyzer = new Analyzer();
analyzer->setGeneratePreProcessedDataOption(true, 1000, 7);
```
生成されたデータはPreProcessedDataディレクトリにフォルダで保存される．
フォルダ名はSolverクラスの定義時に設定できる．（例：Solver *slv1 = new Solver("hogehoge");）

## 前処理データ作成方法
DataProcessor.pyで，指定したPreProcessedDataを前処理しPostProcessedDataを作成します．
コード内のPARAMETER欄にある変数によってデータの生成数，削除するデータやモデルの入力を変更することができる．
論文の環境を実装したければ以下の変数のみ確認し，実行すれば良い．

- input_files→PreProcessedDataディレクトリのフォルダ名
- output_file→PostProcessedDataディレクトリ内の.datファイル


## モデル学習方法
**ModelTrainer_oh.py**※で，指定したPostProcessedDataを元にモデルの学習を行う．
モデルのパラメータ情報や入力，出力数等が書かれたONNXファイルと評価指標となる学習曲線とROC曲線（.pngファイル）が作成される．

コード内のPARAMETER欄にある変数によってデータの生成数，削除するデータやモデルの入力を変更することができる．
論文の環境を実装したければ以下の変数のみ確認し，実行すれば良い．

- input_file→PostProcessedDataディレクトリ内の.datファイル群
- output_file→ONNXディレクトリ内の.onnxファイル
- model_name→モデル名，output_fileのファイル名と同じにすれば良い
- batch_size→バッチサイズ
- epoch→学習するエポック数
- learning_rate→学習率

※ModelTrainer.pyはデータセットのワンホット表現非対応の旧式のコード．多分消す．

## 数値計算実行方法
Solveクラスで設定された計算条件をAnalyzerクラスが統合し，Solve関数で並行して数値計算を行い，グラフを出力します．
setGeneratePreProcessedDataOption関数のデータセット作成フラグがfalseになっていることを確認してください．
また，onnx_pathとisGPUで，NNBVD計算時の使いたいモデルを選び，モデル推論時にGPUを使用することができる．

1. setSolver関数で，AnalyzerとSolverを結びつける
2. 再構築関数/リーマンソルバ/時間積分法をオブジェクト化し，指定したSolverと結びつける
3. setProblem関数で，計算する問題を設定する（問題の詳細はSolver.cpp参照）．0→sine wave, 1→Square wave, 2→Jiang & Shu test, 3~→データセット生成に使う
4. Parameter.Hで計算条件を設定
5. make allでコンパイルして実行
