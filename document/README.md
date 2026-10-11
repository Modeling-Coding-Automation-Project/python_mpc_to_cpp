# python_mpc_to_cpp ドキュメント

このドキュメントは、本リポジトリの `mpc_utility` / `python_mpc` と `sample` の使い方を中心にまとめたガイドです。
クラス内部アルゴリズムの詳細導出ではなく、**実際にどう使うか**に焦点を当てています。

## 目次

1. [全体像とディレクトリ構成](./01_overview_and_architecture.md)
2. [線形MPCの使い方（LTI/LTV）](./02_linear_mpc_usage.md)
3. [適応MPCの使い方（Adaptive MPC）](./03_adaptive_mpc_usage.md)
4. [非線形MPCの使い方（SQP / OpEn）](./04_nonlinear_mpc_usage.md)
5. [コード生成と Python/C++ インターフェース対応](./05_code_generation_and_cpp_interface.md)
6. [シミュレーションと実行フロー](./06_simulation_and_run_flow.md)
7. [利用している外部ライブラリ要約](./07_external_libraries_summary.md)

## 最短で読む順番

1. [全体像とディレクトリ構成](./01_overview_and_architecture.md)
2. [線形MPCの使い方（LTI/LTV）](./02_linear_mpc_usage.md)
3. [コード生成と Python/C++ インターフェース対応](./05_code_generation_and_cpp_interface.md)

## このガイドの前提

- 主に `sample` 内コードを根拠に説明しています
- `mpc_utility` / `python_mpc` の公開的な利用面を整理しています
- 外部サブモジュールの説明は、このリポジトリ内で完結するよう要点を再掲しています
