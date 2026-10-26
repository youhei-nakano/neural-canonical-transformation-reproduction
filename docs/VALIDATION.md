# 公開前検証

2026-10-26実施。今回のGitHub整理では追加学習せず、元実験成果物を保全した。

| 検証 | 結果・範囲 |
|---|---|
| runnerとsetupの構文 | Pythonのpy_compile／AST parseで成功 |
| ポータブル版32粒子runner | 保存済み自作initial.pt・trained.pt・loss.csvを無視対象runs/へコピーし、evaluateのみ再実行。初期／最終KL・共分散、周波数、モード、float32／shadow幾何学の全6グループが元metricsと完全一致 |
| 4粒子公開版 | 数値処理は同じで、パス・上書き防止だけ変更。構文と成果物は検査。整理時の再学習は行っていない |
| 上流 | 別のローカルcloneでsetupの既存checkout検証に成功。固定commitとclean状態を確認 |
| 依存 | 元の専用venvのfreezeを固定。新規venvへの全再インストールは未実施 |
| 小結果 | 4粒子loss600行、32粒子loss5000行、JSON・周波数CSV・PNGを確認。図は元実験で実際に開いて確認済み |
| リンク・公開対象 | validate_package.pyで相対リンク、サイズ、除外対象、個人絶対パス・一般的credentialパターンを検査 |
| 除外 | venv、vendor、runs、checkpoint、NPZ、ログ、cache、個人絶対パス、他プロジェクト資料を掲載しない |
| CI | GitHub Actionsは追加しない。クラウド再学習・課金なし |

ローカルの静的検証を再実行するコマンド：

```sh
python3 scripts/validate_package.py
```

秘密情報検査は上記パターンと掲載対象の目視確認であり、あらゆる秘密情報を検出できる保証ではない。元checkpointは公開していないので、clone後に評価だけを走らせる場合は先に学習する。元の学習済みモデルとの数値一致は同じMac・同じ既存venvで検証しており、他のOS・Torch・BLASでのbitwise一致は未検証。
