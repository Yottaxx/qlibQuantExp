      # 2) 进入之前训练好的 experiment / recorder
        exp_name = "Official_Alignment_RST_MoE"
        rec_id = "<你的 recorder_id>"  # 可以在命令行输出 / qlib UI 中看到

        with R.start(experiment_name=exp_name, recorder_id=rec_id):
            rec = R.get_recorder()
            # 3) 加载已保存的模型对象（训练脚本里 R.save_objects(model=model) 写进去的）
            model = rec.load_object("model")

            # 4) 直接预测
            pred_infer = model.predict(dataset, segment="infer")
            # pred_infer 是 pd.Series，MultiIndex: (datetime, instrument)

            # 5) 保存或下游操作
            pred_infer.to_pickle("pred_rst_moe_infer.pkl")
            print(pred_infer.head())
            

[12536:MainThread](2025-12-16 20:24:08,385) INFO - qlib.timer - [log.py:127] - Time cost: 0.231s | CSRankNorm Done
[12536:MainThread](2025-12-16 20:24:08,406) INFO - qlib.timer - [log.py:127] - Time cost: 5.737s | fit & process data Done
[12536:MainThread](2025-12-16 20:24:08,406) INFO - qlib.timer - [log.py:127] - Time cost: 40.458s | Init data Done
[INFO] Segment 'train': loaded DataFrame with shape (1112999, 158) using Method 2: handler.fetch()
Processing train: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████| 3711/3711 [00:22<00:00, 165.60day/s]
[INFO] Segment 'valid': loaded DataFrame with shape (1112999, 158) using Method 2: handler.fetch()
Processing valid: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████| 3711/3711 [00:21<00:00, 173.90day/s]
[INFO] Segment 'test': loaded DataFrame with shape (1112999, 158) using Method 2: handler.fetch()
Processing test: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 3711/3711 [00:22<00:00, 162.73day/s]
Saved market state: market_state_csi300.pkl shape=(11133, 816)

>>> [Sanity] loss=4.565823 grad_norm=1.189e+00 score_std=4.559e-02 label_std=1.002e+00 x_last_std=1.007e+00
