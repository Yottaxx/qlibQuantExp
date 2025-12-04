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
            