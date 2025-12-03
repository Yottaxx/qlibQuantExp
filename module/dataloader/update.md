fit中update
# 自动判断是否需要 DDP Sampler
        if torch.distributed.is_initialized():
            sampler = DistributedDailyBatchSampler(
                train_tsds, 
                self.batch_size, 
                shuffle=True
            )
        else:
            # 单卡模式退回之前的 Simple Sampler (其实 Distributed Sampler 也兼容单卡，num_replicas=1)
            # 为了简单，直接统一用 DistributedDailyBatchSampler 即可！
            sampler = DistributedDailyBatchSampler(
                train_tsds, 
                self.batch_size,
                num_replicas=1,
                rank=0,
                shuffle=True
            )

        train_loader = DataLoader(
            dataset=train_tsds,
            batch_sampler=sampler, # 传给 batch_sampler
            num_workers=4,
            pin_memory=True
        )
        
        # ... Training Loop ...
        for epoch in range(self.epochs):
            # [关键] DDP 必须在每个 epoch 开始前 set_epoch
            train_loader.batch_sampler.set_epoch(epoch)