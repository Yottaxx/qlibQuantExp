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




--------------------------------------------------------------------------------
PARAMETER STATISTICS BY MODULE
--------------------------------------------------------------------------------
Module                    Params          Trainable       %Total    
-----------------------------------------------------------------
layers                    345,988         345,988         60.8      %
regime_encoder            93,632          93,632          16.4      %
factor_pooling            65,920          65,920          11.6      %
factor_gate               33,280          33,280          5.8       %
factor_id_emb             20,224          20,224          3.6       %
time_embedding            9,602           9,602           1.7       %
val_proj                  256             256             0.0       %
final_norm                256             256             0.0       %
head                      129             129             0.0       %
-----------------------------------------------------------------
TOTAL                     569,287         569,287         100.0     %

--------------------------------------------------------------------------------
DETAILED LAYER BREAKDOWN
--------------------------------------------------------------------------------
Layer Name                                         Shape                     Params      
---------------------------------------------------------------------------------------
val_proj.weight                                    [128, 1]                  128         
val_proj.bias                                      [128]                     128         
factor_id_emb.weight                               [158, 128]                20,224      
time_embedding.pos                                 [8, 128]                  1,024       
time_embedding.tau_base                            []                        1           
time_embedding.tau_norm.weight                     [128]                     128         
time_embedding.tau_norm.bias                       [128]                     128         
time_embedding.tau_fc1.weight                      [64, 128]                 8,192       
time_embedding.tau_fc1.bias                        [64]                      64          
time_embedding.tau_fc2.weight                      [1, 64]                   64          
time_embedding.tau_fc2.bias                        [1]                       1           
factor_gate.regime_norm.weight                     [128]                     128         
factor_gate.regime_norm.bias                       [128]                     128         
factor_gate.factor_norm.weight                     [128]                     128         
factor_gate.factor_norm.bias                       [128]                     128         
factor_gate.proj_gamma.weight                      [128, 128]                16,384      
factor_gate.proj_beta.weight                       [128, 128]                16,384      
regime_encoder.encoder.0.weight                    [64, 1328]                84,992      
regime_encoder.encoder.0.bias                      [64]                      64          
regime_encoder.encoder.3.weight                    [128, 64]                 8,192       
regime_encoder.encoder.3.bias                      [128]                     128         
regime_encoder.encoder.4.weight                    [128]                     128         
regime_encoder.encoder.4.bias                      [128]                     128         
layers.0.router.0.weight                           [64, 128]                 8,192       
layers.0.router.0.bias                             [64]                      64          
layers.0.router.2.weight                           [2, 64]                   128         
layers.0.router.2.bias                             [2]                       2           
layers.0.time_expert.mha.in_proj_weight            [384, 128]                49,152      
layers.0.time_expert.mha.out_proj.weight           [128, 128]                16,384      
layers.0.factor_expert.mha.in_proj_weight          [384, 128]                49,152      
layers.0.factor_expert.mha.out_proj.weight         [128, 128]                16,384      
layers.0.norm1.weight                              [128]                     128         
layers.0.norm1.bias                                [128]                     128         
layers.0.norm2.weight                              [128]                     128         
layers.0.norm2.bias                                [128]                     128         
layers.0.ffn.0.weight                              [128, 128]                16,384      
layers.0.ffn.0.bias                                [128]                     128         
layers.0.ffn.3.weight                              [128, 128]                16,384      
layers.0.ffn.3.bias                                [128]                     128         
layers.1.router.0.weight                           [64, 128]                 8,192       
layers.1.router.0.bias                             [64]                      64          
layers.1.router.2.weight                           [2, 64]                   128         
layers.1.router.2.bias                             [2]                       2           
layers.1.time_expert.mha.in_proj_weight            [384, 128]                49,152      
layers.1.time_expert.mha.out_proj.weight           [128, 128]                16,384      
layers.1.factor_expert.mha.in_proj_weight          [384, 128]                49,152      
layers.1.factor_expert.mha.out_proj.weight         [128, 128]                16,384      
layers.1.norm1.weight                              [128]                     128         
layers.1.norm1.bias                                [128]                     128         
layers.1.norm2.weight                              [128]                     128         
layers.1.norm2.bias                                [128]                     128         
layers.1.ffn.0.weight                              [128, 128]                16,384      
layers.1.ffn.0.bias                                [128]                     128         
layers.1.ffn.3.weight                              [128, 128]                16,384      
layers.1.ffn.3.bias                                [128]                     128         
final_norm.weight                                  [128]                     128         
final_norm.bias                                    [128]                     128         
factor_pooling.attention_pool.query                [1, 1, 128]               128         
factor_pooling.attention_pool.mha.in_proj_weight   [384, 128]                49,152      
factor_pooling.attention_pool.mha.out_proj.weight  [128, 128]                16,384      
factor_pooling.attention_pool.norm.weight          [128]                     128         
factor_pooling.attention_pool.norm.bias            [128]                     128         
head.weight                                        [1, 128]                  128         
head.bias                                          [1]                       1           
================================================================================

>>> [Sanity] loss=1.934618 grad_norm=1.520e+01 score_std=8.996e-03 label_std=1.325e+00 x_last_std=8.937e-01

================================================================================
COMPLETE MODEL CONFIGURATION (Resolved)
================================================================================

--- Architecture ---
Parameter                           Value                          Description              
------------------------------------------------------------------------------------------
d_model                             128                            Hidden dimension         
n_heads                             4                              Attention heads          
n_layers                            2                              Number of layers         
d_ff                                128                            FFN dimension            
num_alphas                          158                            Number of factors (N)    
context_len                         8                              Sequence length (T)      
dropout                             0.1                            Dropout rate             
initializer_range                   0.02                           Weight init std          

--- Loss & Training ---
Parameter                           Value                          Description              
------------------------------------------------------------------------------------------
main_loss                           mse                            Primary loss function    
loss_weights                        {'listmle': 1.0, 'mse': 1.0, 'ic': 1.0} Loss weight dict         
listmle_tau                         0.8                            ListMLE temperature      
rank_topk                           5                              RankNet top-k            
huber_delta                         1                              Huber delta              
mse_normalize                       ✗ OFF                          MSE normalize flag       

--- Regime-Adaptive Time Embedding ---
Parameter                           Value                          Description              
------------------------------------------------------------------------------------------
use_regime_time_embedding           ✓ ON                           Enable time embedding    
time_tau_min                        0.5                            Min tau (short memory)   
time_tau_max                        50                             Max tau (long memory)    
time_tau_init                       5                              Initial tau              
time_emb_init_std                   0.02                           Time emb init std        
time_decay_normalize                ✓ ON                           Normalize decay weights  

--- Regime-Adaptive Factor Gate ---
Parameter                           Value                          Description              
------------------------------------------------------------------------------------------
use_regime_factor_gate              ✓ ON                           Enable factor gate (FiLM)
factor_gate_scale                   0.5                            Gate scale (γ range)     
factor_gate_shift_scale             0                              Gate shift scale (β)     

--- MoE Router ---
Parameter                           Value                          Description              
------------------------------------------------------------------------------------------
router_noise                        0.1                            Logit noise std          
router_temperature                  1                              Softmax temperature      
router_z_loss_coef                  0.01                           Z-loss coefficient       
router_use_layer_summary            ✗ OFF                          Use layer summary token  

--- Feature Selection ---
Parameter                           Value                          Description              
------------------------------------------------------------------------------------------
use_feature_selection               ✗ OFF                          Enable feature selection 
selection_reg_lambda                1e-05                          Sparsity regularization  
selection_temperature               0.1                            Gumbel temperature       
selection_noise_std                 0.5                            Selection noise std      

--- Positional Encoding ---
Parameter                           Value                          Description              
------------------------------------------------------------------------------------------
use_alibi                           ✗ OFF                          Use ALiBi bias           

--- Regime Context Encoder ---
Parameter                           Value                          Description              
------------------------------------------------------------------------------------------
use_external_macro                  ✓ ON                           Use external macro features
d_macro_input                       1328                           Macro input dimension    
regime_macro_dropout                0.1                            Macro dropout            
regime_internal_mode                long                           Internal mode (short/long)
regime_internal_lag                 5                              Internal lag steps       
regime_internal_use_batch_stats     ✓ ON                           Use batch statistics     
regime_internal_tail_threshold      2                              Tail threshold           

--- Pooling ---
Parameter                           Value                          Description              
------------------------------------------------------------------------------------------
pooling_alpha                       0.7                            Attention vs mean weight 

--- Trainer Configuration ---
Parameter                           Value                          Description              
------------------------------------------------------------------------------------------
lr                                  0.0001                         Learning rate            
epochs                              40                             Number of epochs         
batch_size                          128                            Batch size (stocks/day)  
grad_accum_steps                    5                              Accumulate K days/step   
early_stop                          3                              Early stop patience      
use_warmup                          ✓ ON                           Enable LR warmup         
warmup_config                       682/13640 steps                Warmup steps/ratio       
total_steps                         13640                          Total training steps     
random_seed                         42                             Random seed              
num_workers                         0                              DataLoader workers       
label_dim                           1                              Label dimension          
market_state_path                   market_state_csi300.pkl        Market state file        
market_state_shift                  0                              Market state shift       
market_state_strict                 ✓ ON                           Strict market state      
================================================================================