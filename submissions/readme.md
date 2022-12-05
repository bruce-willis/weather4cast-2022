# Submissions:
Since there is a limit of 5 submissions in the leaderboard, we collected all results here.

To parse all results and find best overall submission and best submission per region, run [`parse_all_submissions.py`](./parse_all_submissions.py) script.


# Baseline
### baseline_stage2_mIoULoss_tune
* Epoch 11: 0.1869852 (https://pastebin.com/bHsEVma0)
* Epoch 15: 0.1911728 (https://pastebin.com/SU21zZWU)
* Epoch 25: 0.1903876 (https://pastebin.com/YbxEixMA)
* Epoch 38: 0.1940629 (https://pastebin.com/kShfyXVC)

### baseline_stage2_BCELoss_tune_from_mIoU
* Epoch 23: 0.2130381 (https://pastebin.com/8ydQ4LGJ)

### baseline_stage2_BCELoss_train_all (train + valid dataset for training)
* Epoch 24: 0.2218933 (https://pastebin.com/pqwVb6jS)
* Epoch 33: 0.2144217 (https://pastebin.com/WKz3W8Pk)
* Epoch 48 (tune): 0.203126 (https://pastebin.com/E8mPqrN4)
* Epoch 53 (last, tune): 0.166404 (https://pastebin.com/SEJNuRqJ)

### baseline_stage2_BCELoss_improved_train_all (train + valid dataset for training)
* Epoch 12: 0.2341024 (https://pastebin.com/abQeHPwL)
* Epoch 15: 0.2454266 (https://pastebin.com/ZpJcP0Em)
* Epoch 31: 0.2337205 (https://pastebin.com/r1qVVk98)
* Epoch 32: 0.240163 (https://pastebin.com/3qzr4T6L)
* Epoch 33: 0.2436366 (https://pastebin.com/MpMNgiRR)
* Epoch 83 (tune): 0.2367744 (https://pastebin.com/xdX2ebZV)
* Epoch 88 (tune): 0.22088 (https://pastebin.com/WwigWJk3)
* Epoch 89 (last, tune): 0.2269136 (https://pastebin.com/0VyH273R)

### baseline_stage2_BCELoss_improved_without_transpose_train_all (train + valid dataset for training)
* Epoch 32: 0.2337182 (https://pastebin.com/QVkn9Ug5)
* Epoch 33: 0.2460254 (https://pastebin.com/YaiqaKvM)
* Epoch 34 (=last): 0.2282431 (https://pastebin.com/21F2BcyM)


# Swin
### swin_mIoULoss:
* Epoch 0: 0.1466337 (https://pastebin.com/KGZZ7D0F)
* Epoch 2: 0.1899196 (https://pastebin.com/byVAhNJC)
* Epoch 4: 0.1864095 (https://pastebin.com/1cjt1qHR)

### swin_dice_focalLoss_channel_conv_tune_train_all:
* Epoch 0: 0.2044082 (https://pastebin.com/9rE2mtpv)
* Epoch 1: 0.2101684 (https://pastebin.com/0auuq9ru)

### swin_bceLoss:
* Epoch 1: 0.2483156 (https://pastebin.com/Ruq26ZA4)
* Epoch 2: 0.2510357 (https://pastebin.com/7xLvkfib)
* Epoch 3: 0.2516998 (https://pastebin.com/jCRqX8S9)
#### threshold
* Epoch 3 (0.2 threshold): 0.2274491 (https://pastebin.com/kEZq3gQS)
* Epoch 3 (0.65 threshold): 0.1937764 (https://pastebin.com/CyrU7STV)
* Epoch 3 (0.7 threshold): 0.1615253 (https://pastebin.com/iRVq81Ux)
#### masking
* before sigmoid activation doesn't change anything since it doesn't change the sign lol.
* Epoch 3 (masks_all): 0.251463 (https://pastebin.com/TZpTQ8kA)
* Epoch 3 (masks_nan_9): 0.2515391 (https://pastebin.com/puNMWLaL)

### swin_bceLoss_tune_train_all:
* Epoch 4: 0.2490757 (https://pastebin.com/QjnPGJW0)
* Epoch 5: 0.2495659 (https://pastebin.com/Dg2xNF3F)           

### swin_bceLoss_16bit_tune + checkpoint gradient
* Epoch 0: 0.2433458 (https://pastebin.com/Zu80aPFw)
* Epoch 1: 0.2428661 (https://pastebin.com/UxS88xYh)
* Epoch 1: 0.2428749 (16 bit predict) (https://pastebin.com/LUG4Hmue)
* Epoch 3: 0.251606 (https://pastebin.com/LPqCsLmh)
* Epoch 6 (last): 0.2460025 (https://pastebin.com/JERLCiXf)

### swin_bceLoss_adabelief_train_all + without checkpoint weights
* Epoch 3: 0.2441566 (https://pastebin.com/E65uRjny)
* Epoch 4: 0.2350705 (https://pastebin.com/2uRNNuwK)

### swin_bceLoss_adabelief_train_all + with checkpoint weights
* Epoch 3: 0.2584281 (https://pastebin.com/UfiNJaAW)
* Epoch 4: 0.2663436 (https://pastebin.com/wQbZeu5j)

### swin_bceLoss_channel_conv (swin weights from swin_bceLoss Epoch 3)
* Epoch 0: 0.2346285 (https://pastebin.com/HfUBaAaf)
* Epoch 1: 0.2438615 (https://pastebin.com/JPCCg6wR)
* Epoch 2: 0.2411255 (https://pastebin.com/Gqs1f1G2)
* Epoch 3 (last): 0.2345053 (https://pastebin.com/4k8p6UAz)

### swin_bceLoss_channel_conv_adabelief_train_all + without checkpoint
* Epoch 3: 0.2588333 (https://pastebin.com/UhBvipX4)
* Epoch 4: 0.2589834 (https://pastebin.com/x8MUYk8v)

### swin_bceLoss_channel_conv_adabelief_train_all + with checkpoint weights
* Epoch 0: 0.2469998 (https://pastebin.com/vTsLBT6K)
* Epoch 3: 0.2589735 (https://pastebin.com/8Y3D8mwr)
* Epoch 4: 0.2430648 (https://pastebin.com/81MScAPx)

### swin_bceLoss_upsample
* Epoch 1: 0.2239677 (https://pastebin.com/775cZTtv)


# Vivit
### vivit_bceLoss_8_8_768
* Epoch 2: 0.1932861 (https://pastebin.com/m1VFzEqC)



# Combine
### swin_bceLoss Epoch 3 +  baseline_stage2_BCELoss_improved_train_all Epoch 33
* all from baseline except 007, 76 region for 2022: 0.2572105 (https://pastebin.com/LExVqmjh)
### all best
* take best per region: 0.2859607 (https://pastebin.com/bHmsvSgX)


# Ensemble
### Majory vote
* from all submissions (50): 0.2649146 (https://pastebin.com/hKgS6Ge3)
* after removing outliers (36 left): 0.2646577 (https://pastebin.com/EezC1bTm)
