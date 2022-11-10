# Current top:
1. swin_bceLoss Epoch 3: 0.2516998
2. swin_bceLoss_16bit_tune Epoch 3: 0.251606 
3. swin_bceLoss Epoch 2: 0.2510357
4. swin_bceLoss Epoch 1: 0.2483156
5. swin_bceLoss_16bit_tune Epoch 6: 0.2460025 

# Models:
# Baseline
### baseline_stage2_mIoULoss_tune
* Epoch 11: 0.1869852 (https://pastebin.com/bHsEVma0)
* Epoch 15: 0.1911728 (https://pastebin.com/SU21zZWU)
* Epoch 25: 0.1903876 (https://pastebin.com/YbxEixMA)
* Epoch 38: 0.1940629 (https://pastebin.com/kShfyXVC)


### baseline_stage2_BCELoss_tune_from_mIoU
* Epoch 23: 0.2130381 (https://pastebin.com/8ydQ4LGJ)


### baseline_stage2_BCELoss_train_all (train + valid dataset for training)
* Epoch 24: ?
* Epoch 33: 0.2144217 (https://pastebin.com/WKz3W8Pk)
* Epoch 48 (tune): 0.203126 (https://pastebin.com/E8mPqrN4)
* Epoch 53 (last, tune): 0.166404 (https://pastebin.com/SEJNuRqJ)


### baseline_stage2_BCELoss_improved_train_all (train + valid dataset for training)
* Epoch 12: 0.2341024 (https://pastebin.com/abQeHPwL)
* Epoch 15: 0.2454266 (https://pastebin.com/ZpJcP0Em)
* Epoch 31: \?
* Epoch 32: \?
* Epoch 33: \?


### baseline_stage2_BCELoss_improved_without_transpose_train_all (train + valid dataset for training)
* ?


# Swin
### swin_mIoULoss:
* Epoch 0: 0.1466337 (https://pastebin.com/KGZZ7D0F)
* Epoch 2: 0.1899196 (https://pastebin.com/byVAhNJC)
* Epoch 4: 0.1864095 (https://pastebin.com/1cjt1qHR)


### swin_bceLoss:
* Epoch 1: 0.2483156 (https://pastebin.com/Ruq26ZA4)
* Epoch 2: 0.2510357 (https://pastebin.com/7xLvkfib)
* Epoch 3: 0.2516998 (https://pastebin.com/jCRqX8S9)


### swin_bceLoss_tune_train_all:
* ?


### swin_dice_focalLoss_tune_train_all:
* ?


### swin_bceLoss_16bit_tune + checkpoint gradient
* Epoch 0: 0.2433458 (https://pastebin.com/Zu80aPFw)
* Epoch 1: 0.2428661 (https://pastebin.com/UxS88xYh)
* Epoch 1: 0.2428749 (16 bit predict) (https://pastebin.com/LUG4Hmue)
* Epoch 3: 0.251606 (https://pastebin.com/LPqCsLmh)
* Epoch 6 (last): 0.2460025 (https://pastebin.com/JERLCiXf)


### swin_bceLoss_channel_conv (swin weights from swin_bceLoss Epoch 3)
* Epoch 0: 0.2346285 (https://pastebin.com/HfUBaAaf)
* Epoch 1: 0.2438615 (https://pastebin.com/JPCCg6wR)
* Epoch 2: 0.2411255 (https://pastebin.com/Gqs1f1G2)
* Epoch 3 (last): 0.2345053 (https://pastebin.com/4k8p6UAz)


# Vivit
### vivit_bceLoss_8_8_768
* Epoch 2: 0.1932861 (https://pastebin.com/m1VFzEqC)
