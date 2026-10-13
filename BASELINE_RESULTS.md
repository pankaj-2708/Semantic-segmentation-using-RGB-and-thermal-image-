# Baseline Results: Custom RGB-Thermal Segmentation

## Model
Custom Dual-Stream U-Net with Multi-Scale Cross-Modal Attention
Trainable parameters: 2,997,071
Input resolution: 480 x 640
Output classes: 9

## Dataset
Dataset: MFNet RGB-Thermal
Training entries: 1,568
Unique training scenes: 784
Validation scenes: 392
Train-validation overlap after removing _flip suffix: 0

## Training
Epochs completed: 50
Best checkpoint epoch: 34
Best validation mIoU: 0.345645 (34.56%)
Optimizer: AdamW
Initial learning rate: 0.001
Loss: Weighted Cross-Entropy + Dice

## Observations
Training mIoU increased substantially while validation performance plateaued.
Validation loss increased during later epochs, indicating a generalization problem.
The next experiment will investigate data augmentation and other causes of overfitting.

## Saved artifacts
- checkpoints/custom_dual_unet_attention/best_checkpoint.pt
- plots/epoch_loss.png
- plots/epoch_metrics.png
- plots/learning_rate.png
- plots/step_train_loss.png
- plots/epoch_050_predictions.png
- plots/latest_predictions.png
