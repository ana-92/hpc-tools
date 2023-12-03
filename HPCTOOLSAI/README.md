
**Note:** All this project is according to the following instructions: [Lab AI](https://awesome-archduke-bec.notion.site/Lab-AI-HPC-Tools-e647da3f04dc4e66a40692da0d5f9c27)

## Baseline Implementation
The modified code uses PyTorch Lightning to structure the deep learning training process. It defines a custom Lightning module, LightningCIFAR10, with the model, loss function, and optimizer. The training step computes the loss and accuracy, and the Trainer class handles the training loop. TensorBoard is utilized for logging training metrics, and the total training time is recorded. Profiling is performed using PyTorch's torch.profiler, and the results, such as self CPU time, are logged to TensorBoard for analysis.

This particular case study focuses on training a Convolutional Neural Network (CNN) on the CIFAR-10 dataset, a widely used dataset for image classification. The CNN architecture consists of convolutional and fully connected layers. The code showcases the implementation of key deep learning practices, such as data loading, model definition, training loop, and performance profiling, using PyTorch and PyTorch Lightning.


## Results

*All this results can be check with slurm-xxx.out file.*

| Configuration                  | Training Time        | Speedup |
| ------------------------------ | -------------------- | ------- |
| 1 Nvidia A100 (Baseline)       | 2 minutes 0 seconds  | -----   |
| 2 Nodes w/ 2 Nvidia A100 (DDP) | 1 minute 5 seconds   | 1.85x   |
| 2 Nodes w/ 2 Nvidia A100 (DP)  | 2 minutes 21 seconds | 0.85x   |

  
### 2 Nodes with 2 Nvidia A100 (DDP):

- The DDP (Data Parallelism) configuration shows a speedup of **1.85x** compared to the baseline.
- This indicates that using two nodes with two Nvidia A100 GPUs in a distributed data parallel setup significantly improved training time, almost doubling the speed.

### 2 Nodes with 2 Nvidia A100 (DP):

- The DP (Data Parallelism) configuration, on the other hand, has a speedup of **0.85x** compared to the baseline.
- This suggests that using two nodes with two Nvidia A100 GPUs in a traditional data parallel setup did not provide as much acceleration as the DDP configuration. In fact, it appears to be slightly slower than the baseline.

## Conclusion:
- Distributed Data Parallelism (DDP) is more effective in improving training speed compared to traditional Data Parallelism (DP) in this scenario.
- Leveraging multiple nodes and GPUs with DDP results in a significant speedup, making the training process nearly twice as fast as the baseline. -