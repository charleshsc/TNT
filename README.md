# TNT Re-implementation :car:

Paper: [TNT: Target-driveN Trajectory Prediction](https://arxiv.org/pdf/2008.08294.pdf)

**Note.** If you cannot see the formulation correctly, you need to install this [plugin](https://chrome.google.com/webstore/detail/mathjax-plugin-for-github/ioemnmodlmafdkllaclgeombjnmnbima/related) in Chrome

This paper focus on the task of future trajectory prediction for moving road agents, where both the states and the targets are represented by their physical locations $(x_t, y_t)$. 

It total contains 4 parts:

1. Scene context encoding.

   Since HD map is available, it use the VectorNet to encode the context. Specifically, polylines are used to abstract the HD map elements $c_P$ (lanes, traffic signs) and agent trajectories $s_P$; a subgraph network is applied to encode each polyline, which contains a variable number of vectors; then a global graph is used to model the interactions between polylines. The output is a global context feature $x$ for each modeled agent.

2. Target prediction

   It models the potential future targets:

   $$
   T = \{\tau^n \} = \{(x^n,y^n) + (\Delta x^n, \Delta y^n)  \}_{n=1}^N
   $$
   
   The distribution over targets can be modeled via a discrete-continuous factorization:

   $$
   p(\tau^n | x) = \pi (\tau^n | x) \cdot \mathcal{N} (\Delta x^n | v_x^n (x)) \cdot \mathcal{N} (\Delta y^n | v_y^n (x))
   $$
   
   The loss function for training this stage is given by 

   $$
   \mathcal{L_{S1}} = \mathcal{L_{cls}} (\pi, u) + \mathcal{L_{offset}} (v_x, v_y, \Delta x^u, \Delta y^u)
   $$

3. Target-conditioned motion estimation

   It models the likelihood of a trajectory given a target as $p(s_F|\tau, x) = \prod_{t=1}^T p(s_t|\tau, x)$. It takes context feature $x$ and a target location $\tau$ as input, and outputs one most likely future trajectory $[\hat{s_1}, \dots, \hat{s_T}]$ per target. During training stage, it applies a teacher forcing technique by feeding the ground truth location $(x^u, y^u)$ as target. The loss term for this stage is the distance between predicted states $\hat{s_t}$ and ground truth $s_t$ :

   $$
   \mathcal{L_{S2}} = \sum_{t=1}^T \mathcal{L_{reg}} (\hat{s_t}, s_t)
   $$
   
   where $\mathcal{L}_{reg}$ is implemented as Huber loss over per-step coordinate offsets.

4. Trajectory scoring and selection

   It use a maximum entropy model to score all the $M$ trajectories from the second stage:
   $$
   \phi(s_F | x) = \frac{\exp(g(s_F, x))}{\sum_{m=1}^M \exp (g(s_F^m, x))}
   $$
   the ground truth score of each predicted trajectory is defined by its distance to ground truth  trajectory
   $$
   \psi(s_F) = \frac{\exp (-D(s,s_{GT})/ \alpha)}{\sum_{s'} \exp (-D(s', s_{GT})/ \alpha)}
   $$
   The loss function for training this stage is the cross entropy between the predicted scores and ground truth scores:
   
   $$
   \mathcal{L_{S3}} = \mathcal{L_{CE}} (\phi(s_F | x), \psi(s_F))
   $$
   
   Then we first sort the trajectories according to their score in descending order, and then pick them greedily; if one trajectory is distant enough from all the selected trajectories, we select it as well, otherwise exclude it.



The total loss function is as follow:


$$
\mathcal{L} = \lambda_1 \mathcal{L_{S1}} + \lambda_2 \mathcal{L_{S2}} + \lambda_3 \mathcal{L_{S3}}
$$

The total framework is as follows:

<img src=".\imgs\TNT_framework.png" alt="TNT_framework" style="zoom:67%;" />

At inference time, it works as follows:

1. encode context;
2. sample N target candidates as input to the target predictor, take the top $M$ targets as estimated by $\pi(\tau | x)$;
3. take the MAP trajectory for each of the $M$ targets from motion estimation model $p(s_F | \tau, x)$;
4. score the $M$ trajectories by $\phi(s_F|\tau, x)$ and select a final set of $K$ trajectories.



---

## Table of Contents

- [Environment](#Environment)
- [Usage](#Usage)
- [Results and Visualization](#Results and Visualization)
- [Citations](#Citations)
- [Parameters](#Parameters)
- [References](#References)
---



## Environment

Training on Linux server; CUDA version 10.1

> pytorch == 1.7.1, torchvision==0.8.2, torchaudio==0.7.2, cudatoolkit=10.1
>
> apt-get install -y python3-opencv
>
> pip install opencv-python

And need to install the [argoverse-api](https://github.com/argoai/argoverse-api).
Except follow these instructions in the offical web, there are still others package need to install.
If you encounter this error:
```angular2html
ImportError: libGL.so.1: cannot open shared object file: No such file or directory
```
You need to follow these instructions:
```angular2html
sudo apt update
sudo apt install libgl1-mesa-glx
```



## Usage

After configuring the Environment, you can follow the instruments below to train the model end-to-end.

1. download the dataset from the [website](https://www.argoverse.org/data.html#download-link), notice that we are predicting the agent trajectory, you need to download the Argoverse Motion Forecasting and pull the data in the root path of this repository.

   An example  folder structure:

   ```
   .
   | -- TNT
   |    | -- Dataset
   |    | -- eval
   |    | -- ...
   |    | -- train.py
   | -- train
   |    | -- data
   |         | -- *.csv
   | -- val
   |    | -- data
   |         | -- *.csv
   | -- test
   |    | -- data
   |         | -- *.csv
   ```

   or you can add the suffix `--train_data_locate`  or `--val_data_locate` when running the code to specify the data location manually.

2. After preparing the data, you can begin to train the model. And there are several parameters which you can add to the suffix when running the training code and you can refer the table to look at each parameters meaning.

   Simple training code just as follows:

   ```
   python train.py --gpu 0 --batch_size 12 --num_worker 6
   ```

   notice that this repository only supports for single-GPU running, and you can specify which gpu card to use.

3. infer the model.

   Simple training code just as follows:

   ```
   python test.py --gpu 0
   ```

**Note.** The root name of this repository do not to be changed, or it cannot find the save path. Keep this repository name `TNT`.

~~**Note.** it's better to set the `num_worker`  as 0 since if being set more than 0, it will occur some unexpected bugs. The reason may be that it is complex in the dataloader and due to my little knowledge, I cannot make it to become parallel.~~

**Note.** In order to deal with the dataset, I manually remove those map data whose length is not equal 18 in order to pack 
it as torch.tensor where every data dimension must be the same.



## Results and Visualization





## Citations

```bibtex
@misc{hu2021TNTimple,
	auther = 		{Shengchao Hu},
	title = 		{TNT Re-implementation},
	howpublished = 	        {\url{https://github.com/charleshsc/TNT}},
	year = 			{2021}
}
```



## Parameters

| Name              | Default       | meaning                                          |
| ----------------- | ------------- | ------------------------------------------------ |
| N                 | 1000          | over-sample number of target candidates          |
| M                 | 50            | keep a small number for processing               |
| alpha             | 0.01          | the temperature in scoring psi                   |
| last_observe      | 30            | the last observe in the trajectory for the agent |
| total_step        | 49            | the total step in the trajectory for the agent   |
| batch_size        | 2             | the batch size for the training stage            |
| gpu               | 0             | if the device is cuda then which card to use     |
| lambda_1          | 0.1           | the weight for the loss1                         |
| lambda_2          | 1.0           | the weight for the loss2                         |
| lambda_3          | 0.1           | the weight for the loss3                         |
| K                 | 6             | the final number of candidate trajectory         |
| min_distance      | 0.5           | the min distance in selection stage              |
| seed              | 12345         | the seed to init the random                      |
| learning_rate     | 0.001         | the learning rate for the optimizer              |
| train_data_locate | ../train/data | the train dataset root directory                 |
| val_data_locate   | ../val/data   | the val dataset root directory                   |
| test_data_locate  | ../test/data  | the test dataset root directory                  |
| num_worker        | 0             | the num worker for the data loader               |
| epochs            | 50            | the num epochs for the training                  |
| steps_to_print    | 10            | the steps to print the loss                      |
| epochs_to_save    | 1             | the num epochs to save the model                 |
| resume            | None          | the pretrained model to reload                   |
| ft                | True          | fine-tuning in optimizer                         |
| miss_threshold    | 2.0           | The miss threshold in the eval stage for the MR  |


## References

1. The VectorNet re-implementation refers to this [repo](https://github.com/Liang-ZX/VectorNet).
2. Hang Zhao, Jiyang Gao, Tian Lan, Chen Sun, Benjamin Sapp, Balakrishnan Varadarajan, Yue Shen, Yi Shen, Yuning Chai, Cordelia Schmid, et al. Tnt: Target-driven trajectory prediction. arXiv preprint arXiv:2008.08294, 2020.

