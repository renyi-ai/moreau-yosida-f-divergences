# Moreau-Yosida f-divergences

This repository contains source code to reproduce results presented in the ICML 2021 paper [Moreau-Yosida f-divergences](https://arxiv.org/abs/2102.13416)

**Dávid Terjék** (Alfréd Rényi Institute of Mathematics) **[dterjek@renyi.hu](mailto:dterjek@renyi.hu)**

* For the Gaussian and Categorical experiments , we used torch version 1.7.1. and the tqdm package.

* For the CIFAR-10 GAN experiments, we used tensorflow-gpu 1.14.0 and the numpy and tqdm packages.

* To run the GAN experiments, first download the CIFAR-10 dataset from https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz 
and extract into the "data/cifar10" directory (e.g. /home/username/data/cifar10),
and run e.g.

   * python gan_cifar10.py --run_name <string: name of the tensorboard log directory>

     * most relevant arguments:\
--alpha <float: alpha parameter of Moreau-Yosida regularization>\
--init_beta <float: initial value of beta parameter of Moreau-Yosida regularization>\
--final_beta <float: final value of beta parameter of Moreau-Yosida regularization>\
--discriminator <one of ["base", "quotient"]>\
--divergence <one of ["trivial", "kl", "reverse_kl", "chi2", "reverse_chi2", "hellinger2", "js", "jeffreys", "triangular", "tv"]>\
--forward or --reverse <switch: controls forward or reverse MYf-GAN formulation>

     * other arguments:\
--log_dir <string: path to a directory where the log directory is created>\
--log_freq <integer: write logs every number of iterations>\
--iterations <integer: number of training iterations>\
--batch_size <integer: number of real or fake samples, actual batch size is twice this number>\
--val_freq <integer: evaluate Inception Score every number of iterations>\
--val_size <integer: batch size for evaluation>\
--random_seed <integer: sets the random seed>\
--lr <float: learning rate>\
--b1 <float: beta 1 parameter of Adam optimizer>\
--b2 <float: beta 2 parameter of Adam optimizer>\
--ema_decay <float: coefficient for exponential moving average of generator weights>\
--lambda_gp <float: gradient penalty multiplier>\
--K <float: maximum required gradient norm parameter for gradient penalty>\
--n_critic <integer: the critic is trained for this number of gradient descent steps per iteration>\
--penalized or --non_penalized <switch: controls penalized mean deviation or basic formulation>\
--decay_lr or --const_lr <switch: controls learning rate decay>\
--tight or --loose <switch: controls tight or non-tight (original f-GAN) formulation>