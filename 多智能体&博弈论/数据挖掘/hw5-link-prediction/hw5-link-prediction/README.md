# Programming Assignment 5 - Node2Vec

In this assignment, we will implement the Node2Vec algorithm for node embedding, and use the learned embeddings to perform a Link Prediction task.

## 1. Introduction

### 1.1. Dataset

You are given an undirected network with 16,863 nodes and 46,116 edges. Your task is to learn node embeddings on the given graph using the Node2Vec algorithm and use the learned embeddings to perform link prediction on a given test dataset. The test set consists of 10,246 (`src`, `dst`) node pairs obtained from the original network. You need to predict the probability that a link exists between the two nodes using the learned node embeddings for each node pair.

The dataset files (under `data/`) include

1. `p3_graph.csv`. This csv file contains the graph to be used for training. It contains 46,116 (`src`, `dst`) pairs. The provided code includes the necessary functions for loading the graph.
2. `p3_test.csv`. This csv file contains the test data, in the format of `id, src, dst`. For each (`src`, `dst`) pair, give the probability that a link exists between the two nodes.
3. `label_reference.csv`. This csv file contains 300 labels, which we release as a validation set to verify your algorithm. The file is in the format of `id, label`, where `id` corresponds to the `id` in `p3_test.csv`, and `label` is 1 if there is an edge, and 0 otherwise.

### 1.2. Environment

Python >= 3.6 is required for this project. Additionally, this project requires `torch`, `scikit-learn` and `tqdm`.

Please refer to the [official website of PyTorch](https://pytorch.org/get-started/locally/) for instructions on installing PyTorch. 

After installing PyTorch, install `scikit-learn` and `tqdm` via `conda` or `pip`,

```sh
conda install scikit-learn
conda install tqdm
```

- Note that the CUDA version of PyTorch is NOT required (though recommended). The Node2Vec model in this project is small enough to run on CPU-only devices.
- Please feel free to include other packages as needed. However, **do NOT use existing Node2Vec modules**, including but not limited to `Word2Vec` from `gensim` and `Node2Vec` from `pytorch-geometric`.

### 1.3. Files

- `data_utils.py` contains a PyTorch `Dataset` for link prediction and a corresponding collator function.
- `loss.py` contains the Negative Sampling Loss. You need to complete its implementation.
- `walker.py` contains the biased random walker. You need to complete its implementation.
- `metrics.py` contains a function for computing AUC scores.
- `model.py` contains a simple Node2Vec model and a Sigmoid classifier.
- `node2vec_trainer.py` contains the training process for the Node2Vec algorithm. You need to complete part of the training loop.
- `main.py` is the entrance of the program. It loads the data, runs the Node2Vec algorithm to obtain node embeddings, uses the embeddings for link prediction and stores the results.

## 2. Task

This project consists of **3 progressive tasks**.

### 2.1. Implementing the Biased Random Walker (40%)

> Related Files: `walker.py`

The first step is to complete the biased random walk algorithm.

1. **Complete the `get_probs_biased()` method of `BiasedRandomWalker` in `walker.py`**.
   - This function computes the transition probability of the biased random walker.
2. **Complete the `walk()` method of `BiasedRandomWalker` in `walker.py`**.
   - This function performs a random walk and returns a trajectory.

Please refer to the comments in the code for more instructions and hints. You might need to use the `Graph` data structure (which we have provided for you). Please refer to `graph.py` for its methods and documentations.

### 2.2. Implementing the Negative Sampling Loss (40%)

> Related Files: `loss.py`

1. **Complete the `forward()` method of `NegativeSamplingLoss` in `loss.py`**.
   - This module computes the negative sampling loss.

ℹ️ **Note.** In practice, even if your implementation faithfully follow the mathematical formulations, the loss sometimes goes to `NaN` or causes your model to degenerate. We leave it up to you to figure out ways to mitigate such issues.

### 2.3. Completing the Training Loop (20%)

> Related files: `node2vec_trainer.py`

We have provided a `Node2VecTrainer` class for handling the training of your Node2Vec model. Follow the instructions and complete the training loop of the trainer class.

1. **Choose and create an optimizer for your model.** Complete the `create_optimizer()` method in `Node2VecTrainer`. You may choose any optimizer and tune its hyper-parameters as needed.
2. **Complete the training loop.** Complete the sections marked with `TODO` in `train_one_epoch()`. Please refer to the comments for further instructions.
3. **(Optional). Experiment with different hyper-parameters to improve the performance of your algorithm.** Try experiment with different values for hyper-parameters such as $p$, $q$, window size, walk length and number of negative samples. This task is **optional but encouraged**, the default parameters should already be able to achieve a decent result if the algorithm is correctly implemented.

We have made most of the hyper-parameters configurable via commandline arguments and provided a driver script `run.sh`. Your submission will be evaluated on our Ubuntu 20.04 grading server with the following command,

```sh
# Your submission will be evaluated as follows (on our Ubuntu 20.04 grading server).
# If you have changed any hyper-parameters, please include them as commandline arguments in run.sh.
bash run.sh
```

If you have changed any hyper-parameters, it is suggested that you should update your `run.sh` and pass your hyper-parameters as command-line arguments. Other methods (e.g., directly modifying the code to change hyper-parameters) are also acceptable (but not encouraged).

Note that this `run.sh` shell script might not be runnable on a Windows platform. It is only used for automated grading.

## 3. Submission & Requirements

### 3.1. Submission

The provided code will automatically produce a `p3_prediction.csv` under `./data`. The prediction csv file has two columns: `id` and `score`, where `id` corresponds to the `id` in `p3_test.csv` and score is the probability that an edge exists, rounded to 4 decimal places.

Submit your results and all source code as a ZIP file, named as `[StudentID]_[ChineseName]_HW5.zip` (e.g., `522030910000_张三_HW5.zip`). The following files are REQUIRED in your submission.

```txt
张三_522030910000_HW5/
├── data/
│   └── p3_prediction.csv  # prediction produced by `main.py`
├── loss.py                # your NegativeSamplingLoss impl.
├── graph.py
├── node2vec_trainer.py    # your main training loop
├── walker.py              # your BiasedRandomWalker impl.
├── main.py                # main driver code
├── run.sh                 # driver bash script
└── *.py                   # other necessary python files (if any)
```

The TAs will run you code by

```sh
bash run.sh                # For verifying your link prediction results
```

**Please ensure:** (1) Your submission is self-contained, i.e., it contains all necessary Python files for your project to run. (2) Your `run.sh` contains the required commandline arguments (if any) for the TAs to reproduce your results.

## 4. Requirements and Notes

### 4.1. Requirements

1. Do NOT use any existing Node2Vec / Word2Vec implementations, or link prediction APIs.
2. Do NOT use Graph Neural Networks (GNNs). The focus of this assignment is the Node2Vec algorithm. Using GNNs would diverge from this goal.
3. Do NOT modify files or functions marked with `XXX: DO NOT MODIFY`. Other than these functions, please feel free to make other modifications if necessary.
4. DO NOT copy others' code, whether from your classmates or previous years' solutions. Plagiarism will result in *zero* scores for *all* involved parties.
5. Please ensure that your submission is runnable and reproducible. We will run your code and verify the results. A penalty will be imposed if there is a significant gap between your reported results and our reproduced ones.
6. You are encouraged (but not required) to follow good programming habits. E.g., use meaningful variable names, avoid extremely long lines, etc.

## 5. Grading

#### 5.1. Biased Random Walker (40%)

We have prepared a few test cases (not released) to test your implementation of `get_probs_biased()`. Your score will be given according to the results of our test cases. Full 40% if your code passes all our test cases, and deducted proportionally if some cases fail.

#### 5.2. Node2Vec Link Prediction (40% + 20%)

We will score your implementation based on the AUC of your algorithm on the full test set.

|                 Metrics                  | Score (for this part) |
| :--------------------------------------: | :-------------------: |
| Code runs without error. AUC above 0.93. |         100%          |
| Code runs without error. AUC above 0.85. |      90% - 100%       |
| Code runs without error. AUC above 0.75. |       80% - 90%       |
| Code runs without error. AUC above 0.65. |       70% - 80%       |
|               Other cases.               |   Manually scored.    |

## References

1. Grover, Aditya, and Jure Leskovec. "node2vec: Scalable feature learning for networks." Proceedings of the 22nd ACM SIGKDD international conference on Knowledge discovery and data mining. 2016.
2. Perozzi, Bryan, Rami Al-Rfou, and Steven Skiena. "Deepwalk: Online learning of social representations." Proceedings of the 20th ACM SIGKDD international conference on Knowledge discovery and data mining. 2014.
3. [PyTorch Geometric | Node2Vec](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.models.Node2Vec.html).
