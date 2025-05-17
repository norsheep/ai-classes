# Programming Assignment 3 - PageRank & HITS

> TA Contact: Borui Yang, Hangyu Ye  
> E-mail: ybirua@sjtu.edu.cn, hangyu_ye@outlook.com

## 1. Introduction

In this assignment, we will implement two link analysis algorithms: PageRank and HITS.

### 1.1. Environment

The programming will be done in Python, mainly relying on the `numpy` library.

⚠️ You are **NOT ALLOWED** to use off-the-shelf graph computation packages (including but not limited to `networkx`).

### 1.2. Datasets

You are given two randomly generated directed graphs, `graph-full.txt` and `graph-randnew.txt`, which are stored under `data/` directory.

- `graph-full` contains `n=1000` nodes and `m=8192` edges.
- `graph-randnew` contains `n=768` nodes and `m=10240` edges.

A smaller graph is also included for checking your implementation.

- `graph-small` contains `n=100` nodes and `m=1024` edges.

The txt files contains two columns, separated by `\t`, where the first column refers to the source node, and the second column refers to the destination node. **Node ids start from 1**. The files might contain multiple edges between a pair of nodes, but your implementation should treat them as the same edge. Additionally, the graphs do not contain dead ends.

In this assignment, the graphs are represented by **dense 2D `numpy` adjacency matrices** where `adj_matrix[i, j] == 1` if a directed edge exists from `i` to `j`, and `adj_matrix[i, j] == 0` otherwise. While a more efficient way would be to represent the graphs with sparse matrices (e.g., `scipy.sparse`), we do not consider this for simplicity.

## 2. Tasks

### 2.1. PageRank (50%)

> Related file: `pagerank.py`

Complete the missing code blocks in `pagerank.py` marked with `# TODO` labels, following the steps below,

Given a graph $G = (V, E)$ with $n = |V|$ nodes,

1. Initialize PageRank vector (in the function `_init_uniform()`)
   $$ \mathbf{r} = [ 1/n ]_n. $$
2. Initialize the column stochastic matrix $\mathbf{M}$ (in the function `_build_stochastic_matrix()`)
   $$ \mathbf{M}_{ji} = \begin{cases}
    & \frac{1}{OutDeg(i)} \quad& (i, j) \in E\\
    & 0                   \quad& otherwise
   \end{cases}, $$
   where $OutDeg(i)$ denotes the number of outgoing edges of node $i$.
3. Iteratively update the PageRank vector with power iteration, using the update rule (in the function `page_rank()`)
   $$ \mathbf{r} = \beta \mathbf{M} \mathbf{r} +  \left[ \frac{1 - \beta}{n} \right]_n, $$
   where $\beta$ is the probability of following the links. **In this assignment, we set $\beta=0.8$**.

Complete the function `page_rank()` in `pagerank.py`. The function should return the final PageRank vector.

For each graph (`graph-full.txt`, `graph-randnew.txt`), run PageRank for **40 iterations**, and report the following results

- The top 5 nodes with **the highest** PageRank scores.
- The bottom 5 nodes with **the lowest** PageRank scores.

**Note that you only need to complete `pagerank.py`**. We have provided fully-functioning driver code in `main_pagerank.py`. After completing your implementation, execute

```sh
python main_pagerank.py
```

It will automatically load the data, run your PageRank implementation, and output the results. The results will be stored as a JSON file under the `results/` directory (created once `main_pagerank.py` is executed).

ℹ️ For sanity check, in `graph-small.txt`, the node with the highest rank has id 53 and PageRank score around 0.0357.

### 2.2. HITS (50%)

> Related file: `hits.py`

Complete the missing code blocks in `hits.py` marked with `# TODO` labels, following the steps below,

Given a graph $G = (V, E)$ with $n = |V|$ nodes and its adjacency matrix $\mathbf{A}$,

1. Initialize hub and authority vectors $\mathbf{a} = [1/\sqrt{n}]_n$, $\mathbf{h} = [1/\sqrt{n}]_n$.
2. Iterate until convergence
   1. Update $\mathbf{h} = \mathbf{A} \cdot \mathbf{a}$.
   2. Normalize $\mathbf{h}$.
   3. Update $\mathbf{a} = \mathbf{A}^T \cdot \mathbf{h}$.
   4. Normalize $\mathbf{h}$.

Complete the function `hits()` in `hits.py`. The function should return the final hub and authority vectors.

For each graph (`graph-full.txt`, `graph-randnew.txt`), run HITS for **40 iterations**, and report the following results

- The top 5 authority and hub nodes with the highest authority and hub scores.
- The bottom 5 authority and hub nodes with the lowest authority and hub scores.

Similarly, **you only need to complete `hits.py`**. We have provided the driver code in `main_hits.py`. After completing your implementation, execute

```sh
python main_hits.py
```

It will automatically load the data, run your HITS implementation, and output the results. The results will be stored as a JSON file under the `results/` directory.

ℹ️ For sanity check, in `graph-small.txt`, the node with the highest hub score is node 59, while the node with the highest authority score is node 66.

## 3. Submission

Submit your assignment as a ZIP file, named after `[ChineseName]_[StudentID]_HW3P1.zip` (e.g., `张三_522030910000_HW3P1.zip`). Your submitted zip file should follow the structure specified below

```plaintext
张三_522030910000_HW3P1/
├── results/
│   └── *.json    # your results for `graph-full` and `graph-randnew`
├── pagerank.py   # your PageRank implementation
├── hits.py       # your HITS implementation
└── *.py          # other necessary python files
```

Please DO NOT upload the datasets.

Your submission will be checked against our ground truth answers. We will only check the top-k or bottom-k node ids. You will get full score if the node ids are correctly matched, and the scores will be deducted proportionally if some node ids are incorrect. Note that the PageRank scores will not be checked (so please do not worry about floating point precision issues).

## 4. Requirements and Notes

1. DO NOT use off-the-shelf graph computation libraries.
2. DO NOT modify files or functions marked with `XXX: DO NOT MODIFY`. Other than these functions, please feel free to make other modifications if necessary.
3. DO NOT copy others' code, whether from your classmates or previous years' solutions. Plagiarism will result in *zero* scores for *all* involved parties.
4. Please ensure that your submission is runnable and reproducible. We will run your code and verify the results. A penalty will be imposed if there is a significant gap between your reported results and our reproduced ones.
5. You are encouraged (but not required) to follow good programming habits. E.g., use meaningful variable names, avoid extremely long lines, etc.
