# 自然语言处理 2024秋

## 作业二：实现Word2Vec的连续词袋模型

### 作业要求

基于附件提供的Jupyter Notebook文件，以代码填空的形式，实现Word2Vec的连续词袋模型（CBOW）的相关代码，填空完毕后，需展示代码中相应测试部分的输出结果。本次作业，代码实现14分、实验总结1分，共计15分。
提示：只需填写代码中TODO标记的空缺位置即可，具体的代码说明和评分细则详见提供的Jupyter Notebook文件。

### 环境配置

本次作业需要安装torch包和tqdm包，一般地，如果已经正确安装Python但没有该库，在命令行输入

```
pip install torch tqdm
```

即可完成安装。

### 提交方式

直接提交带有已补全的代码及运行结果的Jupyter Notebook文件，文件中要求对所填代码作必要的说明。提交时请重命名文件。**请务必保留单元格的运行结果！**

### 文件说明

```
├── data
│    ├── debug1.txt                         # 用于debug的小语料1
|    ├── debug2.txt                         # 用于debug的小语料2
│    ├── synonyms.json                      # 用于测试词向量的数据
│    └── treebank.txt                       # 用于训练词向量的语料
├── README.md
└── HW2_{your-id-number}_{your-name}.ipynb  # 提供的Jupyter Notebook文件
```

### 宽限时间说明

本学期所有作业共享7天的宽限时间，即所有作业的迟交时间之和不能超过7天。宽限时间以整天为计算，如果发生迟交，迟交时间与截止时间的差值按天数向上取整即为单次迟交所使用的宽限时间。如：第一次小作业迟交了3天，那么第二次小作业便至多迟交4天；第二次小作业再次迟交4天后，后面的作业便不再享有宽限时间。所有宽限时间用完后再迟交者会惩罚一定分数，具体的惩罚方案最终会由老师与助教讨论后确定并公布。同时，即使宽限时间未用完，最终的提交时间也不能晚于本学期最后的提交期限。