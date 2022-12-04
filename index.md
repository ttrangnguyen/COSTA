## A Context-based Vulnerable Statement Localization Approach

<p align="justify">
The number of attacks exploring software vulnerabilities has dramatically increased, which has caused various severe damages. Thus, early and accurately detecting vulnerabilities becomes essential to guarantee software quality and prevent the systems from malicious attacks. Multiple automated vulnerability detection approaches have been proposed and obtained promising results. However, most studies detect vulnerabilities at a coarse-grained, i.e., file or method level. Thus, developers still have to spend significant investigation efforts on localizing vulnerable statements. In this paper, we introduce COSTA, a novel context-based approach to effectively and efficiently localize vulnerable statements. In particular, given a vulnerable function, COSTA identifies its vulnerable statements based on the suspiciousness scores of the statements in that. Specifically, the suspiciousness of each statement is measured according to its semantics captured by four contexts, including operation context, dependence context, surrounding context, and vulnerability type. Our experimental results on a large real-world dataset show that COSTA outperforms the state-of-the-art approaches up to 96% in F1-Score and 167% in Accuracy. In addition, COSTA also improves these approaches up to two times in Top-1 Accuracy. Interestingly, COSTA obtains about 80% at Top-3 Recall. This result indicates that developers can find about 80% of the vulnerable statements by investigating three first-ranked statements in each function.
</p>


## About dataset
<table>
  <thead>
  <tr>
    <th></th>
    <th>#Vulnerable statements</th>
    <th>#Non-vulnerable statements</th>
  </tr>
</thead>
 <tbody>
   <tr>
   <td>Training set</td>
   <td align="right">12,383</td>
   <td align="right">67,619</td>
   </tr>
   
   <tr>
   <td>Validating set</td>
   <td align="right">1,660</td>
   <td align="right">8,010</td>
   </tr>
   
   <tr>
   <td>Testing set</td>
   <td align="right">1,414</td>
   <td align="right">8,215</td>
   </tr>
  
   <tr>
   <td>Total</td>
   <td align="right">15,475</td>
   <td align="right">83,844</td>
   </tr>
</tbody>
</table>
  
  


To download the testing dataset used for evaluation in our experiments, run the following commands:

```
gdown https://drive.google.com/uc?id=1ZGIdzKdlzyjX7wSJbP0AfMf5BFovRv1g
```

To download the training and validation dataset used for evaluation in our experiments, run the following commands:

```
gdown https://drive.google.com/uc?id=1dvvZeynTCNdLSBdX7H3wEnRKIZWyILlv
gdow https://drive.google.com/uc?id=11pyuNbkop_5uk10uAoNr4__Tpww65HXb
```

For more information of our dataset, please refer to <a href="https://ieeexplore.ieee.org/document/9796256">LineVul</a> and <a href="https://dl.acm.org/doi/abs/10.1145/3379597.3387501"> Big-Vul.

## Train and test the vulnerability localization models

We provide python source code for training and testing the vulnerability localization models. The source files can be found <a href="https://github.com/ttrangnguyen/COSTA/tree/main/Models">here</a>.
We recommend to use Google Colab to execute the Jupiter notebook COSTA.ipynb.

Please modify hyper-parameters such as batch_size, epoch, vector_length, etc. to fit your own experiments.

## Anlyze source code to obtain contexts
We use <a href="https://github.com/joernio/joern">Joern</a> to analyze source code. The python script for reading CPG nodes and edges can be found <a href="https://github.com/ttrangnguyen/COSTA/blob/main/Joern/joern_script.py">here</a>. 
  



