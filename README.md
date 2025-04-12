
# ConQuatE
The source code of article **Contextualized Quaternion Embedding Towards Polysemy in Knowledge Graph for Link Prediction（TALLIP 2024）**
## Dependencies
Python 3.6+
PyTorch 1.0+
NumPy 1.17.2+

## Reproduce the Results

To reproduce the results of RQE and HRQE on WN18RR, FB15k237, WN18 and FB15K, please run the following commands.

### WN18RR

```text
python train_models.py  --model ConQuatE --dataset WN18RR --train_times 50000 --nbatches 10 --alpha 0.1 --dimension 300 --lmbda 0.3 --lmbda_two 0.01 --ent_neg_rate 1 --valid_step 2000
```

### FB15K237

```text
python train_models.py  --model ConQuatE --dataset FB15K237 --train_times 5000 --nbatches 100 --alpha 0.05 --dimension 500 --lmbda 0.5 --lmbda_two 0.01 --ent_neg_rate 10 --valid_step 400
```

## Citation


This code is based on the OpenKE project.
To cite this work:

```text
@article{chenContextualizedQuaternionEmbedding2025,
  title = {Contextualized {{Quaternion Embedding Towards Polysemy}} in {{Knowledge Graph}} for {{Link Prediction}}},
  author = {Chen, Jie and Wang, Yinlong and Zhao, Shu and Zhou, Peng and Zhang, Yanping},
  date = {2025-02-08},
  journaltitle = {ACM Trans. Asian Low-Resour. Lang. Inf. Process.},
  issn = {2375-4699},
  doi = {10.1145/3714411},
}
```

