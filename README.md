# SharpSeq: Empowering Continual Event Detection through Sharpness-Aware Sequential-task Learning
Source code for the ACL Rolling Review submission SharpSeq.


## Data & Model Preparation

To preprocess the data similar to [Lifelong Event Detection with Knowledge Transfer](https://aclanthology.org/2021.emnlp-main.428/) (Yu et al., EMNLP 2021), run the following commands:
```bash
python prepare_inputs.py
python prepare_stream_instances.py
```

## Training and Testing

To start training and testing on MAVEN, run:
```bash
sh sh/maven_herding_multitask_naloss4_gmm_sam_2loss.sh
```

## Requirements:
- transformer == 4.23.1
- torch == 1.9.1
- torchmeta == 1.8.0
- numpy == 1.21.6
- tqdm == 4.64.1
- scikit-learn
- cvxpy
