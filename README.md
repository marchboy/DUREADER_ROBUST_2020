## 1.训练 && 评估
1. 需要先下载模型，以及 train_bert.sh 中相应的路径修改
2. 运行以下命令：
```shell script
bash train_bert.sh
```

## 2.更换模型
比如更换百度的 ERNIE 模型，可参考：https://github.com/nghuyong/ERNIE-Pytorch。下载对应的模型，
更改模型加载部分，其他部分无需更改即可使用 ERNIE 进行微调