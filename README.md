# Efficient-Paddle

项目目前能够正常训练和评估。采用paddle2.1.2+PaddleDetection开发。
但是由于训练所需的计算资源问题，原论文需要300个epochs以及128 batchsize的配置。
单卡16G配置下的config需要的训练时间非常长，需要大半个月，我并没有进行成功的完整复现，只进行了几个epoch设置的训练（train_0.log有训练的日志）。
**实验结果表示能够稳定训练，且loss下降指标缓慢上升，这和其他训练框架下的训练日志类似，前期指标上升非常缓慢**
希望有计算资源的朋友可以试用验证复现的正确性吧～


```python
## unzip datasets
数据集放在dataset/coco下
! unzip -q /home/aistudio/data/data7122/annotations_trainval2017.zip -d dataset/coco
! unzip -q /home/aistudio/data/data7122/val2017.zip -d dataset/coco
! unzip /home/aistudio/data/data7122/train2017.zip -d dataset/coco
! unzip /home/aistudio/data/data7122/test2017.zip -d dataset/coco
```

```python
## Install third-party library
!pip install pycocotools
!pip install -r requirements.txt
```


## train
```python
!python tools/train.py -c configs/efficientdet/efficientdet_d0_1x_coco.yml --eval --use_vdl=True
```

## AIStudio项目

[AIStudio](https://aistudio.baidu.com/aistudio/projectdetail/2405619?contributionType=1)
