# 环境配置
https://mmdetection.readthedocs.io/zh_CN/latest/get_started.html



# 方案
  
## 1.数据
### 1.1训练数据处理
  
训练数据1：在13000张照片中随机划分出10000张作为训练集，另外3000张设定为验证集
  
训练数据2：从训练数据1中选取的10000张训练集中选出已标注和未标注的图片，然后将已标注的图片上的篡改部分随机粘贴到任意10张未标注的图片上，从而获得30000~40000张扩充的图片作为粗训练集，用于强化模型对篡改部分样本的识别能力（切图代码见cut_image.py）
  
训练数据3：将13000张照片全部视为训练集

  
  
## 2.模型
### 2.1模型选择
  
使用mmdetection开源库（ https://github.com/open-mmlab/mmdetection ）中的模型进行训练。
采用将多种类型模型推理结果相融合的方式，互补得出较单个模型更为有效的推理结果。
采用的模型配置都是基于DINO框架的设计,具体配置如下（包括训练，测试参数等）：
A.backbone为ResNet的小模型dino-4scale_r50_8xb2-36e_coco.py
B.backbone为SwinTransformer的大模型dino-5scale_swin-l_8xb2-36e_coco.py

### 2.2具体训练流程

| 模型      | 数据 |  训练周期  |加载权重| 最终权重 | 备注|
| ----------- | ----------  | -----------| ----------- | ----------  |  ----------  | 
| ResNet-Base | 训练数据1    | 34 |dino-4scale_r50_8xb2-12e_coco_20221202_182705-55b2bba2.pth |*ResNet_epoch_34.pth|在train_config中设置maxepoch为50，在第34个epoch后停止训练|
| Swin-Base   | 训练数据1    | 47 |dino-5scale_swin-l_8xb2-36e_coco-5486e051.pth |*Swin_10000_epoch_47.pth    |在train_config中设置maxepoch为50，在第47个epoch后停止训练|
| Swin-Base   | 训练数据2    | 4  |dino-5scale_swin-l_8xb2-36e_coco-5486e051.pth |rough_epoch_4.pth  |在训练数据2上进行4个epoch的粗训练
| Swin-Base   | 训练数据1    | 36+9 |rough_epoch_4.pth|*afterrough_epoch36+epoch9.pth | 加载粗训练的模型，再在训练数据1上训练36+9个epoch（36为默认配置，9为保持学习率后的再9个epoch的训练）    | 
| Swin-Base   | 训练数据3    | 36 |dino-5scale_swin-l_8xb2-36e_coco-5486e051.pth |*Swin_13000_epoch_36.pth    |默认配置，在第36个epoch后终止|

  
  
  
## 3.结果推理和融合
### 3.1TTA
为了得到更准确的预测，增加融合使用测试时增强（TTA）的推理结果
本方案中的TTA为多尺度放缩和随机翻转；由于需要结果融合，则考虑增加不同多尺度放缩结果来提高准确度，以下A和B为选用的TTA尺度：

```python
A.
img_scales = [(480,1333), (640, 1333), (800, 1333),(1333,800)]
#之后称TTA.A
B.
img_scales=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
           (608, 1333), (640, 1333), (672, 1333), (704, 1333),
           (736, 1333), (768, 1333), (800, 1333), (1333,800)]
#之后称TTA.B
```
```python
tta_model = dict(
    type='DetTTAModel',
    tta_cfg=dict(nms=dict(
                   type='nms',
                   iou_threshold=0.5),
                   max_per_img=100))

tta_pipeline = [
    dict(type='LoadImageFromFile',
         backend_args=None),
    dict(
        type='TestTimeAug',
        transforms=[
            [
                dict(
                    type='Resize',
                    scale=s,
                    keep_ratio=True) for s in img_scales],
            [dict(type='RandomFlip', prob=0.5)],
            [dict(
               type='PackDetInputs',
               meta_keys=('img_id', 'img_path', 'ori_shape',
                       'img_shape', 'scale_factor', 'flip',
                       'flip_direction'))]])]
```


### 3.2结果融合
单个权重推理的结果置信度不高，使用多个模型的无TTA推理(test)和部分使用TTA的推理结果相融合以得到最终的提交结果
#### 3.2.1待融合推理结果
|权重|推理方式|推理结果|
|----|----|----|
|ResNet_epoch_34.pth|test|ResNet_epoch_34_test.bbox.json|
|Swin_10000_epoch_47.pth|TTA.A|Swin_10000_epoch_47_TTAA.bbox.json|
|Swin_10000_epoch_47.pth|TTA.B|Swin_10000_epoch_47_TTAB.bbox.json|
|Swin_10000_epoch_47.pth|test|Swin_10000_epoch_47.bbox_test.json|
|Swin_13000_epoch_36.pth|TTA.A|Swin_13000_epoch_36_TTAA.bbox.json|
|Swin_13000_epoch_36.pth|TTA.B|Swin_13000_epoch_36_TTAB.bbox.json|
|Swin_13000_epoch_36.pth|test|Swin_13000_epoch_36_test.bbox.json|
|afterrough_epoch36+epoch9.pth|test|afterrough_epoch36+epoch9_test.bbox.json|

#### 3.2.2融合阈值
通过计算推理结果在验证集上的最佳f1，确定每个推理结果的最佳分数阈值，计算最佳f1的代码如下所示：
```python
####################################################################
# self._coco_api & coco_dt
            max_score=-1
            max_i=-1
            for i in range(20,90):
                coco_gt = self._coco_api
                tp = fp = fn = 0
                # 遍历所有图像
                for img_id in self.img_ids:
                    # 获取图像对应的gt_anns
                    gt_ann_ids = coco_gt.get_ann_ids(img_ids = [img_id])
                    gt_anns = coco_gt.load_anns(ids = gt_ann_ids)
                    # 获取图像对应的dt_anns,并按分数排序
                    dt_ann_ids = coco_dt.getAnnIds(imgIds = [img_id])
                    dt_anns = coco_dt.loadAnns(ids = dt_ann_ids)
                    dt_anns.sort(key=lambda ann:ann['score'],reverse=True)       
                    # 匹配
                    for ann1 in dt_anns:
                        if(ann1['score']>(i/100)):
                            mch = False
                            mch_ann = None
                            max_iou = 0
                            for ann2 in gt_anns:
                                # 计算交集面积
                                x_inter1 = max(ann1['bbox'][0],ann2['bbox'][0])
                                y_inter1 = max(ann1['bbox'][1],ann2['bbox'][1])
                                x_inter2 = min(ann1['bbox'][0]+ann1['bbox'][2],ann2['bbox'][0]+ann2['bbox'][2])
                                y_inter2 = min(ann1['bbox'][1]+ann1['bbox'][3],ann2['bbox'][1]+ann2['bbox'][3])
                                area_inter = (x_inter2 - x_inter1) * (y_inter2 - y_inter1)
                                # 计算并集面积
                                area_union = (ann1['bbox'][2] * ann1['bbox'][3]) + (ann2['bbox'][2] * ann2['bbox'][3]) - area_inter
                                # 计算iou
                                iou = area_inter / area_union
                                # 比较
                                if iou > self.iou_thrs[0]:
                                    if (not mch) or max_iou < iou:
                                        mch = True
                                        mch_ann = ann2
                                        max_iou = iou
                            if mch:
                                tp += 1
                                gt_anns.remove(mch_ann)
                            else:
                                fp += 1
                    fn += len(gt_anns)
                计算ap,ar,f1
                micro_p = tp / (tp + fp)
                micro_r = tp / (tp + fn)
                f1 = (2 * micro_p * micro_r) / (micro_p + micro_r)
                if(f1>max_score):
                   max_score=f1
                   max_i=i
                logger.info(f'iou_thr = {self.iou_thrs[0]}, micro_f1 = {f1} at threshold={i/100}')
            logger.info(f'max_score={max_score} at threshold={max_i/100}')

####################################################################
```
由于部分结果本身置信度不高，则选择融合的阈值会相对其最佳阈值提高，各结果的选择阈值如下：

|推理结果|选择阈值|
|----|----|
|ResNet_epoch_34_test.bbox.json|0.97|
|Swin_10000_epoch_47_TTAA.bbox.json|0.47|
|Swin_10000_epoch_47_TTAB.bbox.json|0.64|
|Swin_10000_epoch_47_test.bbox.json|0.23|
|Swin_13000_epoch_36_TTAA.bbox.json|0.50|
|Swin_13000_epoch_36_TTAB.bbox.json|0.69|
|Swin_13000_epoch_36_test.bbox.json|0.29|
|afterrough_epoch36+epoch9_test.bbox.json|0.39|

#### 3.2.3融合
按照阈值过滤掉部分置信度较低的结果,过滤代码见output_to_be_fusion_ans.py

然后对每张图像的推测结果,按预测框分数降序排序,对iou大于0.5的预测框进行加权融合,得到最终的预测结果,融合代码见fusion_.py




# 参考文献
1.Kai Chen, Jiaqi Wang, Jiangmiao Pang, Yuhang Cao, Yu Xiong, Xiaoxiao Li, Shuyang Sun, Wansen Feng, Ziwei Liu, Jiarui Xu, Zheng Zhang, Dazhi Cheng, Chenchen Zhu, Tianheng Cheng, Qijie Zhao, Buyu Li, Xin Lu, Rui Zhu, Yue Wu, Jifeng Dai, Jingdong Wang, Jianping Shi, Wanli Ouyang, Chen Change Loy, Dahua Lin.MMDetection: Open MMLab Detection Toolbox and Benchmark.ArXiv1906.07155;Available from:https://arxiv.org/abs/1906.07155

2.Hao Zhang, Feng Li, Shilong Liu, Lei Zhang, Hang Su, Jun Zhu, Lionel M. Ni, Heung-Yeung Shum.Hao Zhang, Feng Li, Shilong Liu, Lei Zhang, Hang Su, Jun Zhu, Lionel M. Ni, Heung-Yeung Shum.ArXiv2203.03605;Available from:https://arxiv.org/abs/2203.03605
