# ASM_core

Active Sample Learning on VOC detection dataset.

## Features

### model
- [x] **early stop**. 设置 `ap_range`, `ap_shift`，如果 `ap_range` 范围内 ap 平均变化 < `ap_shift` 停下
- [x] **tensorboard**. 观察 `AP_50`, `AP_75`, `AP_shift`, `SL/AL ratio`, `loss`, `lr` 等参数
- [x] **resume training**. 在 `save_model()` 中保留 `optimizer`, `epoch` 保持 `lr_schedule` 同步
- [ ] 尝试不同的 `optimizer`, `lr_schedule`

### ASM
- [x] **img-level -> box-level**. 精确到筛查每张图像上的 box，引入 `certian`/`uncertain` boxes 比较机制
- [x] **batch import unlabel data**. 设置 `K`，每个 epoch，`detect_unlabel_imgs()` 从 unlabel data 中取 `K` 个使用当前 model 进行检测，更新 gt_anns 得到 sa_anns，模拟完成一次 human-machine cooperated anns，更新算法见 `utils/asm_utils.py` 
- [x] **SL/AL ann_ratio/img**. 显示每张图像 SL/AL anns 标注数量平均占比，反应模型 SL 性能变化
- [x] **most hard samples**. 使用 `AL ann_ratio` 作为较困难图像的衡量方式，在 update training dataset 是优先选择最 hard 的样本 
- [ ] **update eval dataset**. 随着模型新数据引入，原始 eval data 已不能反应模型在最新数据上的性能，需要类似 Continual Learning 的方式更新 eval data

### Harder Question
如何在更少的 initial data 上 train model，进一步得到标注成本更低的 asm


## import informative samples

### v1

Don't change the label data, and incrementally import `K` SA_anns unlabel data

1. select `K` samples from unlabel data
2. detect on `K` unlabel samples and generate `K` SA_anns
3. randomly select `K` samples from label data
4. train on the new dataset: `K` SA_anns + `K` label anns
5. goto 1, until `ap_shift < ap_shift_thre`


### v1+

Expand the label data with SA_anns every batch so that we can **choose more just SA annotated unlabel data in next batch**. 

- **expanding label data:** `label_anns += SA_anns`
- update the step 3 of **v1**


### topK v1

排序 `AL ann_ratio`，K largest 加入 uncertain, K smallest 加入 certain 并增加给 label anns

`AL ann_ratio` is a good indicator of the informative images.
- low. more objects are annotated by the model. 
- high. more objects are annotated by the human. (**what we need most**) 

`sa_anns`,`sa_ratios`: store the `AL ann_ratio` and `SA_anns` of the already detected unlabel data in previous batches.

After the detection of current batch

```py
# 当前所有已经检测的 unlabel data 的 sa_anns, AL ann_ratio
sa_anns += batch_sa_anns
sa_ratios += batch_al_ratio

# 排序 AL ann_ratio
cer_idxs = np.argsort(sa_ratios)  # 默认从小到大，从 certain -> uncertain

# 选择 top K certain/uncertain

topK_cer_anns = [sa_anns[i] for i in cer_idxs[:args.K]]  # top K certain
topK_uncer_anns = [sa_anns[i] for i in cer_idxs[-args.K:]]  # top K uncertain

# 随机选 K 个 label anns
random_label_anns = random.sample(label_anns, args.K)

# 形成新的 trianset
asm_train_anns = topK_uncer_anns + random_label_anns

# 使用 top K certian anns 补充 label_anns，下轮再从中随机选择
label_anns += topK_cer_anns
```

### topK v2

With the model improved, `sa_anns` and `sa_ratios` of the previous batches should be updated. 

Judge certain/uncertain anns by AL ratio threshold.
- `AL ann_ratio <= 0.3`，加入 certain 增加给 label anns
- `AL ann_ratio >= 0.6`，加入 uncertain 并保存 pre_gt_uncer_anns 作为下轮引入 batch_unlabel_anns，实现旧的 uncertain 样本也能交给新模型检测并用于下个 batch 训练

## Code Structure

- **`datasets`**.
  - `configs.py`. voc data class names and statistics.
  - `voc_parser.py`. parse voc data from original xml data, save in pickle. `list[dict, ...]`
  - `VOC.py`. torch Dataset class.
- **`net`**. 
  - `faster_rcnn.py`. faster rcnn using torchvision implementation 
- **`tools`**. useful functions in model training and evaluating
- **`train_asm_one.py`**. train asm from scratch, it will train a model on initial dataset first, then trian on combination of label_anns and sa_anns. This `*.py` is a combination of `train.py` and `train_asm.py`. 


