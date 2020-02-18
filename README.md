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
- [ ] **most hard samples**. 使用 `AL ann_ratio` 作为较困难图像的衡量方式，在 update training dataset 是优先选择最 hard 的样本 
- [ ] **update eval dataset**. 随着模型新数据引入，原始的 eval data 已不能反应模型在最新数据上的性能，需要类似 Continual Learning 的方式更新 eval data

### Harder Question
如何在更少的 initial data 上 train model，进一步得到标注成本更低的 asm

## Code Structure

- **`datasets`**.
  - `configs.py`. voc data class names and statistics.
  - `voc_parser.py`. parse voc data from original xml data, save in pickle. `list[dict, ...]`
  - `VOC.py`. torch Dataset class.
- **`net`**. 
  - `faster_rcnn.py`. faster rcnn using torchvision implementation 
- **`tools`**. useful functions in model training and evaluating
- **`train_asm_one.py`**. train asm from scratch, it will train a model on initial dataset first, then trian on combination of label_anns and sa_anns. This `*.py` is a combination of `train.py` and `train_asm.py`. 


