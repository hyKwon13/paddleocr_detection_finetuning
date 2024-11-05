# 프로젝트 개요
한국어로 작성된 PaddleOCR의 Detection 모델에 대한 세부 문서가 부족하여, 본 프로젝트를 추진하게 되었습니다. 본 프로젝트의 주요 목적은 사용자 맞춤형 전이 학습을 통해 Text Detection 및 Text Recognition 모델을 최적화하여, 특정 목적에 부합하는 활용성을 높이는 데 있습니다. <br/>Fine-tuning 과정에서 발생하는 오류나 질문이 있으시면, 언제든지 문의해 주시기 바랍니다. 상세히 답변해 드리겠습니다. 

## 목차

- [프로젝트 개요](#프로젝트-개요)
- [설치 지침](#설치-지침)
- [데이터 주석](#데이터-주석)
- [구성 파일 예시](#구성-파일-예시)
- [모델 학습](#모델-학습)
- [모델 내보내기](#모델-내보내기)

## 설치 지침

### 환경 요구 사항
- PaddlePaddle >= 2.1.0
- Python 3.5 <= Python < 3.9
- PaddleOCR >= 2.1

### 설치 명령어
```sh
# 프로젝트 클론
!git clone https://gitee.com/paddlepaddle/PaddleOCR.git

# PaddleOCR 설치
!pip install fasttext==0.8.3
!pip install paddleocr --no-deps -r requirements.txt

# 디렉토리 이동
%cd PaddleOCR/
```

## 데이터 주석
paddleocr 설치 후 cmd에서 ppocrlabel을 입력하여 전용 프로그램으로 라벨링을 진행합니다.

## 구성 파일 예시
이 프로젝트는 configs/det/ch_PP-OCRv2/ch_PP-OCRv2_det_student.yml구성 파일을 사용합니다. 파일의 변경된 지점은 다음과 같습니다.

```yaml
Global:
  debug: false
  use_gpu: true
  epoch_num: 1200
  log_smooth_window: 20
  print_batch_step: 10
  save_model_dir: ./output/det_dianbiao_v3
  save_epoch_step: 1200
  eval_batch_step:
  - 0
  - 100
  cal_metric_during_train: false
  pretrained_model: my_exps/student.pdparams
  checkpoints: null
  save_inference_dir: null
  use_visualdl: false
  infer_img: M2021/test.jpg
  save_res_path: ./output/det_db/predicts_db.txt
Architecture:
  model_type: det
  algorithm: DB
  Transform: null
  Backbone:
    name: MobileNetV3
    scale: 0.5
    model_name: large
    disable_se: true
  Neck:
    name: DBFPN
    out_channels: 96
  Head:
    name: DBHead
    k: 50
Loss:
  name: DBLoss
  balance_loss: true
  main_loss_type: DiceLoss
  alpha: 5
  beta: 10
  ohem_ratio: 3
Optimizer:
  name: Adam
  beta1: 0.9
  beta2: 0.999
  lr:
    name: Cosine
    learning_rate: 0.0001 
    warmup_epoch: 2 
  regularizer:
    name: L2
    factor: 0
PostProcess:
  name: DBPostProcess
  thresh: 0.3
  box_thresh: 0.6
  max_candidates: 1000
  unclip_ratio: 1.5
Metric:
  name: DetMetric
  main_indicator: hmean
Train:
  dataset:
    name: SimpleDataSet
    data_dir: ./
    label_file_list:
    - M2021/M2021_label_train.txt  
    ratio_list:
    - 1.0
    transforms:
    - DecodeImage:
        img_mode: BGR
        channel_first: false
    - DetLabelEncode: null
    - CopyPaste: null  
    - IaaAugment:
        augmenter_args:
        - type: Fliplr
          args:
            p: 0.5
        - type: Affine
          args:
            rotate:
            - -10
            - 10
        - type: Resize
          args:
            size:
            - 0.5
            - 3
    - EastRandomCropData:
        size:
        - 1600  
        - 1600
        max_tries: 50
        keep_ratio: true
    - MakeBorderMap:
        shrink_ratio: 0.4
        thresh_min: 0.3
        thresh_max: 0.7
    - MakeShrinkMap:
        shrink_ratio: 0.4
        min_text_size: 8
    - NormalizeImage:
        scale: 1./255.
        mean:
        - 0.485
        - 0.456
        - 0.406
        std:
        - 0.229
        - 0.224
        - 0.225
        order: hwc
    - ToCHWImage: null
    - KeepKeys:
        keep_keys:
        - image
        - threshold_map
        - threshold_mask
        - shrink_map
        - shrink_mask
  loader:
    shuffle: true
    drop_last: false
    batch_size_per_card: 4
    num_workers: 4
    use_shared_memory: False 
Eval:
  dataset:
    name: SimpleDataSet
    data_dir: ./
    label_file_list:
    - M2021/M2021_label_eval.txt
    transforms:
    - DecodeImage:
        img_mode: BGR
        channel_first: false
    - DetLabelEncode: null
    - DetResizeForTest:
        limit_side_len: 1280
        limit_type: min
    - NormalizeImage:
        scale: 1./255.
        mean:
        - 0.485
        - 0.456
        - 0.406
        std:
        - 0.229
        - 0.224
        - 0.225
        order: hwc
    - ToCHWImage: null
    - KeepKeys:
        keep_keys:
        - image
        - shape
        - polys
        - ignore_tags
  loader:
    shuffle: false
    drop_last: false
    batch_size_per_card: 1
    num_workers: 2
    use_shared_memory: False
profiler_options: null
```

## 모델 학습
```sh
!python tools/train.py -c configs/det/ch_PP-OCRv2/ch_PP-OCRv2_det_student.yml
```

## 모델 내보내기
```sh
!python tools/export_model.py -c configs/det/ch_PP-OCRv2/ch_PP-OCRv2_det_student.yml -o Global.pretrained_model=./my_exps/det_dianbiao_size1600_copypaste/best_accuracy Global.save_inference_dir=./inference/det_db
```

참고 출처
https://aistudio.baidu.com/projectdetail/3429765
