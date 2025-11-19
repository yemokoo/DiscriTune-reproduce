# DiscriTune-reproduce
Reproduce main table of "Cross-Domain Image Captioning with Discriminative Finetuning"

## Eval 폴더
COCO-final.json -> 초기 학습 데이터 셋 잘못 구현하여 COCO train + valid 셋으로 학습된 최종 모델 eval 결과  
COCO-kaparthy_N.sjon -> 학습 데이터 셋 논문과 동일하게 kaparthy tarin split으로 변경 + reward 정규화 추가된 N번째 epoch의 모델 eval 결과

다른 벤치마크도 동일.


## My results 
| Model | COCO | ConCap | Flickr30k | NoCaps-Near | NoCaps-Out | Concadia |
|-------|------|-----|-----------|-------------|------------|----------|
| ClipCap-COCO | 74.64 | 74.25 | 65.37 | 77.46 | 75.36 | 56.04 |
| DiscriTune - COCO | 77.68 | 77.60 | 71.62 | 82.35 | 78.71 | 61.52 |
| DiscriTune-COCO + 정규화 | 88.74 | 85.06 | 86.00 | 89.19 | 84.93 | 68.52 |


## Paper results
| Model | COCO | ConCap | Flickr | nocaps near | nocaps out | Concadia |
|-------|------|--------|--------|-------------|------------|----------|
| ClipCap-COCO | 74.2 | 73.0 | 65.9 | 77.3 | 73.9 | 53.74 |
| DiscriTune-COCO | 84.8 | 83.6 | 79.4 | 86.0 | 82.5 | 64.79 |
