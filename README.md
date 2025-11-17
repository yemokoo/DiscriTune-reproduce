# DiscriTune-reproduce
Reproduce main table of "Cross-Domain Image Captioning with Discriminative Finetuning"

## Eval 폴더
COCO-final.json -> 초기 학습 데이터 셋 잘못 구현하여 COCO train + valid 셋으로 학습된 최종 모델 eval 결과  
COCO-kaparthy_N.sjon -> 학습 데이터 셋 논문과 동일하게 kaparthy tarin split으로 변경 + reward 정규화 추가된 N번째 epoch의 모델 eval 결과

다른 벤치마크도 동일.

## NEWS
Advantage 계산 과정에서 그대로 값을 사용하지 않고, 계산된 advantage 값에 Z 정규화를 적용하였더니
원본 논문에서 제안된 결과보다 더 좋은 성능을 보이는 것을 확인함.

## My results 
| Model | COCO | ConCap | Flickr30k | NoCaps-Near | NoCaps-Out | Concadia |
|-------|------|-----|-----------|-------------|------------|----------|
| ClipCap-COCO | 74.64 | 74.25 | 65.37 | 77.46 | 75.36 | 56.04 |
| DiscriTune-COCO: epoch 5 | 81.76 | 79.19 | 74.50 | 82.19 | 79.64 | 61.37 |
| DiscriTune-COCO: epoch 15 | 87.58 | 84.27 | 81.75 | 87.00 | 83.43 | 67.98 |



## Paper results
| Model | COCO | ConCap | Flickr | nocaps near | nocaps out | Concadia |
|-------|------|--------|--------|-------------|------------|----------|
| ClipCap-COCO | 74.2 | 73.0 | 65.9 | 77.3 | 73.9 | 53.74 |
| DiscriTune-COCO | 84.8 | 83.6 | 79.4 | 86.0 | 82.5 | 64.79 |
