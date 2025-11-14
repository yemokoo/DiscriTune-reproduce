# DiscriTune-reproduce
Reproduce main table of "Cross-Domain Image Captioning with Discriminative Finetuning"

## Eval 폴더
COCO-final.json -> 초기 학습 데이터 셋 잘못 구현하여 COCO train + valid 셋으로 학습된 최종 모델 eval 결과
COCO-kaparthy_N.sjon -> 학습 데이터 셋 논문과 동일하게 kaparthy tarin split으로 변경 + reward 정규화 추가된 N번째 epoch의 모델 eval 결과

다른 벤치마크도 동일.
