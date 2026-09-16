# 탐색 실험 v1 — 학습 가능성·비용 (연구 증거 아님)

코드 해시 `6df7331008bb4b20…`, 데이터 `draft-v4.3`, 실제 cb5k 배선, CUDA, 배치 256, 시드 20001, 프로토콜 `protocol-v4-ai.json`. 검수 기록이 불완전한 상태에서 예비비로 수행한 **탐색 실험**이며 인증·pilot-ready·G2/G3를 통과한 것이 아니다. 언어 우열 결론은 내지 않는다.

## 결과 요약

| 런 | 노출 | 종료 | 학습 손실 (처음→끝, 최소) | 학습 문항 정확도 | 보류 패널 최고 | 독립 시험 | 실제 GPU시간 |
|---|---:|---|---|---|---|---|---:|
| explore-v1-mono-en | 200,000 | administrative_cap | 0.700→0.696, min 0.678 | 0.53–0.55 | 0.53 | 0.50–0.52 | 0.017 h |
| explore-v1-mono-de | 200,000 | administrative_cap | 0.698→0.695, min 0.680 | 0.52–0.54 | 0.51 | 0.50–0.50 | 0.020 h |
| explore-v1-mono-ko | 200,000 | administrative_cap | 0.697→0.158, min 0.122 | 0.89–1.00 | 0.52 | 0.50–0.50 | 0.019 h |
| explore-v1-mixed | 900,000 | administrative_cap | 0.699→0.228, min 0.161 | – | 0.56 | 0.48–0.52 | 0.186 h |
| explore-v1-seq-en-de-ko | 900,000 | administrative_cap | 0.700→0.359, min 0.034 | 0.59–1.00 | 0.60 | 0.49–0.51 | 0.108 h |

우연 수준은 0.50, 손실 ln2 = 0.693, 숙달 기준은 셀당 0.80 연속 2회.

## 해석

- EN and DE monolingual runs did not learn within the 200,000-exposure cap: training loss stayed at ln2 (0.696->0.695/0.698->0.695) and accuracy on their own training items was 0.52-0.55.
- KO monolingual fit its training items (loss 0.16; train-item accuracy 0.89-1.00) but held-out panels (other construction families, 2-4x longer sequences) stayed at 0.50-0.52 on dev_a/dev_b and 0.50 on test.
- Sequential en->de->ko (900k) and mixed (900k) started fitting all three languages in-distribution (loss min 0.03/0.17) but held-out accuracy never exceeded 0.60 in any cell; no cell reached the 0.80 mastery threshold, so every run ended at its administrative cap.
- Interpretation: with train sentences of 10-20 tokens and evaluation panels of 38-73 tokens in unseen outer frames, the model shows no cross-frame generalisation at this scale; EN/DE also optimise far more slowly than KO. This is a data/design and optimisation issue to resolve before any pilot or main run.

## 비용

- 5런 실제 GPU시간 0.350 h (예약 2.25 h, 예비비), 진단 평가 0.0062 h. 상한 도달 기준 사전 예상(mono 0.016 h, 다국어 0.101 h)과 일치.
- 장부 누적 사용 0.389 / 672 GPU시간.

## KO 새 AI 평가 (격리 세션, Opus 5)

- 50문항 전부 fluent=no: 내포절 '-다는 것' 안의 주어에 주제 조사 '는'이 쓰여 부자연스럽다는 일관된 지적(예: '큰 빨간 사자는 … 있다는 것' → '사자가').
- 새 의미 판정은 원본 label과 50/50 일치. KO space 구문 4계열(assertion·truth_question·reported_clause·conditional) issue=yes, noun_forms/dat는 issue=no.
- 함의: KO space 문장 틀은 새 데이터 버전에서 수정해야 하며, 현재 버전은 인증 대상이 아니다. 원판정(KO-R1/R2)은 그대로 보존했다.

## 파일

- `runs/exploratory-v1/` (런별 events·summary·체크포인트·독립 시험, 저장소 미포함), `reports/exploratory-v1-report.json`, `reports/figures/exploratory-v1/*.png`, `runs/exploratory-v1/diagnostic-train-accuracy.json`.
