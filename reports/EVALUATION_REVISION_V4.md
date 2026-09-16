# 평가 구조 개편 v4.4 — 주 평가(같은 틀·새 의미)와 보조 평가(외곽 틀 전이) 분리

작성일 2026-09-17. 새 학습은 하지 않았다. 학습 문항·모델·토크나이저·학습률·학습 상한은 바꾸지 않았다.
기존 v1/v2 탐색 결과와 체크포인트는 그대로 보존하며, 새 기준의 확증 결과로 소급하지 않는다.
코드 해시(이 커밋): `926c1a6729140c46255cb914113f84b31ec363ddbf7864e75eb5b0c177b2501c`.

## 1. 보존 커밋과 원격 통합 상태

- 탐색 v2 스크립트·설정·보고서·장부는 이 지시 이전에 이미 `f75807f`로 커밋·푸시되어 있다. 그 커밋 메시지에는 당시 기본 규칙에 따라 AI 공동 작성자 트레일러가 들어 있다. 이미 원격 `main`에 있어 트레일러를 지우려면 이력 재작성(강제 push)이 필요하므로 그대로 둔다. **이번 커밋부터는 트레일러를 넣지 않는다.**
- 원격의 RNG 복원 수정 `cd037cd`는 fast-forward로 반영했고, 강제 push는 하지 않았다.

## 2. CUDA 회귀 검사

`tests/test_model_runtime.py::test_cuda_mapped_checkpoint_restores_all_random_streams`를 이 서버 `cuda:0`에서 실제로 실행했다: 단독 실행 1 passed(0.32 s), 전체 스위트에서도 건너뜀 없이 통과. 기존 체크포인트 해시와 과거 보고서는 수정하지 않았다(아래 5절의 해시 검증).

## 3. 구현

### 데이터 형식 (`draft-v4.4`)

| 항목 | 내용 |
|---|---|
| 궤도 소속 | v4.3과 동일. 의미 ID·오답 궤도·번역본의 train/dev A/dev B/test 소속 유지 |
| 학습 행 | `train.jsonl`이 v4.3과 **바이트 동일** (`6c885700…`); 필드·순서·줄바꿈(CRLF)까지 같다 |
| 주 평가 | 보류 행의 최상위 `sentence_a/b`: 보류 의미 조합을 학습과 같은 평서문 틀로 렌더링(`template_family="assertion"`) |
| 보조 평가 | 같은 행의 `auxiliary`: 같은 장면·정답을 분할별 보류 외곽 틀로 렌더링(dev A `truth_question`, dev B `reported_clause`, test `conditional`) — v4.3의 문장과 동일 |
| 같은 문항의 두 표현 | 한 행 = 한 문항(같은 `id`·`meaning_id`·`scene`·`label`)에 두 렌더링. manifest `evaluation.auxiliary.same_items_as_primary=true` |
| test | `independent_test.confirmatory=false` (사유: v4.3 test 궤도는 탐색 런에서 이미 평가됨) |
| 파일 해시 | dataset_hash `488d8f22…`; dev_a `86bfa8dd…`, dev_b `bb0d409c…`, test `e590cd9e…` |

파생 명령: `python -m flystudy revise-evaluation --source data/draft-v4.3 --output data/draft-v4.4`. 기존 데이터는 덮어쓰지 않았다.

### 코드 변경 (테스트 선행, 신규 11개·전체 80개 통과)

- `data.py`: `example_record`(주+보조 렌더링), `dataset_manifest`, `revise_evaluation`, `require_confirmatory_test`; `generate()`도 같은 형식을 낸다; `audit()` 확장(4절).
- `runtime.py`: `Corpus.split(split, view="primary"|"auxiliary")`, `Corpus.has_auxiliary`, `evaluate(..., view=...)`. 기본은 주 평가이므로 정기 패널·숙달 판정·독립 시험은 자동으로 주 표현을 쓴다.
- `train.py`: 정기 시점마다 보조 표현도 채점해 `kind="auxiliary"` 이벤트로만 기록(커리큘럼·숙달 판정 불변, `clocks.auxiliary` 별도 집계); 종료 시 `independent-test.json`에 `auxiliary_scores` 추가; smoke 외 코호트는 `require_confirmatory_test`로 비확증 test 데이터 실행을 거부.
- `cli.py`: `revise-evaluation` 명령.
- 문서: `docs/PROTOCOL_V4.md`(데이터·평가 정의), `README.md`.

### 토크나이저

`train.jsonl`이 같으므로 v4.4로 다시 학습한 토크나이저는 v4.3과 **바이트 동일**(`d30ef71c…`). `artifacts/tokenizer-v4.4.meta.json`은 v4.4 dataset_hash를 가리키며, 학습 코드의 출처 검사가 v4.4에 대해 통과하도록 한다. 토크나이저 자체는 바뀌지 않았다.

### 생성기 일치

같은 시드(1729)로 `generate()`를 다시 돌리면 dev_a·dev_b·test는 파생본과 바이트 동일하고, train은 줄바꿈만 다르다(파생본은 v4.3의 CRLF 원본을 그대로 복사, 생성기는 LF). CRLF→LF 정규화 후 완전히 같다. 따라서 v4.4는 "v4.3 파생"과 "생성기 출력"이 같은 내용이다.

## 4. 누출 검사 (`audit`)

| 검사 | 결과 (v4.4) |
|---|---|
| 의미 궤도·오답 궤도의 분할 배타 | 위반 0 |
| 번역 정렬(세 언어 모두 존재) | 위반 0 |
| 외곽 틀 계열의 분할 배타 — **보조 표현** 기준 | 위반 0 (train=assertion, dev A=truth_question, dev B=reported_clause, test=conditional) |
| 주 표현이 학습 틀(assertion)인지 | 36,000행 모두 |
| 주 표현 문장 쌍이 학습 문장 쌍에 존재하는지 | 0 |
| 주·보조 표현이 같은 장면·정답에서 재렌더되는지 | 36,000/36,000 일치 |
| 셀 크기·정답 균형·독일어 성별 할당 | 기존과 동일, 위반 0 |

주 평가가 학습과 평서문 틀을 공유하는 것은 의도된 설계이며 오류로 잡지 않는다. 변조 검출은 테스트로 확인했다(보조 문장 바꿔치기 → `auxiliary/primary mismatch`; 주 문장 쌍을 학습에 추가 → `primary sentence leakage`).

새 주 평가에 대한 기준선(dev A 평서문): nuisance 12셀 모두 0.500; hypothesis-only 역할 0.544·나머지 0.500, grammar cue 역할 1.000 — v4.3과 같은 값, 감사 요구 없음.

## 5. 기존 체크포인트 재현 검증 (`scripts/verify_evaluation_revision.py`, `reports/evaluation-revision-check.json`)

- CPU: dev A/B/test 각 12,000행에서 id·의미 ID·정답·장면·성별쌍이 v4.3과 같고, 주 표현은 v1 진단의 `assertion_view` 문장과, 보조 표현은 v4.3 원문과 전부 일치. 언어별 문항 수·토큰 수·평균 길이·학습 미노출 토큰 비율이 `reports/token-exposure-v1.json`과 정확히 일치(주 표현 0%, 보조 EN 71.4/71.6%, DE 76.8/75.4%, KO 68.1/70.7%).
- GPU: v1 `primary.pt` 5개 + v2 단계 체크포인트 12개 = 17개를 v4.4 주/보조 경로(`runtime.evaluate`)로 dev A/B 채점 → 기록된 평서문/원본 정답 수와 **모든 셀(총 368셀)에서 정확히 일치**. 체크포인트 해시는 기록과 같고, `immutable_evaluation` 통과에 더해 평가 전후 파라미터 텐서 동일을 별도로 확인.
- 비용 0.0125 GPU시간(`eval-revision-check-1`).

## 6. test 취급

v4.3의 test 궤도는 탐색 런의 독립 시험에 이미 쓰였다. v4.4는 이를 `confirmatory=false`로 표시하고, `train.run`은 smoke 외 코호트에서 이 데이터를 거부한다(테스트로 확인: pilot 코호트 거부, smoke 코호트 허용). 새 확증용 시험셋은 본실험 진입 전에 새 데이터 버전으로 만든다. 기존 v1/v2 결과는 탐색 결과로만 남는다.

## 7. 남은 문제: EN 공간, DE 역할·공간

v2 900k 단일 언어 체크포인트를 v4.4 주 평가(dev A/B 평서문)로 문항 단위 채점(`scripts/error_breakdown.py`, `reports/remaining-issues-v4.4.json`, 0.0009 GPU시간):

| 셀 | dev A / dev B | 하위군 범위 | 정답별 |
|---|---:|---|---|
| EN 공간 | 0.590 / 0.561 | 축·방향 6조합 0.49–0.71, 특정 축에 몰리지 않음 | label 0: 0.51–0.52, label 1: 0.62–0.66 |
| DE 공간 | 0.580 / 0.575 | 좌우 0.63–0.65 > 상하·전후 0.52–0.56 | label 0: 0.53–0.54, label 1: 0.62 |
| DE 역할 | 0.715 / 0.681 | 동사쌍 3종 0.67–0.75, 동사 순서 6종 0.63–0.77 | label 0/1 차이 ≤0.03 |
| (대조) EN 역할 | 0.948 / 0.960 | 0.89–0.99 | – |
| (대조) KO 공간 | 1.000 / 0.999 | 전부 ≥0.994 | – |

같은 체크포인트의 학습 표본 정확도는 EN 공간 0.93, DE 공간 0.79, DE 역할 0.81이므로 세 셀 모두 "학습 문항은 맞히지만 같은 틀의 새 의미 조합으로 옮겨가지 못하는" 상태다. 하위군이 고르게 낮아 특정 축·방향·동사쌍의 템플릿 결함으로 보이지는 않는다.

토큰 구조(`dev_a` 주 표현, 참/거짓 `sentence_b`의 차이 토큰 분석):

| 셀 | 결정 토큰 종류 수 | 결정 토큰에 명사가 융합된 쌍 | 결정 토큰 학습 빈도 최소/중앙값 |
|---|---:|---:|---:|
| EN 공간 | 74 | 100% | 58 / 278 |
| EN 역할 | 68 | 100% | 58 / 395 |
| DE 공간 | 144 | 100% | 65 / 134 |
| DE 역할 | 72 | 100% | 236 / 278 |
| KO 공간 | 30 | 0% | 374 / 426 |
| KO 역할 | 154 | 37% | 56 / 211 |

whole-string BPE는 EN/DE에서 방향어·동사를 인접 명사와 한 토큰으로 묶는다(예 `foxĠisĠaboveĠtheĠ` vs `foxĠisĠbelowĠtheĠ`). KO 공간만 명사 융합이 없고 결정 토큰이 30종으로 적으며 빈도가 높다 — 가장 빨리·완전히 학습된 셀과 일치한다. 그러나 EN 역할도 명사 융합 100%이면서 0.95로 일반화하므로 융합만으로 EN 공간의 실패를 설명하지 못한다. 남는 차이는 과제 구조다: 공간 과제의 참 답은 두 독립 개체쌍 각각에서 방향어를 **반의어로 뒤집는** 대응이고, 역할 과제는 같은 두 개체에 두 동사를 **재결합**하는 대응이다. 어느 쪽이 원인인지는 이 자료로 확정할 수 없다.

후속 점검 후보(이번에 실행하지 않음): (a) 단어 경계를 유지하는 토크나이저 조건은 토크나이저 변경이므로 별도 비교 조건으로만 다룬다; (b) 공간 과제를 한 절(개체 2개)로 줄인 진단 셀; (c) DE 역할의 수동태·여격 표현이 문항 간 얼마나 다양한지 점검. KO 표현 수정(내포절 주제 조사)은 별도 변경으로 남기며 이번 평가 구조 변경과 섞지 않았다.

## 비용과 장부

| 예약 ID | 용도 | 예약 | 실사용 |
|---|---|---:|---:|
| `eval-revision-check-1` | 17개 체크포인트 재현 검증 | 0.10 h | 0.0125 h |
| `eval-revision-errors-1` | 오류 분해 | 0.05 h | 0.0009 h |

누적 사용 0.678 / 672 GPU시간. 열린 예약 없음. 새 학습 없음.

## 다음 단계로 남기는 것

- 새 확증용 시험셋(새 데이터 버전)과 v4.4 두 표현에 대한 검수 인증. 검수 패키징 스크립트(`scripts/build_review_packages.py`, `ai_review.py`)는 아직 v4.3의 "분할=외곽 틀" 가정을 쓴다.
- 이번 커밋으로 코드 해시가 바뀌므로 다음 학습 전에 G0/G1 재검증을 묶어 수행한다.
- `reports/implementation-status.json`은 v4.3 패키징 산출물이며 재생성하지 않았다.
