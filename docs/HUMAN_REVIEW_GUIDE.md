# 사람 검수 운영 안내 (draft-v4.3, 블라인드 배포)

이 문서는 **운영자**용이다. 검수자에게는 `review/dist/<코드>.zip`만 배포한다.
검수 판정·식별자 등록은 반드시 사람이 하며, AI나 스크립트가 `judged_label`·`fluent`를 채운 파일은 인정하지 않는다.

## 왜 필요한가

`data/draft-v4.3/`의 96,000문항은 규칙 기반으로 **생성된 초안**이다. 연구계획 v4의 데이터 검증 게이트는
언어별 능숙자 두 명이 (1) 층화 200문항의 정답과 자연스러움, (2) 명사 활용표, (3) 문장 틀 전체를 검수해야
통과한다. 이 게이트를 통과하기 전에는 G2·G3·순서 파일럿·본실험을 시작하지 않는다.

## 배포본이 원본과 다른 이유

원본 `audit-sample.csv`는 정답이 세 경로로 드러난다: `label` 열, `id`의 `…:{label}:{lang}` 형식, 그리고 0/1이 교대하는 행 순서.
`construction-examples.csv`에도 `label` 열이 있고 검수 문항과 겹치는 행이 있다. 따라서 `scripts/build_review_packages.py`가
다음을 적용한 배포본을 만든다.

- 정답 `label`·원본 `id` 제거, 무작위 10자리 검수 ID(`review_id`) 부여, 검수자마다 다른 ID와 다른 행 순서.
  무작위 ID는 정답의 직접 노출을 줄이는 장치다. 같은 언어의 두 파일은 문장 내용으로 200문항 전부 대응되므로 **비교 불가가 아니다.**
  독립성은 절차로 확보한다: 상대 파일·원본 정답을 보지 않고 각자 제출하며, 두 제출이 모두 접수되기 전에는 상의하지 않는다(`INSTRUCTIONS.md` 0절).
- 검수자 식별은 실명 대신 `EN-R1-XXXXX` 형식의 코드. `review/private/reviewer-registry.csv`에는 코드·언어·능숙자 근거·발송/수신일만 기록한다.
  실명·연락처가 필요하면 운영자가 저장소 밖 별도 명부로 관리한다.
- `construction-examples-<lang>.csv`에서 `label` 열 제거, 언어별 필터링.
- 대응표 `review/private/mapping.csv`와 원본은 배포하지 않는다. `review/private/`는 700 권한이다.

## 폴더 구조

| 경로 | 배포 여부 | 내용 |
|---|---|---|
| `review/dist/<코드>/`, `<코드>.zip` | 검수자에게 배포 | `items-<코드>.csv`, `noun-forms-<lang>.csv`, `construction-examples-<lang>.csv`, `template-inventory.json`, `template-checklist-<코드>.csv`, `INSTRUCTIONS.md`(한/영) |
| `review/private/` | **절대 배포 금지** | `mapping.csv`(검수 ID→원본 ID), `reviewer-registry.csv`(코드·언어·능숙자 근거·일자), `build-manifest.json`(배포 파일 해시, refresh 이력) |
| `review/submissions/raw/<코드>/` | 수신함 | 검수자가 돌려준 `items-<코드>.csv`를 그대로 넣는다. 수정하지 않는다 |
| `review/submissions/merged/` | 운영자 | 결합 결과: `reviewed-audit-sample.csv`, `disagreements.csv`, `attestation-draft.json`, `receipt.json` |

## 절차

1. `reviewer-registry.csv`에 코드별 언어 능숙자 근거와 발송일을 기입한다. 실명·연락처는 필요할 때만 별도 명부에 둔다.
2. 언어별 두 검수자에게 각자의 zip만 보낸다. 상대 파일·정답 자료 열람 금지와 제출 전 상호 논의 금지는 `INSTRUCTIONS.md`에 있다.
3. 회신 파일을 `review/submissions/raw/<코드>/`에 넣고 `date_received`를 기록한다. 원판정 파일은 수정하지 않는다.
4. 결합: `python scripts/merge_review_submissions.py --review review --data data/draft-v4.3`
   - 검수 ID·문장 무결성·누락·형식을 검증하고 문제가 있으면 `merge-problems.json`을 남기고 중단한다.
   - 통과하면 원본 ID로 결합한 `reviewed-audit-sample.csv`(두 사람 원판정 그대로)와 `disagreements.csv`, `attestation-draft.json`, `receipt.json`(언어별 일치율·κ)을 만든다.
5. `disagreements.csv`(두 판정 불일치, 원본 정답과 다른 판정, `fluent=no`)로 합의 회의를 열고, 결과를 `attestation-draft.json`의 `resolved_items`에 `label`·`fluent`·`rationale`로 기입한다. 원판정은 그대로 둔다.
   합의 결과가 원본 정답과 다르면 데이터 오류이므로 **새 데이터 버전**이 필요하며 이 버전은 통과할 수 없다.
6. 문장 틀·활용표 체크리스트(`template-checklist-<코드>.csv`)를 모두 회수·확인한 뒤에만 `all_templates_and_forms_checked`를 `true`로 바꾼다. `inventory_hash`는 결합 도구가 채운 값(`template-inventory.json`의 SHA256, 현재 `6cfd9a2e…`)을 확인한다.
7. 인증:
   ```bash
   python -m flystudy certify-review --data data/draft-v4.3 \
     --csv review/submissions/merged/reviewed-audit-sample.csv \
     --templates review/submissions/merged/attestation-draft.json \
     --output reports/human-review.json
   python -m flystudy pilot-ready --g0 reports/g0-server.json --review reports/human-review.json \
     --data data/draft-v4.3 --graph artifacts/graphs/real.npz --tokenizer artifacts/tokenizer-v4.3.json \
     --output reports/pilot-ready.json
   ```

## 통과 기준 (v4 구현상 고정)

- 언어별 정확히 200문항, 같은 두 코드가 모든 문항을 판정, 원판정 일치율 ≥ 95%, Cohen κ ≥ 0.8.
- 원본 정답과 다른 판정·`fluent=no` 문항은 모두 `resolved_items`에 근거와 함께 기록되고 합의 결과가 원본과 같아야 한다.
- 확인서의 언어별 코드 목록(이름순)이 CSV와 일치하고 `all_templates_and_forms_checked: true`, `inventory_hash` 일치.

## 판정 기준 (검수자 안내와 동일)

두 문장의 내부 진술이 **같은 내용**을 표현하면 1, 다르면 0. 의미가 같으려면 A에서 B뿐 아니라 B에서 A도 성립해야 한다(양방향).
한 방향 함의만 성립하면 0이다. 지침 초판의 단방향 표현("A가 참이면 B도 반드시 참")은 2026-09-16 refresh에서 교정했다.

운영자 참고(검수자 안내에는 적지 않음): roles는 두 동사를 맞바꾼 오답, negation은 극성 반전, space는 방향 반전,
quantity는 두 수량 교환으로 오답을 만든다. 검수자에게는 의미 기준만 제시해 패턴 맞추기를 피했다.

지침·등록부 양식만 바꿀 때는 `python scripts/build_review_packages.py --output review --refresh-instructions "사유"`를 쓴다.
검수 ID·대응표는 바뀌지 않고 `build-manifest.json`에 refresh 이력이 남는다.
