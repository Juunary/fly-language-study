# Claude 단독 검수 — v4-ai-review-1

사용자의 명시적 선택에 따라 사람 검수를 **Claude 단독 검수**로 대체한다. 사람 검수 완료를
주장하거나 AI를 사람 이름으로 등록하지 않는다. 기존 사람 검수 기능은 별도 경로로 보존한다.
학습·평가·예산 기준은 유지하고, 새 실행 설정은 `configs/protocol-v4-ai.json`을 사용한다.

## 고정 운영 기준과 해석

- 언어당 같은 200문항을 서로 분리된 Claude 세션 두 개에서 판정한다. 총 여섯 세션의
  원판정·실제 모델 ID/버전·접속 방식·UTC 시각·생성 설정·프롬프트·첨부 입력·대화 내보내기를 보존한다.
  세션마다 실제 Claude 모델/버전을 기록하며 서로 다른 Claude 모델도 허용한다. 결과를 보고
  유리한 모델의 판정만 선택하지 않는다. UI가 정확한 버전이나 생성 설정을 노출하지 않으면 `not_exposed`로
  명시한다. 표시되는 모델 이름이나 실제 실행 시각을 추측하지 않는다.
- 원판정 일치율 ≥95%, κ ≥0.8은 **AI 반복 판정의 운영상 일관성 기준**으로 유지한다.
  이를 인간 평가자 간 신뢰도나 정답 타당성의 증거로 해석하지 않는다. 별도 세션이어도 오류가
  상관될 수 있으며, 높은 일치율이 언어별 편향을 제거하지 않는다.
- 모든 원본과 다른 의미 판정 및 `fluent=no`는 별도 Claude 재검토가 필요하다. 원판정을
  덮어쓰지 않는다. 정답 오류·실제 비문이면 새 데이터 버전에서 수정하고 재검수한다.
  단지 통과하려고 재실행하거나 `yes`로 바꾸지 않는다. 재검토가 원래 문장을 타당하다고 판단할
  때만 그 근거를 `resolved_items`에 보존하며, 원판정 일치율은 재검토로 높이지 않는다.
- 문장 틀·활용형·어휘 체크리스트 21행 모두 `checked=yes`, `issue=no`여야 한다.
  틀 자체의 문제가 남으면 새 데이터 버전으로 간다. AI 검수는 생성 문항의 언어적 타당성을
  확정하지 않으며, 결과는 **AI로 검수한 통제 코퍼스에서의 학습 효율**에 한정한다.
- 해시는 파일 변조를 검출한다. Claude 제공자가 실제 생성했는지, 답안을 사전에 보지 않았는지
  자체를 암호학적으로 인증하지는 않는다. 원대화·메타데이터는 운영자가 정직하게 보관해야 한다.

## 기존 KO 회신의 처리

사용자는 여섯 회신이 실제 분리된 AI의 판정임을 확인했다. 서버 보고상 KO 공간 관계
50문항씩, 총 100행이 `fluent=no`이고 사유가 비어 있다. 이 PC에는 회신이 없어 그 내용은
직접 확인하지 않았다. 경로에 `test-data`가 있다는 이유만으로 합성 자료로 판정하지 않는다.

1. 최초 CSV와 대화를 읽기 전용 보관하고 해시를 기록한다.
2. 기존 KO 세션에 해당 검수 ID와 문장을 제시하고 누락 사유를 보완하도록 한다.
   [Claude 요청문](CLAUDE_REVIEW_PROMPT.md)의 Supplement 부분을 사용한다.
3. 보완 응답은 새 파일로 받는다. 원판정을 수정했다면 Claude가 수정 이유도 기록해야 한다.
   최신 완전 CSV와 전체 대화, 최초 CSV 및 보완 요청문을 모두 증거에 연결한다.
4. `fluent=no`가 남으면 결합은 가능하지만 인증은 별도 재검토 기록이 있어야 가능하다.
   코드가 사유를 대신 작성하거나 자동으로 `fluent=yes`로 바꾸지 않는다.

## 증거 작업 공간 준비

```bash
python scripts/prepare_ai_review.py --output review/ai/claude-v1
```

이는 기존 ZIP·문항·대응표를 변경하지 않는다. 새 폴더에 입력 사본, 앞으로 사용할 요청문,
`evidence.json`의 미완성 양식만 만든다. 판정·모델 정보·실행 시각은 생성하지 않는다.
이미 수행한 검수는 이 새 요청문을 과거에 사용했다고 기록하면 안 된다. 당시 실제 요청문과
첨부 입력을 별도 보관하고 해당 경로/해시로 연결한다. 당시 기록이 없으면 증거 미완료다.

각 `runs` 항목을 실제 사실로 채운다. `provider=anthropic`, `model_id`, `model_version`,
`interface`(api/claude_web/claude_code), 서로 다른 `session_id`, UTC `executed_at_utc`,
`settings`를 기록한다. UI 미노출 설정은 `{"availability":"not_exposed"}`로 기록한다.
`fresh_context=true`와 `answer_key_exposed=false`는 실제 최초 판정 조건이 맞을 때만 설정한다.
최상위 `data_origin`은 실제 자료임을 확인한 경우 `actual_claude_responses`로 설정한다.

아티팩트는 모두 `{"path":"evidence.json 기준 상대 경로", "sha256":"SHA256"}` 형식이다.
- `inputs`: items/checklist/noun_forms/construction_examples/inventory 다섯 파일.
- `prompt`: 당시 실제 전체 요청문. `transcript`: 전체 대화 내보내기(후속 보완 포함).
- `items_response`, `checklist_response`: Claude의 완전한 원출력 CSV.
- `prior_artifacts`: 최초 회신·추가 요청·중간 응답 등의 아티팩트 목록. 최초 판정을 지우지 않는다.

체크리스트의 `checked`, `issue`는 yes/no로 명시한다. items의 `fluent=no`에는 comment가
필수다. 대화에서 추출한 CSV의 내용은 원응답과 일치해야 한다. 기록은 `review/ai/`에 보관하고
Git에 올리지 않는다. 모델/API 사용료는 GPU시간과 별도로 운영 기록에 남긴다.

## 결합·재검토·인증

기존 merged 결과와 확인서를 덮어쓰지 않도록 새 출력 폴더를 사용한다. raw에는 각 코드당
최신 완전 items CSV 하나만 두고 최초 파일은 별도 보존한다. 체크리스트는 raw 밖에 둔다.

```bash
python scripts/merge_review_submissions.py --review review --data data/draft-v4.3 \
  --review-mode claude_only --output review/submissions/merged-ai-v1
```

결합 도구가 만든 확인서는 자동 통과 자료가 아니다. 21행 체크리스트 결과를 확인한 뒤
`all_templates_and_forms_checked=true`를 기록한다. 불일치가 있으면 별도 새 Claude 세션에서
검토하며, `evidence.json`에 `adjudication` 항목을 추가한다. 기본 메타데이터·prompt·transcript·
prior_artifacts 형식은 runs와 같고, `response`는 `{"resolved_items": {...}}` 형태의 Claude
JSON 원출력이다. 여기에 있는 문항 집합과 판단·근거가 확인서의 `resolved_items`와 정확히
일치해야 한다. 원본 정답과 다른 최종 결론은 데이터 수정이 필요하므로 인증하지 않는다.

```bash
python scripts/prepare_ai_review.py --seal review/ai/claude-v1/evidence.json
python -m flystudy certify-ai-review --data data/draft-v4.3 \
  --csv review/submissions/merged-ai-v1/reviewed-audit-sample.csv \
  --templates review/submissions/merged-ai-v1/attestation-draft.json \
  --evidence review/ai/claude-v1/evidence.sealed.json --output reports/ai-review.json
```

seal은 기존 입력 해시를 검증하고 출력 해시를 채운 **별도 스냅샷**만 만든다. 인증을 대신하지
않으며 기존 sealed 파일을 덮어쓰지 않는다. 거절·수정·보완 이력도 버전별로 보존한다.

## 서버 기술 검증과 다음 단계

`src` 변경으로 전체 코드 해시가 바뀐다. 기존 G0/G1 보고서의 해시를 수정하거나 현재 증거로
재표기하지 않는다. 서버에서 새 코드와 `configs/protocol-v4-ai.json`으로 G0/G1을 새 예약·새
출력 경로에 재실행한다. 기존 장부의 사용량을 유지하고 추가 비용을 청구한다.

AI 인증과 새 G0가 실제 통과한 뒤에만 실행한다.

```bash
python -m flystudy pilot-ready --review-mode claude_only \
  --g0 reports/g0-ai-server.json --review reports/ai-review.json \
  --data data/draft-v4.3 --graph artifacts/graphs/real.npz \
  --tokenizer artifacts/tokenizer-v4.3.json --output reports/pilot-ready-ai.json
```

readiness와 freeze는 원응답·프롬프트·재검토 증거의 해시까지 전달한다. 증거가 바뀌면 실행을
거절한다. 주 런·분석 결과에 review_mode, human_reviewed=false, 개정 ID와 한계를 표시한다.
G2/G3·순서 파일럿·검정력·예산 검증과 freeze는 그대로 필요하며 AI 검수로 대체하지 않는다.
