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

## 세션 실행과 기록 (Claude Code 하위 세션)

각 패키지는 `scripts/prepare_claude_sessions.py --workspace review/<ver>/ai/<name>`가 만든 `prompt-sent.md`
(일반 요청문 + 다섯 입력 파일의 절대 경로와 출력 형식)와 한 단락짜리 `launcher-text.md`로 시작한다. 세션은
새 컨텍스트의 Claude Code 하위 에이전트(도구는 파일 읽기만)이며, 실행 기록(JSONL 전사)이 곧 증거다.

세션이 끝나면 판정을 손대지 않고 아래로 기록한다.

```bash
python scripts/record_ai_review_session.py --workspace review/<ver>/ai/<name> --code <CODE> \
  --agent-transcript <세션 JSONL> --launcher review/<ver>/ai/<name>/<CODE>/launcher-text.md \
  --session-id agent-<id> --submissions review/<ver>/submissions/raw
```

- 전사에서 모델 식별자(`model` 첨부의 `modelId`), 시작·종료 시각, 정지 사유, 출력 토큰 수를 그대로 옮긴다.
  추정값을 쓰지 않는다. 전사 사본은 `<CODE>/session-transcript.jsonl`, 읽기용 정리본은 `transcript.txt`.
- 응답이 출력 한도에서 끊긴 경우 두 형태만 허용한다. (1) **이어쓰기**: 다음 메시지가 헤더 없이 끊긴 행부터
  다시 쓰면 앞 블록을 이어 붙이고, 끊긴 부분 행 하나만 버린다(같은 review_id의 완전한 행이 있어야 하며, 부분
  행이 완전한 행의 접두사가 아니면 거부). (2) **재시작**: 다음 메시지가 헤더부터 완전한 블록을 다시 내면 그 블록이
  최종 응답이고, 끊긴 시도는 `superseded_attempts`에 행 수와 최종 블록과의 판정 불일치 목록을 남긴다. 두 경우 모두
  `evidence.json`의 `response_assembly`에 기록된다.
- 검증(행 수·식별자·문장·코드 불변, 판정 존재, `fluent=no`의 comment, 체크리스트 행 집합 일치)에 실패하면
  기록하지 않고 종료한다. 그 시도의 전사는 `<CODE>/attempt-<n>-<id>/`에 보존해 `prior_artifacts`에 등록하고,
  같은 패키지로 **새 세션**을 실행한다. 판정을 보고 세션을 고르지 않는다(거부 사유는 기계적 검증뿐이다).

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

## 개정 v5.1-ai-review-2 (2026-09-17) — 체크리스트 지적의 쟁점별 조정

**시점과 성격.** 이 개정은 자료 v4.6에 대한 검수 2차(`review/v6`)의 결과를 확인한 뒤, 파일럿·보정·본실험을 하나도 시작하기 전에
운영자 결정(선택지 b)으로 채택했다. 채택 시각·사유·채택 당시 알고 있던 지적 9행은 `reports/ai-review-rule-amendment.json`에 있다.
검수 1차·2차는 v4-ai-review-1 규칙으로 처리했고 그 결과·파일은 바꾸지 않는다. 이 개정을 그 이전부터 정해진 규칙인 것처럼 소급하지 않는다.

**바뀐 것.** `issue=yes` 1행이 자동으로 자료 개정·전체 재검수 사유가 되지 않는다. 대신 모든 `issue=yes` 행은 **근거가 기록된 쟁점별 조정**의
대상이 된다(`flystudy.ai_review.validate_checklist_adjudications`).

- 차단 범주(하나라도 확인되면 인증 거부, 영향받은 부분만 새 버전에서 수정): 주 학습·평가 문항의 정답 오류, 의도한 의미를 훼손하는 문법·어휘
  오류, 언어 간 의미 대응의 실질적 불일치, 과제 정의로 해소되지 않아 정답을 달라지게 하는 모호성, 누출·변조·필수 검수 누락.
- 비차단 범주(근거와 적용 범위를 남기고 인증 보고서의 한계로 기록): 문체 선호, 과제 정의 밖의 해석, 보조 평가에만 해당하는 문제.
  보조 평가 범위의 한계는 본실험 전에 보조 지표의 제외 또는 해석 제한으로 명시한다.
- 단일 지적이라는 이유로 자동 기각하지 않고, 다른 세션이 지적하지 않았다는 이유(다수결)로 통과시키지 않는다. 원래 지적은 수정하지 않는다.

**절차.** `scripts/adjudicate_checklist_issue.py`가 쟁점마다 (1) `prepare`: 지적 행·동료 세션 행·과제 정의·관련 구문 예시(검수한 자료 버전, 참조
label 포함)·관련 어휘표만 담은 패키지와 요청문을 만들고, (2) `run`: 도구 없는 새 Claude 세션(`claude -p --tools '' --strict-mcp-config`,
빈 샌드박스)을 실행해 출력 JSON과 세션 전사를 보관하고, (3) `record`: 단일 구조화 응답(쟁점 ID, 검토한 입력 해시, 영향 범위, 구체적 오류·반례,
차단 여부와 근거, 최소 수정 또는 해석 범위)을 검증·원문 보관하고 메타데이터를 채운다. (4) `assemble`: `evidence.json`에 `rule_amendment`와
`checklist_adjudications`를 더한 새 증거 파일과, 판정별 확인(`checklist_adjudications`)을 담은 확인서를 만든다. 200문항 전체 재판정이나
CSV 재출력은 요청하지 않으며, 통과를 유도하는 문구를 넣지 않는다. 쟁점별 세션은 한 번씩이며 같은 자료를 통과할 때까지 반복 제출하지 않는다.

**검증기.** `validate_ai_evidence`는 모든 지적 행이 정확히 하나의 조정 항목에 덮이는지, 각 항목이 새 세션 메타데이터·요청문·전사·입력 해시
목록·응답 원문을 갖는지, 응답이 입력 해시를 그대로 되풀이하는지, 범주와 차단 여부가 일치하는지, 확인서가 각 판정을 정확히 인정하는지 검사한다.
하나라도 차단이면 인증을 거부한다. 완전 일치(일치율 1.0)는 판정 일관성의 근거일 뿐 오류가 없다는 보증이 아니다.
