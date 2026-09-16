# review/ — 사람 검수 배포·수신·결합 (운영자용)

현재 연구는 **Claude 단독 검수**로 전환했다. 새 절차는 `docs/AI_REVIEW_GUIDE.md`이며
`ai/`는 모델·프롬프트·입력·원응답·대화 기록의 비공개 보관 위치다. 아래는 기존 사람 검수
경로의 설명이다. 기존 dist ZIP은 변경하지 않으며 AI용 실제 요청문은 별도로 기록한다.

- `dist/<코드>.zip` : 검수자에게 보내는 유일한 파일. 언어별 검수자 2명 × 3언어 = 6개.
- `private/` : **배포 금지.** 검수 ID→원본 ID 대응표(`mapping.csv`), 검수자 등록부(`reviewer-registry.csv`: 코드·언어·능숙자 근거·일자; 실명·연락처는 저장소 밖 별도 관리), 배포 파일 해시·refresh 이력(`build-manifest.json`).
- `submissions/raw/<코드>/` : 회신된 `items-<코드>.csv`를 그대로 보관. 편집 금지.
- `submissions/merged/` : `python scripts/merge_review_submissions.py --review review --data data/draft-v4.3` 결과.

절차와 통과 기준은 `docs/HUMAN_REVIEW_GUIDE.md`. 배포본 재생성은 `build_review_packages.py`가 거부한다(검수 ID 안정성).
지침만 고칠 때는 `--refresh-instructions "사유"`로 zip을 다시 만든다(ID·대응표 불변).
같은 언어의 두 파일은 문장 내용으로 대응되므로 독립성은 절차(상대 파일·정답 열람 금지, 각자 제출)로 확보한다.
사람 검수 경로에서는 판정·실명을 사람이 기입한다. AI 경로는 Claude 원판정을 별도 증거로
연결하고 `certify-ai-review`를 사용한다. 두 종류의 검수를 같은 것으로 기록하지 않는다.
