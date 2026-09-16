# review/ — 사람 검수 배포·수신·결합 (운영자용)

- `dist/<코드>.zip` : 검수자에게 보내는 유일한 파일. 언어별 검수자 2명 × 3언어 = 6개.
- `private/` : **배포 금지.** 검수 ID→원본 ID 대응표(`mapping.csv`), 검수자 등록부(`reviewer-registry.csv`: 코드·언어·능숙자 근거·일자; 실명·연락처는 저장소 밖 별도 관리), 배포 파일 해시·refresh 이력(`build-manifest.json`).
- `submissions/raw/<코드>/` : 회신된 `items-<코드>.csv`를 그대로 보관. 편집 금지.
- `submissions/merged/` : `python scripts/merge_review_submissions.py --review review --data data/draft-v4.3` 결과.

절차와 통과 기준은 `docs/HUMAN_REVIEW_GUIDE.md`. 배포본 재생성은 `build_review_packages.py`가 거부한다(검수 ID 안정성).
지침만 고칠 때는 `--refresh-instructions "사유"`로 zip을 다시 만든다(ID·대응표 불변).
같은 언어의 두 파일은 문장 내용으로 대응되므로 독립성은 절차(상대 파일·정답 열람 금지, 각자 제출)로 확보한다.
판정·실명은 사람이 기입한다. 이 폴더의 어떤 파일도 AI가 판정을 채우지 않는다.
