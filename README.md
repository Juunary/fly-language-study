# 초파리 커넥톰 언어·학습 순서 연구 v4

영어·독일어·한국어의 네 의미 과제와 여섯 학습 순서를 비교하는 실행 패키지다. 서버의 과거 G0·G1 기술 검증 기록은 있으며, 주 연구 결과는 아직 없다. **현재 검수 방식은 Claude 단독 검수**다.

현재 작업은 [AI 검수 안내](docs/AI_REVIEW_GUIDE.md)를 따른다. 원응답·모델·프롬프트 기록을
`certify-ai-review`로 검증하며 사람 검수로 표시하지 않는다. 코드 변경 전 G0/G1은 과거
증거로 보존하고, 서버에서 새 코드의 기술 게이트를 재검증한다.

서버 Codex에 인계하려면 [서버용 프롬프트](docs/SERVER_PROMPT.md)를 사용한다. 정확한 실험 정의는 [프로토콜](docs/PROTOCOL_V4.md), 원문 검증은 [근거 검증표](docs/SOURCE_AUDIT.md), 현재 실측 상태는 [구현 상태](reports/implementation-status.json)에 있다.

## 포함된 것

- 실제 cb5k 배선: 5,000뉴런, 524,324개 연결. 학습된 가중치는 불러오지 않는다.
- 학습 가능한 희소 순환 분류 모델, 전체 문항 역전파, AdamW, 논리적 배치 256과 가변 마이크로배치.
- 3언어 96,000문항 초안, 누출·정답 편향 검사, 공유 BPE 4,096, 사람 검수용 파일.
- 고정 간격 전체 평가, A/B 교대, 12개 셀의 연속 두 번 통과, 단계 전환, 중단·재개, 독립 시험.
- G0 CUDA 정합성, G1 암기, G2/G3·여섯 순서 파일럿, 검정력·예산 진입 게이트.
- 5,000회 반복 사전 시뮬레이션, 두 계획 대비의 대응 t 검정·Holm·시드 블록 부트스트랩.
- 672 GPU시간 장부, 두 GPU 실행기, 별도 예산의 고정 복습·셔플·입력 표현 보조 연구.

## 지금 남아 있는 외부 검증

이 PC에는 Intel Arc GPU가 있다. 서버의 과거 CUDA 검증·처리량·G1 암기 기록은 `reports/`에 있다. 현재 코드에 대한 CUDA 검사는 서버에서 수행해야 하며, 로컬의 CUDA 미지원·건너뜀을 통과로 처리하지 않는다.

문항은 생성된 **초안**이다. 개정 Claude 검수 인증과 현재 G0 증거가 없으면 G2·G3·순서 파일럿을 열지 않는다. 본실험은 추가 G1·파일럿·검정력·예산 게이트가 필요하다. G1 통과는 일반화의 증거가 아니다.

## 설치

Python 3.11 이상을 사용한다. 인계 파일에는 가상환경과 CUDA 바이너리가 없다.

압축을 푼 직후, 파일을 갱신하기 전에 `python scripts/verify_handoff.py`로 전달 파일의 SHA256을 확인한다.

```bash
python -m venv .venv
# Linux: source .venv/bin/activate
# PowerShell: .\.venv\Scripts\Activate.ps1
```

GPU 서버에서는 먼저 서버 드라이버·nvcc와 호환되는 **CUDA PyTorch**를 공식 설치 안내에 따라 설치한다. `requirements-cpu-observed.txt`의 CPU torch를 서버에 그대로 설치하지 않는다. `torch.cuda.is_available()`, 장치 두 개, 두 장치의 compute capability `(8, 6)`을 확인한다.

```bash
python -m pip install -e ".[test]"
python -m pip install ninja
```

커널은 고정 커밋을 설치한다. 환경 변수는 Linux에서 `export CK_CUDA_ARCHS=86`, PowerShell에서 `$env:CK_CUDA_ARCHS='86'`이다.

```bash
python -m pip install --no-build-isolation "git+https://github.com/QuixiAI/connectome-kernels@c6be4cfea8ca2210098ef52344b88e85c683fd5c"
python -m pytest -q
python -m flystudy --help
```

PyTorch와 nvcc의 CUDA 주 버전 정합성 및 sm_86 빌드 방법은 [고정 커널 README](https://github.com/QuixiAI/connectome-kernels/blob/c6be4cfea8ca2210098ef52344b88e85c683fd5c/README.md)를 따른다. CPU 테스트 성공은 G0 성공을 대신하지 않는다.

## 전달된 산출물

| 경로 | 용도 |
|---|---|
| `configs/protocol-v4-ai.json` | Claude 검수 개정 후보: N=20, 평가 간격 10,240. 아직 확정 아님 |
| `artifacts/graphs/real.npz` / `.json` | 원문 커밋·해시가 기록된 실제 배선 |
| `data/draft-v4.3/` | 현재 문항·분할·검사 결과·사람 검수 자료 |
| `artifacts/tokenizer-v4.3.json` / `.meta.json` | 학습 분할에서만 학습한 공유 토크나이저 |
| `artifacts/power/final-grid.jsonl` / `.meta.json` | 완성된 사전 시뮬레이션 |
| `reports/power-summary.json` | 가정별 검정력 요약 |
| `reports/implementation-status.json` | 현재 코드·데이터 해시와 검증 상태 |

인계 ZIP에는 이전 데이터 초안, 원본 HF 가중치 파일, 가상환경을 넣지 않는다. 제공 그래프가 있으면 다시 다운로드할 필요가 없다. 재추출할 때만 다음을 쓴다.

```bash
python -m flystudy fetch-graph --output artifacts/graphs/real.npz --cache artifacts/hf-init
```

## CPU에서 재확인

```bash
python -m flystudy audit-data --data data/draft-v4.3
python -m flystudy baselines --data data/draft-v4.3
python -m flystudy cue-baselines --data data/draft-v4.3
python -m flystudy smoke --output artifacts/server-cpu-smoke
python -m flystudy power-report --input artifacts/power/final-grid.jsonl --output reports/power-summary.json
```

Smoke는 작은 합성 그래프의 구현 시험이다. 실제 초파리 모델의 학습 증거로 합치지 않는다. 이미 있는 출력 폴더를 덮어쓰지 않는다. 전체 시뮬레이션을 다시 만들 필요가 있을 때는 `simulate --output 새경로 --repetitions 5000`을 사용한다. `--quick` 결과는 본실험 승인에 사용할 수 없다.

## GPU 실행 순서

상세 인계 지시는 [SERVER_PROMPT.md](docs/SERVER_PROMPT.md)에 있다. 각 명령의 인자는 `python -m flystudy 명령 --help`로 확인한다.

1. `reserve`로 `configs/startup-reservations.json`을 장부에 예약한다. 최초 16 GPU시간의 기술 검증 배정이다.
2. `g0`를 실제 그래프와 두 sm_86 장치에서 실행한다. 현재 코드의 모든 G0 비교가 통과해야 한다.
3. 두 장치에서 `profile`을 실행하고 32/64/128/256 중 물리 배치를 선택한다. 전체 실행 비용은 이것으로 확정하지 않는다.
4. `g1`으로 128문항 암기 가능성을 확인한다. 문항을 수정하면 새 버전을 만들고 다시 검증한다.
5. [AI 검수 안내](docs/AI_REVIEW_GUIDE.md)에 따라 실제 Claude 원응답과 증거를 `certify-ai-review`로 인증하고, 현재 G0와 함께 `pilot-ready --review-mode claude_only`를 실행한다.
6. 별도 독립 시드로 `calibrate`를 언어별 수행한다. 5,120 간격으로 **A/B 두 전체 패널**을 모두 기록하므로 비용을 G2에 별도 예약한다. 기본 파일럿 20런 외에 두 시드×세 언어의 보정용 6런이 필요하다.
7. `reserve-pilots`가 단일 언어 3개·여섯 순서·혼합 1개를 시드별로 예약한다. `campaign`은 두 GPU에서 실행하고 기술 실패 시 새 런 배정을 중단한다.
8. `pilot-report`, `resolution-report`, `cost-report`, `assess-design`을 순서대로 만든다. 검정력·예산 판단이 불확실하면 새 독립 시드 블록을 추가한다. 시뮬레이션 범위 밖이면 확증실험을 승인하지 않는다.
9. 선택된 N·평가 간격으로 설정을 고정하고 `freeze`한다. `campaign`의 `--manifest`와 `--matrix` 모두 그 파일을 사용한다. `freeze`가 전 본실험 예산을 먼저 예약한다.
10. 모든 시드의 10조건이 완료되면 `analyze`한다. 누락·GPU 장애를 학습 미도달로 바꾸지 않는다.

```bash
python -m flystudy reserve --ledger runs/gpu-ledger.json --requests configs/startup-reservations.json
python -m flystudy g0 --graph artifacts/graphs/real.npz --output reports/g0-server.json --ledger runs/gpu-ledger.json --reservation-id g0-server-1
```

평가 간격을 바꾸면 **새 설정·출력 폴더·실제 순서 파일럿**을 사용한다. 두 파일럿 시드가 성공해도 모집단 미도달률의 95% 상한은 약 77.6%이므로 실행 승인과 같지 않다. 현재 시뮬레이션의 최대 미도달률은 50%다. 따라서 추가 반복이 필요할 수 있으며, 배정 예산이 부족하면 그 사실을 보고한다.

## 사람 검수

이 절은 보존된 사람 검수 경로다. 현재는 [Claude 단독 검수](docs/AI_REVIEW_GUIDE.md)와
`configs/protocol-v4-ai.json`을 사용한다. AI 회신을 `certify-review`로 인증하지 않는다.

`audit-sample.csv`는 600문항에 대해 두 판단씩 받는 1,200행이다. 언어마다 다른 두 사람이어도 된다. `noun-forms.csv`, `construction-examples.csv`, `template-inventory.json`, `src/flystudy/data.py`의 렌더링 규칙을 함께 검수한다.

검수자는 자신의 이름, `judged_label`(0/1), `fluent`(yes/no), 의견을 기입한다. 독립 원평가를 보존하고 불일치는 합의 근거를 별도 기록한다. 정답·문장을 고쳐야 한다면 데이터 새 버전과 새 검증 자료가 필요하다. `configs/review-attestation.example.json`은 형식 예시이며 통과 자료가 아니다.

추가 운영 기준으로 언어별 원평가 일치율 95% 이상, κ 0.8 이상을 요구한다. 이는 v4의 구현상 보수적 기준이므로 본실험 전에 기록하고 고정한다. 사람 이름이나 판단을 AI가 대신 채우지 않는다.

## 비용·재개·보조 연구

- 장부의 `reserved`는 배정량, `used`는 실제 GPU 점유 시간이다. GPU 둘을 사용하는 G0는 두 장치의 시간을 합산한다. 평가·체크포인트·독립 시험·준비 중 실패 시간도 비용이다.
- 프로세스가 강제 종료되면 자동 `finally`가 실행되지 않을 수 있다. 재시작 전에 서버 작업 기록으로 해당 예약의 실제 사용량을 복구하고 근거를 남긴다. 시간을 0으로 지우지 않는다.
- 실패한 런은 기존 `latest.pt`에서 명시적으로 재개한다. 새 예약 ID를 만들고 `run --resume ... --reservation-id 새ID`를 사용한다. 원래 비용을 다시 청구하지 않는다. 이미 완료된 런은 다시 시험하지 않는다.
- 예비비는 `transfer-reserve`로 목적 항목에 옮기고, 총 672시간은 유지한다. 보조 연구 삭감은 `release-auxiliary`의 review→shuffle→tokenizer 순서를 따른다.
- `fixed-review`는 본실험 시드 1·2의 여섯 순서 **12런 전체**를 예약한다. 300,000이 불가능하면 100,000, 그마저 불가능하면 생략한다. 주 결과는 변경하지 않는다.
- `reserve-auxiliary`는 셔플 또는 토크나이저 연구를 결과와 무관한 독립 시드 7001·7002의 완전 대응 묶음으로 예약한다. 셔플은 실제/셔플×여섯 순서×두 시드(24런), 입력 표현은 BPE 1,024/4,096×마이크로스텝 2/4×세 단일 언어×두 시드(24런)다. 예산에 안 들면 시작하지 않는다. 반환된 `bundle.json`의 런별 경로를 사용해 `run`을 실행한다. 주 10조건용 `campaign`에 보조 묶음을 넣지 않는다.
- 셔플 한 개 실현값의 결과를 모든 무작위 배선의 효과로 확대하지 않는다. 부분 완료된 보조 묶음의 유리한 결과만 골라 보고하지 않는다.

## 구현상 해석 범위

어휘가 작은 생성 코퍼스에서 단어 내부만 병합하는 BPE는 4,096개에 도달하지 못했다. 따라서 **공백을 넘는 whole-string byte BPE**를 사용한다. 이는 기록된 입력 표현의 선택이며 언어 자체 난이도가 아니다. 어휘 크기 민감도 연구도 같은 경계 규칙을 쓴다.

분할은 의미·오답 궤도·번역본·외곽 문장 틀 계열을 분리한다. 내부 문법 규칙과 단어는 공유한다. 모든 문법 틀이 완전히 새로운 일반화라고 주장하지 않는다. 독일어 성별 9조합은 양·음성 쌍을 보존해 셀 간 최대 두 문항 차이가 나는 정수 균형이며 정확한 수를 공개한다.

조건문·인용·질문 틀의 자연스러움과 의미 일치 정의를 Claude로 점검하더라도 인간 언어적 타당성이 확립되지는 않는다. 단서 기준선 통과는 모든 지름길이 제거됐다는 증거가 아니다. 이 실험은 살아 있는 초파리, CEFR, 언어의 보편적 난이도를 측정하지 않는다.
