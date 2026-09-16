# GPU 체크포인트 재개: RNG 상태 장치 수정

탐색 v2 서버 보고에서 `torch.load(..., map_location="cuda:...")`가 CPU용 난수 상태
ByteTensor까지 GPU로 옮겨 `torch.set_rng_state`가 실패했다. 원래 `restore_rng`에서 CPU·CUDA
난수 상태 텐서를 각각 `.cpu()`로 정규화하도록 수정했다. 시드·난수 바이트·모델·옵티마이저·
샘플러·커리큘럼 상태는 변경하지 않는다. 탐색 스크립트의 별도 우회 대신 공통 경로에서 처리한다.

## 검증 결과

- 로컬 전체 테스트: 68개 통과, 실제 CUDA가 필요한 새 테스트 1개 건너뜀.
- 기존 CPU 중단/재개 일치 검사 통과.
- CPU 회귀 검사: CPU RNG와 복수 CUDA RNG 상태를 모두 정규화한 뒤 복원 API에 전달함을 확인.
- 서버에서 실행할 실제 CUDA 회귀 검사:

```bash
.venv/bin/python -m pytest tests/test_model_runtime.py::test_cuda_mapped_checkpoint_restores_all_random_streams -q
```

이 검사는 체크포인트를 실제 `cuda:0`으로 로드한 뒤 Python·NumPy·CPU torch 및 모든 CUDA
장치의 난수열이 저장 시점 이후의 난수열과 정확히 일치하는지 확인한다. 건너뜀은 통과가 아니다.
서버 스크립트의 RNG 우회를 사용하지 않는다.

## 인계와 기존 결과 보존

- 새 코드 해시: `9b1811cde7e65bc0e284083a7f639110ceb133398137f079c752b433ccd0d773`.
- 과거 탐색 결과는 당시 코드/우회/실패 이력과 함께 보존한다. 재현된 781 업데이트의 일치는
  사용자가 제공한 v2 보고의 사실이며, 이 PC에는 v2 원자료가 없어 독립 재검증하지 않았다.
- 기존 체크포인트의 code_hash를 새 값으로 고치거나 재개 검사를 우회하지 않는다.
  다음 새 학습 전에 최종 코드로 G0/G1 검증을 묶어 수행한다. 그전 문서 편집마다 반복 실행하지 않는다.
- 서버의 아직 푸시하지 않은 v2 스크립트·설정·보고서·장부를 먼저 보존한다. 동기화 때 reset이나
  강제 push로 지우지 않는다. 새 CUDA 회귀 검사의 실제 결과와 추가 사용량을 보고한다.
