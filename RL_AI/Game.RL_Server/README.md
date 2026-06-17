# Game.RL_Server

## Summary

M=4, N=16 기준 (총 64 동시 게임) 빠른 시작 명령어.

**1. 빌드**
```bash
dotnet build Game.RL_Server/Game.RL_Server.csproj -c Release
```

**2. RL_Server 4개 시작 (포트 9000~9003)**

macOS / Linux:
```bash
for i in 0 1 2 3; do
  dotnet run --project Game.RL_Server -c Release -- $((9000+i)) &
done
```

Windows PowerShell:
```powershell
$servers = 0..3 | ForEach-Object {
    Start-Process dotnet -ArgumentList "run --project Game.RL_Server -c Release -- $((9000+$_))" -NoNewWindow -PassThru
}
```

**3. 학습 시작 (Stage 1)**
```bash
python train_client.py --multi --m-servers 4 --n-agents 16 --base-port 9000 --total-episodes 10000 --device auto
```

평가 포함:
```bash
python train_client.py --multi --m-servers 4 --n-agents 16 --total-episodes 10000 --save-interval 2500 --n-eval 50
```

조기 Stage 2 전환 (평가 정체 시 자동 Stage 2):
```bash
python train_client.py --multi --m-servers 4 --n-agents 16 --total-episodes 10000 \
  --save-interval 2500 --n-eval 50 --early-stage2 --stage2-num-envs 32
```

throughput 측정 (eval/checkpoint 없이):
```bash
python train_client.py --multi --m-servers 4 --n-agents 16 --total-episodes 2000 --benchmark-mode
```

**3-2. Stage 2 TCP MCTS 정제 (선택)**

Stage 1 완료 후 동일 서버를 재사용해 MCTS 기반 추가 정제:
```bash
# Stage 1 완료 후 자동으로 Stage 2 연속 실행 (기본 2500 에피소드)
python train_client.py --multi --m-servers 4 --n-agents 16 --total-episodes 10000 \
  --stage2-num-envs 32 --stage2-mcts-depth 1 --stage2-mcts-top-k 2

# Stage 2만 단독 실행 (서버는 이미 떠 있어야 함)
python train_client.py --multi --m-servers 4 --base-port 9000 \
  --stage2-now --stage2-model-path models/final_model.pt \
  --stage2-num-envs 32 --stage2-mcts-depth 1 --stage2-mcts-top-k 2
```

**4. 종료**

macOS / Linux: `kill $(jobs -p)` 또는 각 터미널에서 `Ctrl+C`

Windows PowerShell — 서버 종료:
```powershell
$servers | Stop-Process -Force
```
포트 기반 종료 (변수가 없을 때):
```powershell
9000..9003 | ForEach-Object {
    $c = Get-NetTCPConnection -LocalPort $_ -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($c) { Stop-Process -Id $c.OwningProcess -Force }
}
```

학습: 에피소드 완료 후 자동 종료, 또는 `Ctrl+C`

---

SeaEngine 게임 로직을 TCP 서버로 노출하는 C# 프로젝트.  
Python `train_client.py`와 쌍을 이루어 **M × N 병렬 RL 학습**을 수행한다.

---

## 구조 개요

```
RL_Server 프로세스 (포트 9000)          Python 학습 프로세스 (단일)
┌─────────────────────────────┐        ┌──────────────────────────────────────┐
│  NetworkManager (Tick 루프)  │  TCP   │  TrainingSession                     │
│  ┌──────────────────────┐   │◄──────►│  _PortWorker × M (스레드)             │
│  │ RLInitHandler (ID=1) │   │        │    └─ asyncio × N 코루틴 (에피소드)   │
│  │ RLStepHandler (ID=2) │   │        │  공유 모델 ← M×N 배치로 PPO 업데이트  │
│  │ RLControlHandler     │   │        └──────────────────────────────────────┘
│  └──────────────────────┘   │
│  GameRegistry               │
│  ConnId → RLGameRoom × N    │
└─────────────────────────────┘
```

- **M개 RL_Server 프로세스** : 포트 번호로 구분 (9000, 9001, …)
- **N개 에이전트** : 하나의 서버 안에서 ConnId로 구분
- **총 M × N 동시 게임** / 총 프로세스 수 = M+1 (서버 M + Python 1)
- **단일 모델** : M개 스레드가 공유 — 한 번의 PPO 업데이트에 M×N 배치

---

## 사전 준비

### 1. .NET SDK 설치 확인

```bash
dotnet --version   # 10.0 이상 필요
```

### 2. 프로젝트 의존성 확인

`Game.RL_Server.csproj`는 아래 두 프로젝트를 참조한다.

| 참조 프로젝트 | 경로 |
|---|---|
| Game.Network | `../Game.Network/Game.Network.csproj` |
| SeaEngine | `../SeaEngine/csharp/SeaEngine/SeaEngine.csproj` |

경로가 다를 경우 `Game.RL_Server.csproj`의 `<ProjectReference>`를 수정한다.

### 3. 카드 데이터 확인

기본 경로: `Game.RL_Server/db/Cards.csv`  
빌드 시 출력 디렉토리(`bin/.../db/Cards.csv`)로 자동 복사된다.

---

## 빌드

```bash
# RL_AI 루트에서 실행
dotnet build Game.RL_Server/Game.RL_Server.csproj -c Release
```

---

## 실행

### 단일 서버

```bash
dotnet run --project Game.RL_Server -c Release -- <port> [card_data_path]
```

| 인수 | 기본값 | 설명 |
|---|---|---|
| `port` | `9000` | TCP 리스닝 포트 |
| `card_data_path` | *(자동 탐색)* | `Cards.csv` 절대 또는 상대 경로 |

예시:
```bash
dotnet run --project Game.RL_Server -- 9000
dotnet run --project Game.RL_Server -- 9000 /path/to/cards/Cards.csv
```

### M개 서버 병렬 시작

**macOS / Linux (bash):**
```bash
M=4
BASE_PORT=9000
for i in $(seq 0 $((M-1))); do
  dotnet run --project Game.RL_Server -c Release -- $((BASE_PORT+i)) &
done
```

**Windows PowerShell:**
```powershell
$M = 4; $BASE = 9000
$servers = 0..($M-1) | ForEach-Object {
    Start-Process dotnet -ArgumentList "run --project Game.RL_Server -c Release -- $($BASE+$_)" -NoNewWindow -PassThru
}
# 종료: $servers | Stop-Process -Force
```

서버가 준비되면 아래 로그가 출력된다:
```
[RLServer] Loading cards from: .../cards/Cards.csv
[RLServer] port=9000  MaxPerTick=512
[RLServer] Running. Ctrl+C to stop.
```

---

## Python 학습 클라이언트 연결

서버가 실행 중인 상태에서 `train_client.py`를 사용한다.

### 단일 서버 모드 (서버 1개, N개 에이전트)

```bash
python train_client.py --port 9000 --n-agents 32 --total-episodes 2000
```

### 다중 서버 모드 (M개 서버, M×N 에이전트 → 단일 모델)

```bash
python train_client.py \
  --multi \
  --m-servers 4 \
  --n-agents 32 \
  --base-port 9000 \
  --total-episodes 2000 \
  --device cuda
```

### Stage 2 TCP MCTS 정제

Stage 1 종료 후 **동일한 서버 포트**를 재사용해 MCTS 탐색 기반 PPO 정제를 수행한다.  
AI 턴마다 `MCTS_RESET → MCTS_STEP × depth`로 clone에서 후보를 평가한 뒤, 가장 높은 리프 가치(V(s))를 가진 액션을 **LIVE 게임에 적용**한다.  
상대는 **greedy / rule_based 번갈아** 사용 (random 제외 — 학습 신호가 너무 약하기 때문).

```bash
# Stage 1 완료 후 자동으로 Stage 2 연속 실행 (기본 2500 에피소드)
python train_client.py \
  --multi --m-servers 4 --n-agents 16 --total-episodes 10000 \
  --stage2-num-envs 32 \
  --stage2-mcts-depth 1 \
  --stage2-mcts-top-k 2

# Stage 2 단독 실행 (서버가 이미 실행 중이어야 함)
python train_client.py \
  --multi --m-servers 4 --base-port 9000 \
  --stage2-now \
  --stage2-model-path models/final_model.pt \
  --stage2-output-path models/final_model_stage2.pt \
  --stage2-num-envs 32 \
  --stage2-mcts-depth 1 \
  --stage2-mcts-top-k 2
```

| Stage 2 옵션 | 기본값 | 설명 |
|---|---|---|
| `--stage2-episodes` | `2500` | Stage 2 에피소드 수 |
| `--stage2-now` | false | Stage 1 없이 Stage 2만 실행 |
| `--stage2-model-path` | `models/final_model.pt` | Stage 2 시작 체크포인트 |
| `--stage2-output-path` | `models/final_model_stage2.pt` | Stage 2 저장 경로 |
| `--stage2-num-envs` | `32` | Stage 2 총 동시 env 수 (포트 수에 맞게 자동 분배) |
| `--stage2-mcts-depth` | `1` | MCTS lookahead 깊이 (1=1수 앞, 2=2수 앞) |
| `--stage2-mcts-top-k` | `2` | 후보 액션 수 |

> **포트 재사용**: Stage 2는 `--port` / `--base-port` / `--multi` / `--m-servers` 값을 그대로 사용한다. 별도 서버 포트 옵션 없음.

> **속도 예시**: Stage 1 기준 3.61 ep/s, Stage 2 (depth=1, top_k=2) 기준 약 1.2 ep/s — MCTS 오버헤드(`top_k × (1 + depth)` 라운드).

### 평가 포함 실행

체크포인트마다 baseline 상대(random / greedy / rule_based)와 M×N 병렬 평가:

```bash
python train_client.py \
  --multi \
  --m-servers 4 \
  --n-agents 32 \
  --total-episodes 2000 \
  --save-interval 500 \
  --n-eval 64 \
  --device cuda
```

저장과 평가 주기를 분리하려면:
```bash
  --save-interval 500 \   # 500 에피소드마다 체크포인트 저장 + 평가
  --eval-interval 250 \   # 250 에피소드마다 평가만
  --n-eval 64             # 상대별 64게임
```

---

## MCTS 평가 클라이언트 (mcts_session_client.py)

C# MCTS 핸들러(`MCTS_RESET=3`, `MCTS_STEP=4`)를 사용하는 깊이 제한 탐색 평가 클라이언트.  
기존 TCP 연결을 그대로 사용하면서 서버 측에서 `Game.Fork()`로 게임 상태를 복사, Python이 탐색을 제어한다.

### 동작 원리

```
Python mcts_reset()  →  C# MctsReset(): _mctsGame = _game.Fork()  →  스냅샷 반환
Python mcts_step(uid) →  C# MctsStep(uid): _mctsGame.UseAction(uid) →  스냅샷 반환
```

- `mcts_reset`을 호출할 때마다 live game 상태로 초기화 (후보별 탐색 시작점 동일)
- `mcts_step`을 depth만큼 반복 → 리프 상태 value 추정 (critic V(s) 또는 terminal ±1)
- top_k 후보 중 가장 높은 리프 value를 가진 액션 선택

### 사용 예시

```bash
# RL 모델 MCTS 평가 (depth=2, top_k=5)
python mcts_session_client.py --model-path models/final_model.pt \
  --opp rule_based --depth 2 --top-k 5 \
  --multi --m-servers 4 --total-matches 400

# depth/top_k 조정
python mcts_session_client.py --model-path models/final_model.pt \
  --opp greedy --depth 3 --top-k 3 --total-matches 200
```

### 주요 옵션 (mcts_session_client.py)

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--depth` | `2` | 탐색 깊이 (mcts_step 호출 횟수) |
| `--top-k` | `5` | 평가할 후보 액션 수 |
| `--model-path` | *(자동탐색)* | self RL 모델 경로 |
| `--opp` | `self` | 상대 에이전트 종류 |
| `--total-matches` | `400` | 총 게임 수 (8개 시나리오 균등 분배) |
| `--multi` | false | Multi-server 모드 |
| `--m-servers` | `4` | 서버 수 M |
| `--n-agents` | `32` | 서버당 슬롯 수 N |
| `--device` | `cpu` | PyTorch 디바이스 |
| `--seed` | `7` | 랜덤 시드 |

### 주의사항

- depth가 클수록 TCP 왕복 × 배치 추론 라운드 수 증가: `top_k × (1 + depth)` 라운드
- `depth=2, top_k=5` 기준 eval_client.py 대비 약 10× 느림 (탐색 오버헤드)
- 출력 형식은 eval_client.py와 동일 (`log/mcts_eval_{ts}.log`)

---

## 병렬 평가 클라이언트 (eval_client.py)

`bias_check.py` / `make_balance.py`의 TCP 병렬 버전. BeliefMCTS 미지원, 대신 **M×N 병렬**로 20~50배 빠름.

### 사용 예시

```bash
# RL self-play (make_balance 동등)
python eval_client.py --model-path models/final_model.pt \
  --multi --m-servers 4 --base-port 9000 --total-matches 400

# RL vs rule_based (bias_check 동등)
python eval_client.py --self rl --model-path models/final_model.pt \
  --opp rule_based --total-matches 400

# 두 모델 비교
python eval_client.py --self rl --model-path A.pt \
  --opp rl --opp-model-path B.pt --total-matches 400
```

### 주요 옵션 (eval_client.py)

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--self` | `rl` | self 에이전트 종류 (`rl` / `random` / `greedy` / `rule_based`) |
| `--model-path` | *(자동탐색)* | self RL 모델 경로 |
| `--opp` | `self` | 상대 에이전트 종류 (`self`=동일 모델 self-play) |
| `--opp-model-path` | *(없음)* | 상대 RL 모델 경로 (`--opp rl` 시) |
| `--total-matches` | `400` | 총 게임 수 (8개 시나리오에 균등 분배) |
| `--multi` | false | Multi-server 모드 |
| `--m-servers` | `4` | 서버 수 M |
| `--n-agents` | `32` | 서버당 슬롯 수 N |
| `--device` | `cpu` | PyTorch 디바이스 |
| `--seed` | `7` | 랜덤 시드 |

### 출력 형식

bias_check.py와 동일한 로그 형식 (`=== title ===`, 시나리오별 w/l/d/wr):

```
=== eval_client: rl(final_model) vs rl(final_model)(clone) ===
label=rl
episodes=400
self_wins=210
opp_wins=175
draws=15
self_win_rate_percent=52.50
...
- rl/귤/선공/같은 덱: self=28, opp=19, d=3, wr=56.0%, avg_steps=18.2, avg_turn=42.1
...
best=rl/귤/선공/다른 덱 (65.0%)
worst=rl/샤를로테/후공/같은 덱 (40.0%)
```

### 주의사항

- `greedy` / `rule_based` 에이전트(Python 측)는 board 정보 없이 `effect_id` 기반으로만 동작. C# 내장 상대(`opponentType=2/3`)는 보드 전체 상태에 접근하므로 품질이 더 높음
- BeliefMCTS 미지원 — raw policy 평가 전용
- 출력 파일: `log/eval_{ts}.log`, `log/eval_{ts}_summary.txt`

---

## train_client.py 주요 옵션

### Stage 1 (기본 학습)

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--port` | `9000` | 단일 서버 포트 |
| `--base-port` | `9000` | Multi 모드 시작 포트 |
| `--m-servers` | `4` | Multi 모드 서버 수 M |
| `--n-agents` | `16` | 서버당 동시 에이전트 수 N |
| `--total-episodes` | `10000` | 총 에피소드 수 (M×N 전체) |
| `--seed` | *(없음)* | 랜덤 시드 |
| `--device` | `auto` | PyTorch 디바이스 (`auto` / `cpu` / `cuda`) |
| `--log-interval` | `100` | 학습 로그 출력 간격 |
| `--save-interval` | `2500` | 체크포인트 저장 간격 (저장 시 평가도 실행) |
| `--eval-interval` | `0` | 평가 전용 주기 (0=비활성) |
| `--n-eval` | `50` | 평가당 layout별 에피소드 수 (8 layouts × 50 = 400판/상대) |
| `--max-turns` | `70` | 에피소드당 최대 턴 수 (초과 시 강제 종료) |
| `--reward-mode` | `dense_if_full` | 보상 방식 (`terminal` / `terminal_action` / `dense_if_full`) |
| `--benchmark-mode` | false | eval/checkpoint/save 비활성, throughput만 측정 |
| `--eval-sample-actions` | false | eval 시 stochastic 샘플링 사용 (기본: greedy/argmax). pre-eval baseline 측정 시 권장 |
| `--log-file` | *(자동)* | 로그 경로 (기본: `log/train_YYYYMMDD_HHMMSS.log`) |

### 조기 Stage 2 전환 (Early Stage 2)

Stage 1 평가 점수가 정체되면 자동으로 Stage 2로 전환한다.

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--early-stage2` | false | 정체 감지 시 Stage 2 조기 전환 활성 |
| `--early-stage2-patience` | `2` | 연속 정체 허용 횟수 |
| `--early-stage2-min-episode` | `2500` | 이 episode 이전에는 조기 전환 판단 안 함 |
| `--early-stage2-min-delta` | `0.01` | best score 대비 이 값 이상 향상돼야 개선으로 인정 |

### Stage 2 (TCP MCTS 정제)

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--stage2-episodes` | `2500` | Stage 2 에피소드 수 |
| `--stage2-now` | false | Stage 1 없이 Stage 2만 실행 |
| `--stage2-num-envs` | `32` | Stage 2 총 동시 env 수 (포트 수에 맞게 자동 분배) |
| `--stage2-model-path` | `models/final_model.pt` | 시작 체크포인트 경로 |
| `--stage2-output-path` | `models/final_model_stage2.pt` | 정제 모델 저장 경로 |
| `--stage2-mcts-depth` | `1` | MCTS lookahead 깊이 |
| `--stage2-mcts-top-k` | `2` | 후보 액션 수 |

### 출력 파일

| 파일 | 내용 |
|---|---|
| `log/train_YYYYMMDD_HHMMSS.log` | 전체 stdout 로그 |
| `log/train_summary.txt` | 학습 결과 요약 (에피소드·승률·체크포인트 목록) |
| `log/train_ckpt_{ep}_{ts}.txt` | 체크포인트별 평가 결과 (상대별 w/l/d, score) |
| `models/ckpt_ep{N}.pt` | 중간 체크포인트 모델 |
| `models/final_model.pt` | 최종 모델 |

---

## 권장 설정

| 하드웨어 | M (서버 수) | N (에이전트/서버) | 총 동시 게임 | 디바이스 | 실측 ep/s |
|---|---|---|---|---|---|
| Ultra 7 265K (20코어) | **4** | **32** | **128** | cpu | **3.61** |
| Ultra 7 265K (20코어) | 8 | 32 | 256 | cpu | 1.21 (비권장) |
| RTX 5070 + 고클럭 CPU | 4 | 32 | 128 | cuda | - |

> **하드 한계: M×N < 256** — 총 에이전트 수 256 이상 시 Windows TCP 스택 과부하로 LAN 연결 크래시.  
> **Ultra 7 265K 권장: M=4, N=32** — M=8은 E코어 밀림 + PPO 동기화 장벽으로 M=4 대비 3배 느림 (실측).  
> **RTX 5070에서 M=8은 반드시 `--device cpu`**: GPU TDR → BSOD.  
> M×N=256 이상으로 스케일하려면 프로토콜 멀티플렉싱(연결 1개로 N게임 다중화) 구현 필요.

---

## 확장 설계: 프로토콜 멀티플렉싱 (미구현)

현재 `reserved(i32)` 헤더 필드를 **slotId**로 전용하면, 연결 1개로 N개 게임을 다중화 가능:

```
현재:  ConnId → 게임 1개   (연결 M×N개)
확장:  ConnId + slotId → 게임 1개   (연결 M개, slotId = reserved 필드)
```

- 패킷 구조 변경 없음 — `reserved` 필드 의미만 재해석
- C#: `GameRegistry`를 `ConnId → RLGameRoom[]` 구조로 변경, `slotId`로 2차 dispatch
- Python: 워커당 연결 1개, 패킷마다 슬롯 인덱스 삽입 및 수신 라우팅
- 효과: M×N 연결 수 제한 제거, LAN 크래시 없이 N 확장 가능

---

## 통신 프로토콜 요약

모든 패킷: `[u32 length][16-byte header][body]`  
헤더: `flag(u32) | handlerId(i32) | queryNum(i32) | reserved(i32)`

| 방향 | HandlerID | Body 형식 |
|---|---|---|
| Python → C# (게임 초기화) | `1` (INIT) | `WriteString(p1Deck) WriteString(p2Deck) WriteString(p1Id) WriteString(p2Id) i32(aiSide) i32(opponentType) i32(seed)` |
| Python → C# (액션 전달) | `2` (STEP) | `WriteString(action_uid)` — 예: `"A001"` |
| Python → C# (MCTS 포크) | `3` (MCTS_RESET) | 없음 — live game을 `_mctsGame`으로 딥카피 |
| Python → C# (MCTS 스텝) | `4` (MCTS_STEP) | `WriteString(action_uid)` — STEP과 동일 형식 |
| C# → Python (스냅샷 응답) | 요청과 동일 ID | 바이너리 스냅샷 (result, state_vector, actions 등) |

`WriteString`: `[i32 byte_len][utf-8 bytes]` — `PacketWriter.WriteString`과 동일 형식.

**INIT 추가 필드 (i32 × 3)**

| 필드 | 값 | 설명 |
|---|---|---|
| `aiSide` | `0` | P1이 AI |
| | `1` | P2가 AI |
| | `-1` | 양쪽 외부 제어 (기존 self-play 동작) |
| `opponentType` | `0` | external — Python이 상대 턴 제어 |
| | `1` | random — C# 서버가 랜덤 액션 자동 처리 |
| | `2` | greedy — C# 서버가 greedy 휴리스틱 자동 처리 |
| | `3` | rule_based — C# 서버가 rule-based 휴리스틱 자동 처리 |
| `seed` | `≥ 0` | 해당 seed로 C# Random 초기화 (재현 가능) |
| | `< 0` | 무작위 시드 |

> `opponentType > 0` 이면 C# 서버가 상대 턴을 `Step()` 응답 전에 자동 소진한다.  
> Python은 AI 턴 스냅샷만 수신하므로 `opp_idx` 루프 불필요 (train_client.py에서 자동 스킵).

> MCTS_RESET / MCTS_STEP 응답은 STEP과 동일한 `SnapshotCodec`을 사용하므로 `parse_snapshot`으로 동일하게 파싱 가능.

---

## 종료

**macOS / Linux:** 각 터미널에서 `Ctrl+C`, 또는 `kill $(jobs -p)`

**Windows PowerShell:**
```powershell
# $servers 변수가 있을 때 (권장)
$servers | Stop-Process -Force

# 변수가 없을 때 — 포트로 찾아서 종료
9000..9003 | ForEach-Object {
    $c = Get-NetTCPConnection -LocalPort $_ -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($c) { Stop-Process -Id $c.OwningProcess -Force }
}
```

- Python 클라이언트: 에피소드 완료 후 자동 종료, 또는 `Ctrl+C`.

---

## 트러블슈팅

| 증상 | 원인 | 해결 |
|---|---|---|
| `Card data not found` | Cards.csv 경로 오류 | `Game.RL_Server/db/Cards.csv` 파일 확인, 또는 `-- <port> <절대경로>` 로 직접 지정 |
| Python `ConnectionRefusedError` | 서버 미실행 또는 포트 불일치 | 서버 로그 확인, `--port` 옵션 확인 |
| `KeyNotFoundException` (서버 로그) | 잘못된 action_uid 전송 | Python `parse_snapshot` 버전과 서버 버전 일치 확인 |
| 학습 속도가 느림 (M=8) | E코어 밀림 + PPO 동기화 장벽 | M=4, N=32 사용 권장 (Ultra 7 기준 M=4가 3배 빠름) |
| M×N=256 이상 LAN 연결 크래시 | Windows TCP 소켓 스택 한계 | M×N < 256 유지 (M=4, N=32 = 128으로 제한) |
| CUDA 미인식 (RTX 50 시리즈) | PyTorch Blackwell 미지원 | `pip install torch --index-url https://download.pytorch.org/whl/cu128` (cu128, conda `rl_ai` 환경 권장) |
| BSOD / PC 재부팅 (M=8, RTX 5070) | GPU TDR: 8 CUDA 스레드 동시 실행 → Windows 드라이버 타임아웃 | `--device cpu` 추가. M=4 이하면 `--device cuda` 가능 |
