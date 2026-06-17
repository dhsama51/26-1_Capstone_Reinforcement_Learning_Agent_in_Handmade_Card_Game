# RL_AI Agent Guide

이 파일은 다른 AI 에이전트가 이 프로젝트를 빠르게 파악하고 작업할 수 있도록 작성된 컨텍스트 문서다.

---

## 최신 기준

현재 작업 기준의 최신은 `RL_AI_delta`보다도 **현재 작업트리의 TCP speed-first 흐름**이다.  
즉, 아래 항목들이 지금 코드와 문서의 기준선이다.

- `train_client.py` 중심의 TCP 병렬 학습
- `eval_client.py` / `mcts_session_client.py` 기반 병렬 평가 축
- `Game.RL_Server`의 `RLGameRoom` / `RLHandler` / `RLProtocol` / `RLCodecs`
- `SeaEngine/observation.py` / `SeaEngine/csharp/SeaEngine/RL/RlObservationExporter.cs`
- `training/reward.py`, `training/trainer.py`, `training/storage.py`의 PPO / reward / sample weight 조정
- Windows 런처 `start_w.py`, `bias_check_w.py`, `make_balance_w.py`

### zip 4 기준으로 다시 맞춘 부분

- `HANDLER_INIT`는 이제 `aiSide / opponentType / seed`를 보내는 zip 4 형식으로 정리했다.
- `Game.RL_Server/src/RLGameRoom.cs`는 `OpponentType.External / Random / Greedy / RuleBased` 기반 자동 opponent 루프로 바꿨다.
- `Game.RL_Server/src/RLHandler.cs`는 위 INIT body를 직접 읽어서 `RLGameRoom.Init(..., aiSide, opponentType, seed)`를 호출한다.
- `train_client.py`는 `opponent_name / auto_opponent` 문자열 플래그 대신 `ai_side / opponent_type / seed`를 보내는 쪽으로 맞췄다.
- `training/trainer.py` 기본 PPO는 zip 4 수치 쪽(`lr=3e-4`, `clip=0.2`, `entropy=0.025`, `epochs=4`, `target_kl=0.03`)으로 되돌렸다.
- `train_client.py`의 Stage 2 PPO도 zip 4 수준으로 다시 맞췄다.

### 현재 작업트리의 핵심 방향

- PythonNet 기반 재계산 경로는 기본 학습 경로에서 제거했다.
- TCP로 받은 server vector를 기본으로 쓰고, 런타임 체크 로그로만 상태를 확인한다.
- `RlObservationFrame`은 frame-only wire로 유지한다.
- C# 서버는 `server_auto`와 `AdvanceOpponentUntilAiTurnOrDone()`로 opponent 턴을 내부에서 처리한다.
- `dense_if_full` reward와 speed-first 병렬 기본값을 기준으로 학습한다.

### 참고: RL_AI_delta

- `RL_AI_delta`는 과거 패치셋으로, 현재 코드의 일부 아이디어를 담고 있지만 최신 기준은 아니다.
- `pythonnet`, `BeliefMCTS`, `SnapshotCodec` 옛 설명은 참고용으로만 본다.
- 실제 운영 기준은 현재 작업트리의 `train_client.py` / `charlotte_ablation.py` / `RLCodecs.cs` / `RLHandler.cs` 정합이다.

### 반영 원칙

- 새 변경이 생기면 이 `Agent.md`에 우선 기록한다.
- 현재 작업트리를 기준으로 정리하고, 예전 zip은 비교 참고만 한다.
- wire protocol 안정성을 깨는 변경은 반드시 영향도를 확인한다.

---

## 델타

### TCP / 학습 클라이언트

- `train_client.py`는 **TCP 경로를 기본**으로 쓴다.
- PythonNet 기반 observation 재계산은 기본 경로에서 제거했고, 서버가 준 `state_vector` / `action_feature_vectors`를 우선 사용한다.
- `SEAENGINE_TCP_REBUILD_OBS`는 비교/디버그용 분기로만 남겼다.
- `SEAENGINE_TCP_OBS_RUNTIME_LOG`로 observation mode / server_auto / 차원 정보를 런타임에 찍을 수 있다.
- `_PortWorker` 시작 시 observation mode, server vector 차원, 런타임 체크 로그가 1회 찍히도록 했다.
- `--ai-deck {both,orange,charlotte}` / `--opp-deck {both,orange,charlotte}`로 AI 덱과 상대 덱을 각각 필터링할 수 있다.
- layout 수는 필터 결과에 맞춰 동적으로 줄어든다. 예: `--ai-deck orange --opp-deck both`면 P1/P2 × Orange vs both 조합만 남는다.
- 현재 병렬 기본값은 `M=2, N=64`로 맞추고 있으며, PPO `update_epochs` 기본값도 `2`로 낮춰 업데이트 부담을 줄이는 방향이다.
- `training/trainer.py`는 full-batch PPO에서 mini-batch PPO로 옮겼고, 기본 `mini_batch_size=512`를 사용한다.
- stage 1은 짧게 가져가고(stage1 default `total_episodes=1024`), checkpoint는 `512`마다 보되 checkpoint eval은 `25` games/layout로 분리했다.
- stage 2는 `MCTSTrainingSession` 기반으로 돌리며(`stage2_episodes=10000` 기본), plateau가 보이면 stage 1을 조기 종료하고 stage 2로 넘긴다.
- stage 1 / stage 2의 checkpoint monitor는 이제 patience plateau 대신, checkpoint score가 이전 best보다 떨어지면 즉시 종료하는 방식으로 바뀌었다.
- checkpoint eval에서 가장 약한 layout 승률이 `0.80` 이상이면 plateau 여부와 무관하게 stage 2로 넘기도록 했다.
- stage 2 post-eval에서도 가장 약한 layout 승률이 `0.80` 이상이면 stop 신호를 찍는다.
- stage 2는 기본적으로 500 episode마다 checkpoint를 저장하고, checkpoint eval은 25 games/layout로 돌린다.
- checkpoint eval은 승패만 보므로 `reward_mode=terminal`로 가볍게 돌리고, dense transition reward 계산을 피한다.
- stage 2의 depth=1 경로는 Python이 policy top-k만 뽑고, C# `MCTS_BATCH`(handler 5)로 후보 fork/auto-advance를 한 번에 처리한다.
- MCTS batch 응답은 이제 `result/winner_id/state_vector`만 담는 leaf 최소 포맷이다.
- stage 2의 depth=1 batch 윈도우는 16으로 두어, 한 번에 너무 많은 env를 묶지 않게 했다.
- depth>1은 아직 기존의 개별 `mcts_reset / mcts_step` 경로를 fallback으로 남겨두었다.
- `parse_snapshot()`는 `result`, `winner_id`, `turn`, `active_player`, `state_vector`, `actions`, `players`, `board`를 읽는다.
- TCP snapshot에서 feature 차원 mismatch가 나면 조용히 덮지 않고 런타임 에러로 드러나게 했다.
- `server_auto`는 기본 ON으로 두고, 내부 opponent 처리는 C#에서 처리한다.
- C# 내부 opponent는 `opponentType > 0`일 때만 자동 전진한다.
- stage 2의 MCTS는 policy top-k를 Python이 고르고, C# `RLGameRoom.MctsStep()`이 후보 적용 후 opponent auto-advance까지 처리하도록 바뀌었다.
- 즉 stage 2의 leaf는 이제 C# 서버가 한 번 더 굴린 뒤의 상태를 Python value head가 보조 평가하는 흐름이다.

### Charlotte ablation

- `charlotte_ablation.py`를 TCP 기반 Charlotte-only 실험으로 추가했다.
- `Charlotte vs Charlotte`만 고정하는 layout으로 side / deck 변수를 줄였다.
- 샤를로테 밸런스 패치: `Cl_L` 리더 HP +4, 그 외 샤를로테 유닛 HP +1이 카드 로딩 시점에 적용된다.
- 체크포인트마다 평가하고, plateau가 누적되면 Stage 2로 전환하는 흐름을 붙였다.
- `--checkpoint-interval`, `--early-stage2`, `--early-stage2-patience`, `--early-stage2-min-episode`, `--early-stage2-min-delta` 옵션을 노출했다.
- `--checkpoint-eval-matches`를 추가해 checkpoint 평가를 본평가보다 훨씬 싸게 돌릴 수 있게 했다.
- `--reward-mode`를 노출해서 `terminal`, `terminal_action`, `dense_if_full`을 선택할 수 있게 했다.
- stage 2는 별도 TCP server MCTS 정제 대신 `MCTSTrainingSession`을 사용한다.

### C# frame / codec 정합

- `RlObservationExporter.cs`의 `RlObservationFrame`을 wire 기준으로 맞췄다.
- `RLCodecs.cs` / `RLHandler.cs`는 `RlObservationFrame`만 직렬화/역직렬화하도록 정리했다.
- on-wire에 `P1Id/P2Id` 같은 추가 메타데이터를 붙이지 않도록 수정했다.
- `turn` 타입은 frame 정의에 맞게 `i32`로 맞췄다.
- `SnapshotCodec` read/write roundtrip 테스트를 추가해서 C# 직렬화 정합성을 검증했다.

### Observation / feature

- `SeaEngine/observation.py`에 action relation feature를 추가했다.
- TCP 쪽 `action_feature_vectors` 차원을 `108`로 맞췄다.
- state / action feature shape mismatch를 런타임에서 바로 확인하도록 정리했다.

### Reward / PPO / 학습 강도

- `training/reward.py`에서 `dense_if_full`과 fallback shaping을 조정했다.
- `training/storage.py` / `training/trainer.py`에 sample weight와 PPO update 강도 조정이 들어갔다.
- 승리 trajectory를 더 세게 반영하려는 가중치 흐름을 실험했다.
- learning rate, clip epsilon, entropy, target KL은 여러 번 재조정했다.

### 실행 / 병렬 기본값

- 병렬 기본값은 여러 차례 바뀌었고, 현재는 speed-first 기준으로 다시 정리 중이다.
- 서버 병렬 수와 에이전트 수, auto opponent, observation rebuild 여부를 분리해서 제어할 수 있게 했다.
- 실행 명령은 `train_client.py`와 `charlotte_ablation.py` 각각에 맞춰 정리해두고 있다.

### 문서

- README류 문서에 `train_client.py` 옵션 명세를 맞췄다.
- 런타임 체크, feature 차원, 체크포인트, Stage 2 전환 관련 설명을 업데이트했다.

## 프로젝트 개요

SeaEngine 카드게임을 대상으로 **M×N 병렬 RL 학습**을 수행하는 시스템.

- **C# 서버** (`Game.RL_Server`): SeaEngine 게임 로직을 TCP 서버로 노출
- **Python 클라이언트** (`train_client.py`): M 스레드 × N asyncio로 M×N 게임을 동시 실행, 단일 PPO 모델 학습
- **학습 프레임워크** (`RL_AI` 패키지): PPOTrainer, agents, experiment 등

---

## 디렉토리 구조

```
RL_AI/                          ← 프로젝트 루트 (Python 패키지 루트)
├── train_client.py             ← 메인 학습 클라이언트 (Stage 1 TCP 학습)
├── eval_client.py              ← 병렬 평가 클라이언트 (bias_check/make_balance TCP 버전)
├── mcts_session_client.py      ← MCTS 탐색 병렬 평가 (C# MCTS 핸들러 사용)
├── start.py                    ← 기존 단일 프로세스 학습 스크립트 (참조용)
├── agents/
│   ├── __init__.py
│   └── seaengine_agents.py     ← SeaEngineRLAgent, SeaEngineRandomAgent, SeaEngineGreedyAgent,
│                                  SeaEngineRuleBasedAgent, BeliefMCTSAgent, PPOActorCritic 등
├── training/
│   ├── trainer.py              ← SeaEnginePPOTrainer (update_from_buffer)
│   ├── storage.py              ← RolloutBuffer, RolloutStep
│   ├── experiment.py           ← _build_training_opponent_schedule, _format_plan_counts,
│                                  _load_saved_rl_agent 등 실험 유틸리티
│   └── evaluator.py
├── SeaEngine/
│   ├── action_adapter.py       ← choose_action_with_agent(agent, snapshot)
│   └── observation.py          ← STATE_VECTOR_DIM
├── Game.RL_Server/             ← C# TCP 서버 프로젝트
│   ├── src/
│   │   ├── Program.cs
│   │   ├── RLServerApp.cs      ← NetworkManager 래핑, 핸들러 등록, Tick 루프
│   │   ├── RLProtocol.cs       ← HandlerID 상수 (INIT=1, STEP=2, MCTS_RESET=3, MCTS_STEP=4)
│   │   ├── RLGameRoom.cs       ← ConnId 1개당 게임 슬롯 (live game + mcts 복사본)
│   │   ├── RLHandler.cs        ← 모든 핸들러 (Init, Step, MctsReset, MctsStep, Control)
│   │   └── SnapshotCodec.cs    ← SnapshotPacket 직렬화
│   ├── db/Cards.csv
│   └── README.md
├── log/                        ← 학습 로그, 체크포인트 리포트
└── models/                     ← 체크포인트 (.pt), final_model.pt
```

> **Python 패키지 경로**: `c:\Users\addbum421` 가 `sys.path`에 있어야 `RL_AI.*` 임포트 가능.  
> `train_client.py`의 `_ensure_importable()` 이 `Path(__file__).parent.parent` (= `c:\Users\addbum421`)를 자동으로 추가.

---

## 핵심 아키텍처: train_client.py

### 병렬화 구조

```
Python 프로세스 (단일)
├── TrainingSession.run()
│   ├── _PortWorker (스레드 0) ── asyncio loop ── N개 RLServerEnv (TCP → port 9000)
│   ├── _PortWorker (스레드 1) ── asyncio loop ── N개 RLServerEnv (TCP → port 9001)
│   ├── ...
│   └── _PortWorker (스레드 M-1)
│
└── 공유 trainer (SeaEnginePPOTrainer)
    └── 공유 agent (SeaEngineRLAgent) ← M×N 경험으로 PPO 업데이트
```

- **GIL 해제**: PyTorch forward pass는 GIL을 해제하므로 M 스레드가 실질적으로 병렬 추론
- **큐 인터페이스**: `_task_q` (SimpleQueue) 로 메인→워커 작업 전달, `_result_q` 로 결과 수신

### 워크플로우 (TrainingSession.run)

1. **Workers 시작** — M개 `_PortWorker` 스레드, 각자 N개 TCP 연결
2. **Pre-eval** — random/greedy/rule_based 대상 M×N 병렬 평가 (학습 전 베이스라인)
3. **스케줄 생성** — `_build_training_opponent_schedule` 호출
   - 커리큘럼: 초반 random 위주 → 점진적 self-play 도입
   - `save_interval` 마다 `self_ep_{N}` 이름을 pool에 자동 추가 (스케줄 생성 시 미리 반영)
4. **학습 루프** — 각 이터레이션:
   - 워커별로 스케줄에서 N개 상대를 꺼내 `submit(it, base_idx, total_envs, slot_opps: list)`
   - 결과 수집 → `RolloutBuffer` 병합 → `trainer.update_from_buffer(merged)`
   - `save_interval` 도달 시: 체크포인트 저장 + 동결된 self-play 에이전트 로드 → `pool_dict` 추가
   - `save_interval` or `eval_interval` 도달 시: 체크포인트 평가 + 리포트 저장
5. **Post-eval** — 학습 후 동일 베이스라인 평가
6. **정리** — `final_model.pt` 저장, 워커 종료

### _PortWorker._collect: 스텝 동기화 배치 추론

`_collect`는 N개 게임을 **스텝 단위로 동기화**하여 진행한다 (이전 `async_collect_episode` 순차 방식 제거).

```
while True:
    active  = [AI 차례인 게임들]  + [상대 차례인 게임들]
    ai_idx  = AI 차례 게임들
    opp_idx = 상대 차례 게임들

    # ai_idx 전체를 하나의 배치로 GPU forward pass (1 call)
    outs = trainer.agent.compute_policy_output_batch(sv, av, la)

    # opp_idx는 상대 에이전트로 즉시 처리
    # 모든 active 게임에 동시 apply_action (asyncio.gather)
```

- AI 차례 게임들을 한 번의 `compute_policy_output_batch` 호출로 처리 → GPU 왕복 최소화
- `asyncio.gather`로 `apply_action` 병렬화 → TCP 왕복 지연 최소화
- `max_turns` 초과 시 해당 게임 강제 종료 (패배 보상)

### 주요 클래스/함수

| 이름 | 위치 | 역할 |
|---|---|---|
| `_PortWorker` | `train_client.py` | 스레드-로컬 asyncio로 1서버×N 에이전트 관리 |
| `_PortWorker.submit(it, base_idx, total_envs, opponents)` | | opponents = 단일 에이전트(브로드캐스트) 또는 list[agent] |
| `TrainingSession` | `train_client.py` | M 워커 조율, 전체 학습 워크플로우 |
| `TrainingSession._load_self_play_agent(ep)` | | 체크포인트를 동결된 SeaEngineRLAgent로 로드 |
| `TrainingSession._get_batch_opponents(schedule, ep_start, n, pool_dict, fallback)` | | 스케줄 슬라이스 → 에이전트 리스트 |
| `_PortWorker._collect(envs, it, base_idx, total_envs, opponents)` | `train_client.py` | N개 게임을 스텝 동기화로 병렬 수집 (배치 추론) |
| `RLServerEnv` | `train_client.py` | TCP 연결 1개 = 게임 슬롯 1개 |
| `_build_training_opponent_schedule(...)` | `RL_AI.training.experiment` | 커리큘럼 스케줄 생성 |
| `SeaEnginePPOTrainer.update_from_buffer(buf)` | `RL_AI.training.trainer` | PPO 업데이트, 손실 dict 반환 |

---

## TCP 프로토콜

```
패킷: [u32 length][16-byte header][body]
헤더: flag(u32) | handlerId(i32) | queryNum(i32) | reserved(i32)
```

| 방향 | HandlerID | Body |
|---|---|---|
| Python → C# (게임 초기화) | `1` (INIT) | `WriteString(p1Deck) WriteString(p2Deck) WriteString(p1Id) WriteString(p2Id)` |
| Python → C# (액션) | `2` (STEP) | `WriteString(action_uid)` |
| Python → C# (MCTS 포크) | `3` (MCTS_RESET) | 없음 — live game을 mcts 복사본으로 포크 |
| Python → C# (MCTS 스텝) | `4` (MCTS_STEP) | `WriteString(action_uid)` — STEP과 동일 형식 |
| C# → Python (스냅샷) | 요청과 동일 ID | 바이너리 스냅샷 (result, state_vector, actions 등), `parse_snapshot`으로 파싱 |

`WriteString`: `[i32 byte_len][utf-8 bytes]`

---

## 커리큘럼 스케줄 (experiment.py)

`_build_training_opponent_schedule` 의 가중치:

| 진행도 | random | greedy | rule_based | self-play |
|---|---|---|---|---|
| 0–15% | 0.55 | 0.25 | 0.10 | 0 |
| 15–35% | 0.40 | 0.30 | 0.15 | 점진적 |
| 35–60% | 0.28 | 0.30 | 0.17 | 점진적 |
| 60–80% | 0.20 | 0.25 | 0.20 | 점진적 |
| 80–100% | 0.15 | 0.25 | 0.20 | 점진적 |

- self-play 가중치 = `max(0, 1 - 위 합계)` 를 최근 12 스냅샷에 균등 분배
- `save_interval` 마다 `self_ep_{N}` 이름을 pool에 추가

---

## eval_client.py (병렬 평가 클라이언트)

`bias_check.py` / `make_balance.py`의 TCP 병렬 버전. 동일한 로그 형식, BeliefMCTS 미지원, 대신 M×N 병렬로 20~50배 빠름.

### 구조

```
EvalSession.run_suite()
├── _EvalPortWorker × M (스레드)
│   └── asyncio loop — N개 RLServerEnv (TCP)
│       ├── self 에이전트 차례: compute_policy_output_batch (배치 추론)
│       └── opp 에이전트 차례: 배치 추론 또는 순차 휴리스틱
└── per-scenario 게임 집계 → _format_suite_report (bias_check.py 동일 형식)
```

### 핵심 설계 포인트

- **플레이어 ID 규칙**: `self_id='Self'` / `opp_id='Opp'` 항상 고정.  
  `p1_id = 'Self' if self_is_p1 else 'Opp'`, `p2_id = 'Opp' if self_is_p1 else 'Self'`  
  서버가 반환하는 `winner_id`·`active_player`가 'Self'/'Opp' 문자열이므로 self_id는 항상 'Self'.  
  (**주의**: 이전에 `self_is_p1=False` 시 self_id='Opp'로 잘못 구현한 버그 수정됨.)
- **8 시나리오**: 귤/샤를로테 × 선공/후공 × 같은덱/다른덱, `total_matches`를 8로 나눠 균등 분배
- **배치 분리**: self RL과 opp RL의 `compute_policy_output_batch` 호출을 따로 분리 (같은 모델이라도)
- **draw 처리**: `result='Ongoing'` + `turn > max_turns` → winner_id='' → draw로 집계

### CLI

```bash
python eval_client.py --model-path models/final_model.pt \
  --opp rule_based --multi --m-servers 4 --total-matches 400
```

주요 옵션: `--self`, `--model-path`, `--opp`, `--opp-model-path`, `--total-matches`, `--multi`, `--m-servers`, `--n-agents`, `--device`, `--seed`

---

## C# MCTS 핸들러 (구현 완료)

### 개요

새 TCP 핸들러 2개(`MCTS_RESET=3`, `MCTS_STEP=4`)로 Python이 C# 서버 내 MCTS 탐색을 직접 제어.  
CLR 크로싱·추가 소켓·pythonnet 의존성 없이 기존 M×N 병렬 구조 위에서 동작.

### C# 서버 측 구조

**`RLGameRoom`** 에 `_mctsGame` 필드 추가:

```csharp
private SeaEngine.Game?  _game;       // live game
private SeaEngine.Game?  _mctsGame;   // MCTS 탐색용 복사본

public SnapshotPacket MctsReset()
{
    var g = _game ?? throw new InvalidOperationException("...");
    _mctsGame = g.Fork();           // live game 딥카피 (Data.Clone())
    return BuildMctsSnapshot();
}

public SnapshotPacket MctsStep(string actionUidStr)
{
    var g = _mctsGame ?? throw new InvalidOperationException("...");
    g.UseAction(Uid.Parse(actionUidStr));
    return BuildMctsSnapshot();
}
```

- `Game.Fork()` = 순수 C# 딥카피 (`Data.Clone()`). Python 관여 없음
- `_mctsGame`은 connId 1개(= RLGameRoom 1개)에 종속 → connId마다 독립된 MCTS 상태
- 응답 스냅샷은 `SnapshotCodec.Instance` 사용 → 기존 `parse_snapshot`으로 동일하게 파싱
- `MctsReset`을 호출할 때마다 live game 상태로 초기화 (누적 탐색 없음, Python이 매번 fresh fork)

**핸들러 등록** (`RLServerApp.cs`):
```csharp
_manager.SetReceiveHandler(new RLMCTSResetHandler(registry, _manager));
_manager.SetReceiveHandler(new RLMCTSStepHandler(registry, _manager));
```

### Python 측 제어 흐름

`mcts_session_client.py`의 `RLServerEnv` 확장:

```python
async def mcts_reset(self) -> dict:
    self._writer.write(encode_mcts_reset())   # body 없음
    _, body = await recv_packet(self._reader)
    return parse_snapshot(body)

async def mcts_step(self, action_uid: str) -> dict:
    self._writer.write(encode_mcts_step(action_uid))
    _, body = await recv_packet(self._reader)
    return parse_snapshot(body)
```

---

## mcts_session_client.py (MCTS 병렬 평가)

`eval_client.py`의 MCTS 확장판. 동일한 TCP 연결로 C# MCTS 핸들러를 사용해 깊이 제한 순차 탐색 후 최고 가치 액션을 선택.

### `_mcts_batch_decide` 알고리즘

```
입력: envs[N], snaps[N], self_ids[N], opp_ids[N]
      top_k, depth (MCTSConfig)

1. policy pass  — 현재 상태 전체 배치 추론 → 폴백 액션 + 초기 후보 순위
2. candidates[i] = snaps[i]['actions'][:top_k]   # 상위 k개 후보

for k = 0 to max_k-1:           # 후보 인덱스별 순차 탐색
    has_k = [i | len(candidates[i]) > k]

    Phase 1 — MCTS 포크 (병렬)
        await gather(*[envs[i].mcts_reset() for i in has_k])
        → 서버: live game → _mctsGame 딥카피

    Phase 2 — k번째 후보 적용 (병렬)
        d1_snaps = await gather(*[envs[i].mcts_step(candidates[i][k]['uid']) for i in has_k])

    Phase 3 — depth-1 추가 스텝 (병렬, 반복)
        for d in range(1, depth):
            ongoing = [i | curr[i] still Ongoing and has actions]
            batch RL inference → 양측 모두 same model로 그리디 선택
            new_snaps = await gather(*[envs[i].mcts_step(action_uid) for i in ongoing])

    Phase 4 — 리프 가치 평가 (병렬)
        terminal: +1.0 (self win), -1.0 (opp win), 0.0 (draw/timeout)
        ongoing:  batch RL inference → out.value (critic V(s))
        cand_values[i][k] = leaf_value

최종: argmax over k per agent → 최고 가치 후보 uid
      후보 없음(빈 액션 등): policy 폴백
```

### 배치 효율성

- **TCP 왕복**: M×N 에이전트가 동일 `k` 인덱스를 `asyncio.gather`로 동시 탐색 → top_k × depth 라운드
- **RL 추론**: 한 라운드의 모든 에이전트 상태를 하나의 `compute_policy_output_batch` 호출로 처리
- `depth=2, top_k=5, N=32, M=4` 기준: 5 × (1+1) = 10 라운드 TCP + 10 배치 추론

### CLI

```bash
python mcts_session_client.py --model-path models/final_model.pt \
  --opp rule_based --depth 2 --top-k 5 \
  --multi --m-servers 4 --total-matches 400
```

주요 추가 옵션: `--depth` (기본 2), `--top-k` (기본 5). 나머지는 `eval_client.py`와 동일.

---

## MCTS 적용 분석 결과

### BeliefMCTS 현황

`SeaEngineBeliefMCTSAgent` (agents/seaengine_agents.py)는 다음 조건이 모두 충족될 때만 MCTS 실행:
- `snapshot["_engine_game"]` 존재 (C# Game 객체, pythonnet 전용)
- 없으면 **자동으로 policy-only로 fallback** (경고 없음)

`_engine_game`은 오직 `training/evaluator.py:334`의 `_attach_engine_state_if_needed()`에서만 주입됨.  
TCP 경로에서는 절대 설정되지 않으므로 **TCP + BeliefMCTS 조합 = 항상 policy fallback**.

### BeliefMCTS + TCP 직접 연동 시 문제점

| 병목 | 규모 |
|---|---|
| `_build_snapshot()` CLR 크로싱 | ~161회/call × 9 calls/decision × 128 games = **186,000회/배치** |
| GIL 직렬화 | pythonnet CLR 호출이 GIL 보유 → M 스레드 병렬 불가 |
| 배치 추론 파괴 | `BeliefMCTS.select_action` override → N개 개별 호출로 분산 |
| 추가 소켓 필요 | Clone당 소켓 → M×N 한계(256) 즉시 초과 |

해결: **C# 서버에 MCTS 핸들러 추가** (위 "C# MCTS 핸들러" 섹션 참조). Python은 기존 소켓 그대로 사용하고, 서버가 game fork를 관리.

### 2단계 학습 파이프라인 (확정)

```
Stage 1 (현행)                     Stage 2 (미구현)
TCP 서버 M×N=128                   pythonnet 로컬 백엔드
PPO + curriculum                   BeliefMCTS + PPO refinement
~3.61 ep/s                         ~0.4~0.8 ep/s
                    │
    엔트로피 붕괴 또는 승률 정체 감지
    (update_from_buffer 반환 dict의 entropy 필드)
                    ↓
        Stage 1 체크포인트 로드
        VectorSeaEngineEnv (pythonnet)
        _attach_engine_state_if_needed() 패턴 적용
        BeliefMCTS top_k=2-3, sims=1, rollout=1-2
```

Stage 2는 아직 미구현. 구현 시 참조 파일:
- `training/evaluator.py:329-341` (`_attach_engine_state_if_needed` 패턴)
- `SeaEngine/bridge/pythonnet_session.py` (`fork_game`, `_build_snapshot`)
- `agents/seaengine_agents.py` (`SeaEngineBeliefMCTSAgent`, `_simulate_candidate`)

---

## Self-play 에이전트 로드 패턴

```python
import torch
from RL_AI.agents import SeaEngineRLAgent, infer_hidden_dim_from_state_dict, load_state_dict_flexible
from RL_AI.SeaEngine.observation import STATE_VECTOR_DIM

state_dict = torch.load('models/ckpt_ep2000.pt', map_location='cpu', weights_only=True)
hidden_dim = infer_hidden_dim_from_state_dict(state_dict)
agent = SeaEngineRLAgent(hidden_dim=hidden_dim, device='cpu', sample_actions=True)
agent.ensure_model(STATE_VECTOR_DIM)
load_state_dict_flexible(agent.model, state_dict)
agent.model.eval()
agent.name = 'self_ep_2000'
```

---

## CLI 사용법

```powershell
# 빌드
dotnet build Game.RL_Server/Game.RL_Server.csproj -c Release

# 서버 4개 시작 (포트 9000~9003)
$servers = 0..3 | ForEach-Object {
    Start-Process dotnet -ArgumentList "run --project Game.RL_Server -c Release -- $((9000+$_))" -NoNewWindow -PassThru
}

# 학습 (M=4, N=32, CUDA)
python train_client.py --multi --m-servers 4 --n-agents 32 --base-port 9000 --total-episodes 10000 --device cuda

# 평가 포함
python train_client.py --multi --m-servers 4 --n-agents 32 --total-episodes 10000 --save-interval 2000 --n-eval 64 --device cuda

# 서버 종료
$servers | Stop-Process -Force
```

### 주요 옵션

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--multi` | false | M개 서버 모드 |
| `--m-servers` | 4 | 서버 수 M |
| `--n-agents` | 32 | 서버당 에이전트 수 N |
| `--base-port` | 9000 | Multi 모드 시작 포트 |
| `--total-episodes` | 2000 | 총 에피소드 수 |
| `--save-interval` | 500 | 체크포인트 저장 주기 |
| `--eval-interval` | 0 | 평가 전용 주기 (0=비활성) |
| `--n-eval` | 64 | 평가당 상대별 에피소드 수 |
| `--device` | cpu | cuda / cpu |
| `--max-turns` | 100 | 에피소드당 최대 턴 수 (초과 시 강제 종료) |
| `--seed` | None | 랜덤 시드 |

---

## 출력 파일

| 파일 | 내용 |
|---|---|
| `log/train_YYYYMMDD_HHMMSS.log` | 전체 stdout 로그 |
| `log/train_summary.txt` | 학습 결과 요약 (before/train/checkpoint/after) |
| `log/train_ckpt_{ep}_{ts}.txt` | 체크포인트별 평가 결과 |
| `models/ckpt_ep{N}.pt` | 중간 체크포인트 |
| `models/final_model.pt` | 최종 모델 |

---

## 주의사항

- **RTX 50 시리즈 (Blackwell, CC 12.0)**: `pip install torch --index-url https://download.pytorch.org/whl/cu128` 필요 (cu126 아님); conda `rl_ai` 환경 사용 권장
- **RTX 5070 + M=8 BSOD**: 8개 스레드가 동시에 CUDA 커널을 실행하면 GPU TDR(Timeout Detection and Recovery) 발동 → Windows BSOD (Kernel-Power 41, BugCheck 1001). `--device cpu` 사용 권장. M=4 이하면 GPU 가능
- **GPU vs CPU 추론 속도 (batch=32)**: GPU 6.67ms, CPU 10.88ms (1.63× 느림). 그러나 M=8+CPU는 M=4+GPU 대비 전체 학습 속도가 약 40~60% 빠름 (BSOD 없이 2배 많은 동시 게임)
- **Python 패키지 경로**: `RL_AI.*` 임포트는 `_ensure_importable()` 이후에만 가능. `train_client.py` 밖에서 직접 임포트할 경우 `sys.path`에 `c:\Users\addbum421` 수동 추가 필요
- **self-play 에이전트는 CPU에 로드**: 학습 에이전트(CUDA)와 별도 디바이스로 격리해 메모리 충돌 방지
- **스케줄 인덱싱**: `schedule[it * total_envs + worker_idx * n_agents + slot]` 로 에피소드별 상대 결정
- **`_PortWorker.submit` opponents 인자**: 단일 에이전트를 넘기면 N개 슬롯 모두 동일 상대, 리스트를 넘기면 슬롯별 상대 지정

---

## 확장 설계: 프로토콜 멀티플렉싱 (미구현)

현재 아키텍처의 하드 한계: **M×N < 256** (총 에이전트 수 256 이상 → Windows LAN 스택 크래시).  
원인: 에이전트 1개 = TCP 연결 1개 구조에서 연결 수가 OS 소켓 처리 한계를 초과.

### 설계 방향

패킷 헤더의 `reserved(i32)` 필드를 **슬롯 ID**로 전용하여, 연결 1개로 N개 게임을 멀티플렉싱:

```
현재:  ConnId → 게임 1개 (연결 M×N개)
확장:  ConnId → 워커 1개 (연결 M개)
              └─ slotId(reserved) → 게임 N개 (워커 내 dispatch)
```

헤더 재정의 (변경 없이 의미만 재해석):
```
flag(u32) | handlerId(i32) | queryNum(i32) | reserved(i32) → slotId(i32)
```

### C# 변경 포인트

- `RLHandler.OnReceive(connId, data)` → 헤더에서 `slotId` 추출
- `GameRegistry`: `ConnId → RLGameRoom` → `(ConnId, slotId) → RLGameRoom`
- `RLGameRoom` N개를 연결 1개로 관리

### Python 변경 포인트

- `_PortWorker`: 연결 N개 → 연결 1개, 패킷마다 슬롯 인덱스 포함
- `encode_init` / `encode_step`: `slotId` 파라미터 추가
- `recv_packet`: 수신 패킷을 `slotId` 기준으로 라우팅

### 기대 효과

| | 현재 | 멀티플렉싱 후 |
|---|---|---|
| TCP 연결 수 | M×N | M |
| M=4, N=32 | 128 연결 | 4 연결 |
| 스케일 한계 | M×N < 256 | 제거 (연결 수 무관) |

---

## 이전 세션에서 해결된 문제들

| 문제 | 원인 | 해결 |
|---|---|---|
| State vector 차원 불일치 | C# 476-dim vs Python 771-dim | `RlObservationExporter.cs` 수정 (41-dim board, 20-dim hand, 72-dim action) |
| Card data not found | 기본 경로 오류 | `<binDir>/db/Cards.csv` 로 변경, csproj에 PreserveNewest 추가 |
| PowerShell Ctrl-C로 서버 안 죽음 | 프로세스 핸들 미보관 | `Start-Process -PassThru` 로 `$servers` 저장, `$servers \| Stop-Process -Force` |
| 여러 모델이 각자 학습됨 | M개 프로세스 구조 | 단일 Python 프로세스 + M 스레드로 전환, trainer 공유 |
| 학습 추론 병목 (N 순차 호출) | 에피소드별 개별 forward pass | `_collect` 재설계: 스텝 동기화 + `compute_policy_output_batch` 1회 호출 |
| RTX 5070 + M=8 BSOD | GPU TDR: 8 CUDA 스레드 동시 실행 | `--device cpu` 사용 |
| M=8이 M=4보다 3배 느림 (Ultra 7) | E코어 밀림 + PPO 동기화 장벽 | M=4, N=32가 실질 최적 (3.61 ep/s vs 1.21 ep/s 실측) |
| M×N=256 이상 LAN 크래시 | Windows TCP 스택 소켓 한계 | M×N < 256 유지. 확장은 프로토콜 멀티플렉싱 필요 |
| C# 고CPU 점유 (tick loop) | `Thread.SpinWait(10)` 사실상 busy-spin | `SpinWait.SpinOnce(sleep1Threshold:-1)` 적응형 대기로 교체 |
| C# ArrayPool 도입 후 `IncompleteReadError` | `TrySend(byte[], int)` 오버로드가 비-풀 버퍼도 풀에 반환 | 변경 전체 되돌림 (Connection.cs, NetEvent.cs, NetStreamManager.cs, NetConnectionManager.cs) |
| `eval_client.py` self_id/opp_id 버그 | `self_is_p1=False` 시 `self_id='Opp'`로 잘못 설정 → 상대 턴에 self 추론, self 승리를 opp 승리로 기록 | `self_id='Self'` / `opp_id='Opp'` 항상 고정 (플레이어 라벨 'Self'는 항상 self 플레이어에게 할당) |
| `eval_client.py` 생성 | bias_check/make_balance를 TCP로 실행할 방법 없음 | `eval_client.py` 구현: M×N 병렬, 동일 로그 형식, 20~50배 속도 향상 |
| TCP + MCTS 연동 불가 | CLR 크로싱 병목, GIL 직렬화, 추가 소켓 한계 | C# 서버에 `MCTS_RESET(3)` / `MCTS_STEP(4)` 핸들러 추가: `RLGameRoom._mctsGame`에 `Game.Fork()` 딥카피, Python이 기존 소켓으로 탐색 제어 |
| `mcts_session_client.py` 생성 | TCP + MCTS 평가 도구 없음 | `_mcts_batch_decide`: top_k 후보 × depth 깊이 탐색, M×N 병렬 gather + 배치 리프 추론 |
