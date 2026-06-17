# RL_AI TCP Training Pipeline

SeaEngine 게임을 C# TCP 서버(`Game.RL_Server`)로 실행하고, Python `train_client.py`가 여러 서버와 여러 agent connection을 붙여 PPO 학습을 수행한다.

현재 기본값은 로컬 노트북 기준 공격적 설정이다.

| 항목 | 기본값 |
|---|---:|
| 서버 수 `M` | `2` |
| 서버당 agent 수 `N` | `64` |
| 총 동시 게임 | `128` |
| Stage 1 episode | `10000` |
| Stage 2 episode | `2500` 고정 |
| Stage 2 동시 env | `32` |
| 평가 수 `n_eval` | `50` per layout, 상대별 총 `400`판 |
| reward mode | `terminal_action` |

`start.py`/PythonNet 기존 경로는 legacy로 유지한다. 새 기본 경로는 `train_client.py` + `Game.RL_Server`이다.

---

## 핵심 기능

- C# TCP `Game.RL_Server`
- M개 서버 x N개 agent 병렬 학습
- binary snapshot protocol
- server-side opponent 자동 처리
- C# `Game.ResetAndInit` 기반 game 재사용
- `RlObservationExporter` 771/92 feature shape
- checkpoint 저장 및 `eval_client.py` 평가
- Stage 2 TCP server MCTS refinement using `MCTS_RESET`/`MCTS_STEP`
- TCP MCTS handler:
  - `HANDLER_MCTS_RESET = 3`
  - `HANDLER_MCTS_STEP = 4`

---

## Windows 처음부터 실행

`C:\Users\user\RL_AI.zip`이 있는 상태에서 PowerShell로 실행한다.

```powershell
cd $env:USERPROFILE

if (Test-Path .\RL_AI) {
    Remove-Item -Recurse -Force .\RL_AI
}

Expand-Archive -Force .\RL_AI.zip .

if (Test-Path .\RL_AI\RL_AI) {
    if (Test-Path .\RL_AI_flatten_tmp) {
        Remove-Item -Recurse -Force .\RL_AI_flatten_tmp
    }
    Move-Item .\RL_AI .\RL_AI_flatten_tmp
    Move-Item .\RL_AI_flatten_tmp\RL_AI .\RL_AI
    Remove-Item -Recurse -Force .\RL_AI_flatten_tmp
}

cd $env:USERPROFILE\RL_AI

$env:DOTNET_ROOT = "C:\Program Files\dotnet"
$env:PATH = "$env:DOTNET_ROOT;$env:PATH"

dotnet build .\Game.RL_Server\Game.RL_Server.csproj -c Release

9000..9003 | ForEach-Object {
    $c = Get-NetTCPConnection -LocalPort $_ -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($c) { Stop-Process -Id $c.OwningProcess -Force }
}

$servers = 0..1 | ForEach-Object {
    $port = 9000 + $_
    Start-Process dotnet `
      -ArgumentList "run --project Game.RL_Server -c Release -- $port $env:USERPROFILE\RL_AI\cards\Cards.csv" `
      -RedirectStandardOutput "$env:USERPROFILE\rl_server_$port.log" `
      -RedirectStandardError "$env:USERPROFILE\rl_server_$port.err.log" `
      -PassThru
}

Start-Sleep -Seconds 5

py -X utf8 -u .\train_client.py `
  --multi `
  --m-servers 2 `
  --n-agents 64 `
  --base-port 9000 `
  --total-episodes 10000 `
  --save-interval 2500 `
  --n-eval 50 `
  --device cpu `
  --reward-mode terminal_action
```

서버 종료:

```powershell
$servers | Stop-Process -Force
```

변수가 사라졌으면 포트로 종료:

```powershell
9000..9003 | ForEach-Object {
    $c = Get-NetTCPConnection -LocalPort $_ -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($c) { Stop-Process -Id $c.OwningProcess -Force }
}
```

---

## Stage 1 + Stage 2 실행

Stage 1은 TCP 서버 대규모 병렬 PPO 학습이고, Stage 2는 같은 `Game.RL_Server`의 `MCTS_RESET`/`MCTS_STEP` scratch copy로 2500판 정제한다.

```powershell
cd $env:USERPROFILE\RL_AI

py -X utf8 -u .\train_client.py `
  --multi `
  --m-servers 2 `
  --n-agents 64 `
  --base-port 9000 `
  --total-episodes 10000 `
  --save-interval 2500 `
  --n-eval 50 `
  --device cpu `
  --reward-mode terminal_action `
  --early-stage2 `
  --stage2-num-envs 32 `
  --stage2-mcts-depth 1 `
  --stage2-mcts-top-k 2 `
  --stage2-output-path models\final_model_stage2.pt `
  --stage2-card-data-path cards\Cards.csv
```

Stage 2 episode 수는 코드에서 `2500`으로 고정된다. `--stage2-episodes` 값을 줘도 무시된다.

Stage 2만 바로 테스트:

```powershell
cd $env:USERPROFILE\RL_AI

py -X utf8 -u .\train_client.py `
  --stage2-now `
  --multi `
  --m-servers 2 `
  --base-port 9000 `
  --stage2-model-path models\final_model.pt `
  --stage2-output-path models\final_model_stage2.pt `
  --stage2-num-envs 32 `
  --stage2-mcts-depth 1 `
  --stage2-mcts-top-k 2 `
  --stage2-card-data-path cards\Cards.csv `
  --device cpu
```

정체 시 Stage 2로 조기 전환:

```powershell
py -X utf8 -u .\train_client.py `
  --multi `
  --m-servers 2 `
  --n-agents 64 `
  --base-port 9000 `
  --total-episodes 10000 `
  --save-interval 2500 `
  --n-eval 50 `
  --device cpu `
  --reward-mode terminal_action `
  --early-stage2 `
  --stage2-mcts-depth 1 `
  --stage2-mcts-top-k 2 `
  --early-stage2-min-episode 2500 `
  --early-stage2-patience 2 `
  --early-stage2-min-delta 0.01
```

---

## 짧은 테스트 명령

Stage 1 200판만 빠르게 확인한다. 실제 수집은 `M*N` wave 단위라 200을 조금 넘길 수 있다.

```powershell
cd $env:USERPROFILE\RL_AI

py -X utf8 -u .\train_client.py `
  --multi `
  --m-servers 2 `
  --n-agents 64 `
  --base-port 9000 `
  --total-episodes 10000 `
  --save-interval 2500 `
  --n-eval 50 `
  --device cpu `
  --reward-mode terminal_action
```

Stage 1 200판 후 Stage 2 2500판:

```powershell
cd $env:USERPROFILE\RL_AI

py -X utf8 -u .\train_client.py `
  --multi `
  --m-servers 2 `
  --n-agents 64 `
  --base-port 9000 `
  --total-episodes 10000 `
  --save-interval 2500 `
  --n-eval 50 `
  --device cpu `
  --reward-mode terminal_action `
  --early-stage2 `
  --stage2-num-envs 32 `
  --stage2-mcts-depth 1 `
  --stage2-mcts-top-k 2
```

초반 500판 loss를 최대한 자주 보고 CSV 그래프용 로그를 남기려면:

```powershell
$env:SEAENGINE_EARLY_LOSS_EPISODES = "500"
$env:SEAENGINE_EARLY_LOSS_EVERY_UPDATE = "1"
$env:SEAENGINE_EARLY_LOSS_CSV = "1"

py -X utf8 -u .\train_client.py `
  --multi `
  --m-servers 2 `
  --n-agents 64 `
  --base-port 9000 `
  --total-episodes 10000 `
  --save-interval 2500 `
  --n-eval 50 `
  --device cpu `
  --reward-mode terminal_action `
  --stage2-num-envs 32 `
  --stage2-mcts-depth 1 `
  --stage2-mcts-top-k 2
```

CSV는 `log/early_loss_*.csv`에 저장된다.

---

## Linux / DLPC 기본 실행

```bash
cd ~/RL_AI

dotnet build Game.RL_Server/Game.RL_Server.csproj -c Release

for i in 0 1; do
  dotnet run --project Game.RL_Server -c Release -- $((9000+i)) ~/RL_AI/cards/Cards.csv \
    > ~/rl_server_$((9000+i)).log 2>&1 &
done

sleep 5

python -u train_client.py \
  --multi \
  --m-servers 2 \
  --n-agents 64 \
  --base-port 9000 \
  --total-episodes 10000 \
  --save-interval 2500 \
  --n-eval 50 \
  --device cpu \
  --reward-mode terminal_action
```

DLPC에서 CUDA 환경이 제대로 잡혀 있으면 `--device cuda`로 바꿀 수 있다. 로컬 Radeon 내장 GPU 환경에서는 보통 PyTorch CUDA가 없으므로 `cpu`가 안전하다.

---

## eval_client.py 평가

서버가 켜져 있는 상태에서 checkpoint만 평가한다.

```powershell
cd $env:USERPROFILE\RL_AI

py -X utf8 -u .\eval_client.py `
  --multi `
  --m-servers 2 `
  --n-agents 64 `
  --base-port 9000 `
  --model-path models\final_model.pt `
  --n-eval 256 `
  --device cpu
```

---

## TCP Protocol

모든 패킷은 다음 형식이다.

```text
[u32 length][16-byte header][body]
header = flag(u32) | handlerId(i32) | queryNum(i32) | reserved(i32)
```

| Handler | ID | 방향 | Body |
|---|---:|---|---|
| INIT | `1` | Python -> C# | `p1Deck`, `p2Deck`, `p1Id`, `p2Id`, optional `opponentName`, optional `autoOpponent` |
| STEP | `2` | Python -> C# | `action_uid` |
| MCTS_RESET | `3` | Python -> C# | empty |
| MCTS_STEP | `4` | Python -> C# | `action_uid` |
| Snapshot | same as request | C# -> Python | binary snapshot |

`MCTS_RESET`는 live game을 `Fork()`해서 `_mctsGame` scratch copy를 만든다. `MCTS_STEP`은 이 copy에만 action을 적용하므로 실제 학습 game은 변하지 않는다.

---

## Scaling Notes

현재 구조는 TCP connection 수가 `M*N`이다.

| 설정 | 연결 수 | 판단 |
|---|---:|---|
| `M=2,N=32` | 64 | 안정 |
| `M=2,N=64` | 128 | 현재 로컬 기본값 |
| `M=4,N=64` | 256 | Windows에서는 위험 |
| `M*N > 256` | 256 초과 | 비추천 |

더 크게 확장하려면 header의 `reserved` 필드를 `slotId`로 쓰는 multiplexing이 필요하다.

```text
현재: ConnId -> one game
확장: ConnId + slotId -> one game
```

이 multiplexing은 아직 미구현이다.

---

## Troubleshooting

| 증상 | 원인 | 해결 |
|---|---|---|
| `ConnectionRefusedError` | 서버 미실행 또는 빌드 실패 | 서버 로그와 포트 확인 |
| `Card data not found` | Cards.csv 경로 오류 | 서버 실행 시 `cards\Cards.csv` 절대경로 지정 |
| `state_vector dim mismatch` | C# exporter와 Python feature spec 불일치 | `RlObservationExporter` 771/92 build 확인 |
| Stage 2 `Cards.csv` 없음 | 상대경로 기준 꼬임 | `--stage2-card-data-path cards\Cards.csv` 또는 절대경로 사용 |
| Stage 2 `approx_kl` 과대 | MCTS-PPO가 너무 aggressive | 현재 Stage 2 config는 보수적 LR/target_kl 사용 |
| `M*N` 256 근처에서 불안정 | TCP connection 과다 | `M=2,N=64` 이하부터 사용 |

---

## Charlotte Ablation

샤를로테 vs 샤를로테만 고정해서 데이터 다양성 변수를 줄이는 실험용 스크립트:

```powershell
cd $env:USERPROFILE\RL_AI

py -X utf8 -u .\charlotte_ablation.py `
  --mode both `
  --train-episodes 10000 `
  --eval-matches 128 `
  --num-envs 16 `
  --update-interval 16 `
  --save-interval 0 `
  --device auto `
  --card-data-path cards\Cards.csv `
  --train-opponents random greedy rule_based
```

`--save-interval 0`이면 중간 self-play checkpoint를 끄고, 덱 변수만 줄인 clean ablation으로 돌린다.
