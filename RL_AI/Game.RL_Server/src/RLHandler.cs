using Game.Network;
using SeaEngine.CardManager;

namespace Game.RL_Server;

/// <summary>
/// ConnId → RLGameRoom 매핑 레지스트리.
/// Tick 루프(단일 스레드)에서만 접근되므로 잠금 불필요.
/// </summary>
public sealed class GameRegistry
{
    private readonly Dictionary<ConnId, RLGameRoom> _rooms = new();
    private readonly CardLoader _cardLoader;

    public GameRegistry(CardLoader cardLoader)
    {
        _cardLoader = cardLoader;
    }

    public RLGameRoom GetOrCreate(ConnId id)
    {
        if (!_rooms.TryGetValue(id, out var room))
        {
            room        = new RLGameRoom(_cardLoader);
            _rooms[id]  = room;
        }
        return room;
    }

    public void Remove(ConnId id) => _rooms.Remove(id);
}

// ---------------------------------------------------------------------------
// HANDLER_INIT (1): Python → C# 게임 초기화 요청
//   body: WriteString(p1Deck) WriteString(p2Deck) WriteString(p1Id) WriteString(p2Id)
//         i32(aiSide) i32(opponentType) i32(seed)
//   aiSide: 0=P1 AI, 1=P2 AI, -1=both external
//   opponentType: 0=external, 1=random, 2=greedy, 3=rule_based
//   seed: >=0 deterministic, <0 random
// ---------------------------------------------------------------------------
public sealed class RLInitHandler : INetReceiveEventHandler
{
    public int HandlerId => RLProto.HANDLER_INIT;

    private readonly GameRegistry _registry;
    private readonly INetAPI      _net;

    public RLInitHandler(GameRegistry registry, INetAPI net)
    {
        _registry = registry;
        _net      = net;
    }

    public void OnReceive(ConnId connId, byte[] raw)
    {
        try
        {
            var reader      = new PacketReader(raw.AsSpan());
            var p1Deck      = reader.ReadString();
            var p2Deck      = reader.ReadString();
            var p1Id        = reader.ReadString();
            var p2Id        = reader.ReadString();
            var aiSide      = reader.ReadInt32();
            var opponentType = reader.ReadInt32();
            var seed        = reader.ReadInt32();

            var snap = _registry.GetOrCreate(connId).Init(p1Deck, p2Deck, p1Id, p2Id, aiSide, opponentType, seed);
            _net.SendMessage(RLProto.HANDLER_INIT, connId, snap, SnapshotCodec.Instance);
        }
        catch (Exception ex)
        {
            Log.WriteLog($"[InitHandler] {connId}: {ex.Message}");
        }
    }
}

// ---------------------------------------------------------------------------
// HANDLER_STEP (2): Python → C# 액션 전달 요청
//   body: WriteString(action_uid)   e.g. "A001"
// ---------------------------------------------------------------------------
public sealed class RLStepHandler : INetReceiveEventHandler
{
    public int HandlerId => RLProto.HANDLER_STEP;

    private readonly GameRegistry _registry;
    private readonly INetAPI      _net;

    public RLStepHandler(GameRegistry registry, INetAPI net)
    {
        _registry = registry;
        _net      = net;
    }

    public void OnReceive(ConnId connId, byte[] raw)
    {
        try
        {
            var reader    = new PacketReader(raw.AsSpan());
            var actionUid = reader.ReadString();

            var snap = _registry.GetOrCreate(connId).Step(actionUid);
            _net.SendMessage(RLProto.HANDLER_STEP, connId, snap, SnapshotCodec.Instance);
        }
        catch (Exception ex)
        {
            Log.WriteLog($"[StepHandler] {connId}: {ex.Message}");
        }
    }
}

// ---------------------------------------------------------------------------
// HANDLER_MCTS_RESET (3): live game → MCTS 복사본으로 포크
//   body: 없음
//   응답: 포크 직후 스냅샷 (live game과 동일 상태, 이후 MctsStep으로 탐색)
// ---------------------------------------------------------------------------
public sealed class RLMCTSResetHandler : INetReceiveEventHandler
{
    public int HandlerId => RLProto.HANDLER_MCTS_RESET;

    private readonly GameRegistry _registry;
    private readonly INetAPI      _net;

    public RLMCTSResetHandler(GameRegistry registry, INetAPI net)
    {
        _registry = registry;
        _net      = net;
    }

    public void OnReceive(ConnId connId, byte[] raw)
    {
        try
        {
            var snap = _registry.GetOrCreate(connId).MctsReset();
            _net.SendMessage(RLProto.HANDLER_MCTS_RESET, connId, snap, SnapshotCodec.Instance);
        }
        catch (Exception ex)
        {
            Log.WriteLog($"[MCTSResetHandler] {connId}: {ex.Message}");
        }
    }
}

// ---------------------------------------------------------------------------
// HANDLER_MCTS_STEP (4): MCTS 복사본에 액션 적용
//   body: WriteString(action_uid) — HANDLER_STEP과 완전히 동일한 형식
//   응답: 스냅샷 — parse_snapshot으로 동일하게 파싱 가능
// ---------------------------------------------------------------------------
public sealed class RLMCTSStepHandler : INetReceiveEventHandler
{
    public int HandlerId => RLProto.HANDLER_MCTS_STEP;

    private readonly GameRegistry _registry;
    private readonly INetAPI      _net;

    public RLMCTSStepHandler(GameRegistry registry, INetAPI net)
    {
        _registry = registry;
        _net      = net;
    }

    public void OnReceive(ConnId connId, byte[] raw)
    {
        try
        {
            var reader    = new PacketReader(raw.AsSpan());
            var actionUid = reader.ReadString();

            var snap = _registry.GetOrCreate(connId).MctsStep(actionUid);
            _net.SendMessage(RLProto.HANDLER_MCTS_STEP, connId, snap, SnapshotCodec.Instance);
        }
        catch (Exception ex)
        {
            Log.WriteLog($"[MCTSStepHandler] {connId}: {ex.Message}");
        }
    }
}

// ---------------------------------------------------------------------------
// HANDLER_MCTS_BATCH (5): live game의 현재 상태에서 후보 액션 여러 개를 한 번에 fork 평가
//   body: i32(count) + count × i32(action_index)
//   응답: MctsBatchResult (leaf 정보 여러 개)
// ---------------------------------------------------------------------------
public sealed class RLMCTSBatchHandler : INetReceiveEventHandler
{
    public int HandlerId => RLProto.HANDLER_MCTS_BATCH;

    private readonly GameRegistry _registry;
    private readonly INetAPI      _net;

    public RLMCTSBatchHandler(GameRegistry registry, INetAPI net)
    {
        _registry = registry;
        _net      = net;
    }

    public void OnReceive(ConnId connId, byte[] raw)
    {
        try
        {
            var reader = new PacketReader(raw.AsSpan());
            var count  = reader.ReadInt32();
            if (count < 0)
                throw new InvalidOperationException($"Invalid candidate count: {count}");

            var indices = new int[count];
            for (var i = 0; i < count; i++)
                indices[i] = reader.ReadInt32();

            var frames = _registry.GetOrCreate(connId).MctsBatchStep(indices);
            _net.SendMessage(RLProto.HANDLER_MCTS_BATCH, connId, new MctsBatchResult(frames), MctsBatchResultCodec.Instance);
        }
        catch (Exception ex)
        {
            Log.WriteLog($"[MCTSBatchHandler] {connId}: {ex.Message}");
        }
    }
}

// ---------------------------------------------------------------------------
// Control handler: 연결/해제 이벤트 처리
// ---------------------------------------------------------------------------
public sealed class RLControlHandler : INetControlEventHandler
{
    private readonly GameRegistry _registry;

    public RLControlHandler(GameRegistry registry)
    {
        _registry = registry;
    }

    public void OnHello(ConnId connId, byte[] raw)
        => Log.WriteLog($"[Control] Connected: {connId}");

    public void OnDisconnect(ConnId connId, byte[] raw)
    {
        Log.WriteLog($"[Control] Disconnected: {connId}");
        _registry.Remove(connId);
    }

    public void OnException(ConnId connId, byte[] raw, string msg)
    {
        Log.WriteLog($"[Control] Exception {connId}: {msg}");
        _registry.Remove(connId);
    }
}
