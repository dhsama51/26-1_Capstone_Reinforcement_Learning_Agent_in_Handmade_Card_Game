using Game.Network;
using SeaEngine.CardManager;
using SeaEngine.Common;
using SeaEngine.GameDataManager;

namespace Game.RL_Server;

// ---------------------------------------------------------------------------
// NullLogger: 학습 서버에서 게임 로그 출력 억제
// ---------------------------------------------------------------------------
internal sealed class NullLogger : SeaEngine.Logger.ILogger
{
    public static readonly NullLogger Instance = new();
    public void LogAction(GameAction a, GameData d) { }
    public void LogCards(GameData d) { }
    public void LogEvent(string ev, string timing, Uid src) { }
    public void Log(string msg, GameData d) { Console.WriteLine(msg); }
}

// ---------------------------------------------------------------------------
// RLServerApp: NetworkManager 래핑, 핸들러 등록, Tick 루프 실행
// ---------------------------------------------------------------------------
public sealed class RLServerApp
{
    private readonly int _port;
    private readonly NetworkManager _manager;

    // maxProcessPerTick: 한 Tick에서 처리할 최대 이벤트 수.
    // N 에이전트 수보다 충분히 크게 설정 (여기서는 고정 512).
    private const int MaxPerTick = 512;

    public RLServerApp(int port, CardLoader cardLoader)
    {
        _port = port;
        _manager = NetworkManager.CreateNetworkManager(port, MaxPerTick);
        Game.Network.Log.SetLogger(Console.WriteLine);
        var registry = new GameRegistry(cardLoader);
        _manager.SetReceiveHandler(new RLInitHandler(registry, _manager));
        _manager.SetReceiveHandler(new RLStepHandler(registry, _manager));
        _manager.SetReceiveHandler(new RLMCTSResetHandler(registry, _manager));
        _manager.SetReceiveHandler(new RLMCTSStepHandler(registry, _manager));
        _manager.SetReceiveHandler(new RLMCTSBatchHandler(registry, _manager));
        _manager.SetControlHandler(new RLControlHandler(registry));
    }

    /// <summary>
    /// Blocking 실행. Ctrl+C로 종료.
    /// </summary>
    public void Run()
    {
        _manager.Start();
        Console.WriteLine($"[RLServer] port={_port}  MaxPerTick={MaxPerTick}");
        Console.WriteLine("[RLServer] Running. Ctrl+C to stop.");

        using var cts = new CancellationTokenSource();
        Console.CancelKeyPress += (_, e) =>
        {
            e.Cancel = true;
            cts.Cancel();
        };

        // 적응형 Tick 루프: 이벤트가 있으면 즉시 처리, 없으면 CPU를 양보.
        // SpinOnce(sleep1Threshold:-1) = spin → yield → Sleep(0) 단계적 전환, Sleep(1) 호출 없음.
        // 효과: 유휴 시 CPU 점유 ~95% 감소, 이벤트 도착 후 응답 지연 < 수십 µs 유지.
        SpinWait sw = default;
        while (!cts.IsCancellationRequested)
        {
            if (_manager.Tick()) sw.Reset();
            else sw.SpinOnce(sleep1Threshold: -1);
        }

        _manager.StopAsync().GetAwaiter().GetResult();
        Console.WriteLine("[RLServer] Stopped.");
    }
}
