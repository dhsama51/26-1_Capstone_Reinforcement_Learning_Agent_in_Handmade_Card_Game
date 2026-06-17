using System.Linq;
using SeaEngine.CardManager;
using SeaEngine.Common;
using SeaEngine.GameDataManager.Components;
using SeaEngine.GameEffectManager;
using SeaEngine.RL;

namespace Game.RL_Server;

public enum OpponentType { External = 0, Random = 1, Greedy = 2, RuleBased = 3 }

/// <summary>
/// ConnId 1개에 대응하는 게임 슬롯.
/// opponentType > 0 이면 C# 서버가 상대 턴을 자동 처리.
/// </summary>
public sealed class RLGameRoom
{
    private readonly CardLoader  _cardLoader;
    private SeaEngine.Game?      _game;
    private SeaEngine.Game?      _mctsGame;
    private string _p1Id = "";
    private string _p2Id = "";
    private string _aiId = "";
    private OpponentType _opponentType = OpponentType.External;
    private Random _rng = new();
    private const int MaxAutoOpponentSteps = 256;

    public string P1Id => _p1Id;
    public string P2Id => _p2Id;

    public RLGameRoom(CardLoader cardLoader)
    {
        _cardLoader = cardLoader;
    }

    /// <summary>
    /// 새 게임 초기화. 초기 스냅샷 반환.
    /// aiSide: 0=P1이 AI, 1=P2가 AI, -1=양쪽 외부(기존 동작).
    /// opponentType: 0=external, 1=random, 2=greedy, 3=rule_based.
    /// seed: 0 이상이면 해당 seed로 Random 초기화, 음수면 무작위.
    /// </summary>
    public RlObservationFrame Init(
        string p1Deck, string p2Deck, string p1Id, string p2Id,
        int aiSide = -1, int opponentType = 0, int seed = -1)
    {
        _p1Id = p1Id;
        _p2Id = p2Id;
        _opponentType = (OpponentType)Math.Clamp(opponentType, 0, 3);
        _rng = seed >= 0 ? new Random(seed) : new Random();

        _aiId = aiSide switch
        {
            0 => p1Id,
            1 => p2Id,
            _ => ""
        };

        _game = new SeaEngine.Game(_cardLoader, NullLogger.Instance, p1Id, p2Id);
        _game.Init(p1Deck, p2Deck);
        _mctsGame = null;

        AdvanceOpponentUntilAiTurnOrDone();
        return BuildSnapshot();
    }

    /// <summary>
    /// 액션 적용 후 다음 스냅샷 반환.
    /// opponentType > 0 이면 상대 턴을 자동 소진 후 반환.
    /// </summary>
    public RlObservationFrame Step(string actionUidStr)
    {
        var g = _game ?? throw new InvalidOperationException("Game not initialized");
        g.UseAction(Uid.Parse(actionUidStr));
        _mctsGame = null;
        AdvanceOpponentUntilAiTurnOrDone();
        return BuildSnapshot();
    }

    public RlObservationFrame MctsReset()
    {
        var g = _game ?? throw new InvalidOperationException("Game not initialized. Call Init first.");
        _mctsGame = g.Fork();
        return BuildMctsSnapshot();
    }

    public RlObservationFrame MctsStep(string actionUidStr)
    {
        var g = _mctsGame ?? throw new InvalidOperationException("MCTS game not initialized. Call MctsReset first.");
        g.UseAction(Uid.Parse(actionUidStr));
        AdvanceOpponentUntilAiTurnOrDone(g);
        return BuildMctsSnapshot();
    }

    public SeaEngine.RL.RlMctsLeafFrame[] MctsBatchStep(int[] actionIndices)
    {
        var g = _game ?? throw new InvalidOperationException("Game not initialized");
        if (actionIndices.Length == 0) return Array.Empty<SeaEngine.RL.RlMctsLeafFrame>();

        var results = new SeaEngine.RL.RlMctsLeafFrame[actionIndices.Length];
        for (var i = 0; i < actionIndices.Length; i++)
        {
            var fork = g.Fork();
            var actions = fork.Actions.ToArray();
            var idx = actionIndices[i];
            if (idx < 0 || idx >= actions.Length)
                throw new InvalidOperationException($"Invalid action index: {idx} (action_count={actions.Length})");
            fork.UseAction(actions[idx].Guid);
            AdvanceOpponentUntilAiTurnOrDone(fork, _aiId);
            var turnCounter = fork.Data.TurnCnt + 1;
            results[i] = RlObservationExporter.ExportMctsLeaf(fork, turnCounter);
        }
        return results;
    }

    public RlObservationFrame? TryBuildLiveSnapshot()
    {
        if (_game is null) return null;
        var turnCounter = _game.Data.TurnCnt + 1;
        return RlObservationExporter.Export(_game, turnCounter);
    }

    private RlObservationFrame BuildSnapshot()
    {
        var turnCounter = _game!.Data.TurnCnt + 1;
        return RlObservationExporter.Export(_game!, turnCounter);
    }

    private RlObservationFrame BuildMctsSnapshot()
    {
        var turnCounter = _mctsGame!.Data.TurnCnt + 1;
        return RlObservationExporter.Export(_mctsGame!, turnCounter);
    }

    private bool ShouldAutoAdvance(SeaEngine.Game? g, string aiId)
    {
        if (_opponentType == OpponentType.External || g == null || g.Data.Winner != null)
            return false;
        if (string.IsNullOrEmpty(aiId)) return false;
        return g.Data.ActivePlayerId != aiId;
    }

    private void AdvanceOpponentUntilAiTurnOrDone()
        => AdvanceOpponentUntilAiTurnOrDone(_game, _aiId);

    private void AdvanceOpponentUntilAiTurnOrDone(SeaEngine.Game? g)
        => AdvanceOpponentUntilAiTurnOrDone(g, _aiId);

    private void AdvanceOpponentUntilAiTurnOrDone(SeaEngine.Game? g, string aiId)
    {
        if (g == null) return;

        var steps = 0;
        while (ShouldAutoAdvance(g, aiId) && steps < MaxAutoOpponentSteps)
        {
            var action = SelectOpponentAction(g);
            if (action == null) break;
            g.UseAction(action.Guid);
            steps++;
        }
    }

    private GameAction? SelectOpponentAction(SeaEngine.Game g)
    {
        var actions = g.Actions.ToArray();
        if (actions.Length == 0) return null;

        if (_opponentType == OpponentType.Random)
            return actions[_rng.Next(actions.Length)];

        GameAction? best = null;
        var bestScore = double.NegativeInfinity;
        foreach (var action in actions)
        {
            var score = _opponentType == OpponentType.Greedy
                ? ScoreGreedyAction(g, action)
                : ScoreRuleAction(g, action);
            if (score > bestScore || (score == bestScore && _rng.Next(2) == 0))
            {
                bestScore = score;
                best = action;
            }
        }
        return best ?? actions[0];
    }

    private double ScoreGreedyAction(SeaEngine.Game g, GameAction action)
    {
        var effectId   = action.EffectId ?? string.Empty;
        var targetType = action.Target.Type;
        var target     = targetType == EffectTarget.Types.Unit ? TryGetCard(g, action.Target.Guid) : null;
        var source     = TryGetCard(g, action.Source);
        var (activeId, enemyId) = ActiveAndEnemy(g);
        var enemyLeader = FindLeader(g, enemyId);

        var score = effectId switch
        {
            "DefaultAttack" => 120.0,
            "DeployUnit"    => 70.0,
            "DefaultMove"   => 25.0,
            "TurnEnd"       => -150.0,
            _               => 55.0
        };

        if (target != null)
        {
            var targetHp  = (double)target.Unit.Hp;
            var targetAtk = EffectiveAtk(target);
            var sourceAtk = source != null ? EffectiveAtk(source) : 0.0;
            if (Role(target) == "Leader") score += 200.0;
            if (sourceAtk >= targetHp && targetHp > 0) score += 60.0 + targetAtk * 2.0;
            score += Math.Max(0.0, 18.0 - targetHp);
        }

        if (targetType == EffectTarget.Types.Cell)
        {
            var tx = action.Target.PosX;
            var ty = action.Target.PosY;
            if (enemyLeader != null)
                score += Math.Max(0.0, 16.0 - Distance(tx, ty, enemyLeader.Unit.PosX, enemyLeader.Unit.PosY) * 2.0);
            if (tx is >= 2 and <= 3 && ty is >= 2 and <= 3) score += 8.0;
        }

        if (source != null && effectId == "DeployUnit")
        {
            score += Role(source) switch
            {
                "Leader" => 14.0, "Knight" => 11.0, "Bishop" => 10.0, "Rook" => 10.0, "Pawn" => 7.0, _ => 0.0
            };
            var srcId = source.Data.Id;
            if (srcId is "Or_N" or "Cl_B" or "Cl_N") score += 6.0;
        }

        score += Math.Min(CountPlaced(g, activeId), 7) * 1.0;
        score -= Math.Min(CountPlaced(g, enemyId), 7) * 0.5;
        return score;
    }

    private static double ScoreRuleAction(SeaEngine.Game g, GameAction action)
    {
        var (activeId, enemyId) = ActiveAndEnemy(g);
        var ownLeader   = FindLeader(g, activeId);
        var enemyLeader = FindLeader(g, enemyId);
        var source      = TryGetCard(g, action.Source);
        var target      = action.Target.Type is EffectTarget.Types.Unit or EffectTarget.Types.Unit2 or EffectTarget.Types.Card
            ? TryGetCard(g, action.Target.Guid) : null;
        var effectId   = action.EffectId ?? string.Empty;
        var targetType = action.Target.Type;
        var sourceRole = source != null ? Role(source) : string.Empty;
        var sourceId   = source?.Data.Id ?? string.Empty;
        var sourceAtk  = source != null ? EffectiveAtk(source) : 0.0;

        if (effectId == "TurnEnd")
            return -200.0 - g.Actions.Count(a => (a.EffectId ?? string.Empty) != "TurnEnd");

        var score = effectId switch
        {
            "DefaultAttack" => 110.0,
            "DeployUnit"    => 58.0,
            "DefaultMove"   => 28.0,
            _               => 72.0
        };

        if (target != null)
        {
            var targetRole = Role(target);
            var targetHp   = (double)target.Unit.Hp;
            var targetAtk  = EffectiveAtk(target);
            if (targetRole == "Leader")
            {
                score += 190.0;
                score += Math.Max(0.0, 40.0 - targetHp * 4.0);
            }
            if (sourceAtk >= targetHp && targetHp > 0)
            {
                score += 55.0 + targetAtk * 3.0;
                if (targetRole is "Bishop" or "Knight" or "Rook") score += 12.0;
            }
            if (source != null && targetAtk >= source.Unit.Hp && source.Unit.Hp > 0) score -= 18.0;
            score += Math.Max(0.0, 16.0 - targetHp);
            score += CountAttackersOn(g, target, activeId) * 3.0;
        }

        var (tx, ty) = CellAfterAction(action, source);
        if (targetType == EffectTarget.Types.Cell && tx >= 0 && ty >= 0)
        {
            if (tx is >= 2 and <= 3 && ty is >= 2 and <= 3) score += 12.0;
            if (enemyLeader != null)
            {
                var before = source == null ? 99
                    : Distance(source.Unit.PosX, source.Unit.PosY, enemyLeader.Unit.PosX, enemyLeader.Unit.PosY);
                var after = Distance(tx, ty, enemyLeader.Unit.PosX, enemyLeader.Unit.PosY);
                score += Math.Max(-10.0, (before - after) * 8.0);
                if (after <= 2) score += 18.0;
            }
            if (ownLeader != null && sourceRole != "Leader")
            {
                if (Distance(tx, ty, ownLeader.Unit.PosX, ownLeader.Unit.PosY) <= 2) score += 4.0;
            }
        }

        if (effectId == "DeployUnit")
        {
            score += sourceRole switch
            {
                "Leader" => 14.0, "Knight" => 13.0, "Bishop" => 11.0, "Rook" => 10.0, "Pawn" => 7.0, _ => 0.0
            };
            if (sourceId is "Cl_B" or "Or_N") score += 6.0;
        }
        if (sourceId == "Or_N" && target != null) score += 8.0;
        if (sourceId is "Cl_B" or "Cl_N" or "Cl_R" && effectId is not ("TurnEnd" or "DefaultMove")) score += 5.0;

        score += Math.Min(CountPlaced(g, activeId), 7) * 1.2;
        score -= Math.Min(CountPlaced(g, enemyId), 7) * 0.6;
        return score;
    }

    private static (string ActiveId, string EnemyId) ActiveAndEnemy(SeaEngine.Game g)
    {
        var activeId = g.Data.ActivePlayerId;
        var enemyId  = g.Data.Player1.Id == activeId ? g.Data.Player2.Id : g.Data.Player1.Id;
        return (activeId, enemyId);
    }

    private static string Role(Card card) => card.Data.UnitType.ToString();

    private static double EffectiveAtk(Card card)
    {
        // TempAtk already changes Unit.Atk when applied, so this should reflect the live attack value.
        return card.Unit.Atk;
    }

    private static int Distance(int x1, int y1, int x2, int y2)
    {
        if (Math.Min(Math.Min(x1, y1), Math.Min(x2, y2)) < 0) return 99;
        return Math.Abs(x1 - x2) + Math.Abs(y1 - y2);
    }

    private static (int X, int Y) CellAfterAction(GameAction action, Card? source)
    {
        if (action.Target.Type == EffectTarget.Types.Cell)
            return (action.Target.PosX, action.Target.PosY);
        return source == null ? (-1, -1) : (source.Unit.PosX, source.Unit.PosY);
    }

    private static Card? FindLeader(SeaEngine.Game g, string ownerId)
        => g.Data.Board.Cards.FirstOrDefault(c => c.Owner.Id == ownerId && c.Unit.IsPlaced && Role(c) == "Leader");

    private static int CountPlaced(SeaEngine.Game g, string ownerId)
        => g.Data.Board.Cards.Count(c => c.Owner.Id == ownerId && c.Unit.IsPlaced);

    private static int CountAttackersOn(SeaEngine.Game g, Card target, string byOwnerId)
    {
        if (!target.Unit.IsPlaced) return 0;
        var tx = target.Unit.PosX;
        var ty = target.Unit.PosY;
        var count = 0;
        foreach (var other in g.Data.Board.Cards)
        {
            if (!other.Unit.IsPlaced || other.Owner.Id != byOwnerId) continue;
            try { if (g.Data.GetMoveArea(other).Contains((tx, ty))) count++; }
            catch { }
        }
        return count;
    }

    private static Card? TryGetCard(SeaEngine.Game g, Uid uid)
    {
        if (uid == Uid.None) return null;
        try { return g.Data.Board.GetCardById(uid); }
        catch { return null; }
    }
}
