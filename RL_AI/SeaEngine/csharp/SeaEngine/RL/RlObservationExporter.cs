using SeaEngine.CardManager;
using SeaEngine.Common;
using SeaEngine.GameDataManager;
using SeaEngine.GameDataManager.Components;
using SeaEngine.GameEffectManager;

namespace SeaEngine.RL;

public sealed record RlStatusView(string Type, int Value);

public sealed record RlCardView(
    string Uid,
    string CardId,
    string Name,
    string OwnerId,
    string Role,
    int Atk,
    int EffectiveAtk,
    int Hp,
    int MaxHp,
    bool IsPlaced,
    bool IsMoved,
    bool IsAttacked,
    int PosX,
    int PosY,
    RlStatusView[] Statuses);

public sealed record RlPlayerView(
    string Id,
    int HandCount,
    int DeckCount,
    int TrashCount,
    RlCardView[] Hand);

public sealed record RlActionView(
    string Uid,
    string EffectId,
    string Source,
    string TargetType,
    string TargetGuid,
    string TargetGuid2,
    int PosX,
    int PosY,
    string Text);

public sealed record RlObservationFrame(
    int Turn,
    string ActivePlayerId,
    string Result,
    string WinnerId,
    RlPlayerView[] Players,
    RlCardView[] Board,
    RlActionView[] Actions,
    float[] StateVector,
    float[][] ActionFeatureVectors);

public sealed record RlMctsLeafFrame(
    string Result,
    string WinnerId,
    float[] StateVector);

public static class RlObservationExporter
{
    private const int GlobalFeatureDim = 49;
    private const int BoardTokenDim = 28;
    private const int HandTokenDim = 10;
    private const int MaxBoardCards = 14;
    private const int MaxHandCards = 7;
    private const int StateVectorDim = GlobalFeatureDim + MaxBoardCards * BoardTokenDim + MaxHandCards * HandTokenDim;
    // Keep in sync with Python observation.py.
    private const int ActionFeatureDim = 63;

    private static readonly string[] RoleOrder = ["Leader", "Bishop", "Knight", "Rook", "Pawn"];
    private static readonly string[] EffectBuckets = ["DeployUnit", "DefaultMove", "DefaultAttack", "TurnEnd", "Skill"];
    private static readonly string[] TargetBuckets = ["None", "Cell", "Unit", "Unit2", "Card"];

    private static void PadTo(List<float> values, int targetLength, string label)
    {
        if (values.Count > targetLength)
        {
            throw new InvalidOperationException($"{label} vector too long: {values.Count} > {targetLength}");
        }

        while (values.Count < targetLength)
        {
            values.Add(0.0f);
        }
    }

    public static RlObservationFrame Export(Game game, int turnCounter)
    {
        var data = game.Data;
        var playerId = data.ActivePlayerId;

        var board = data.Board.Cards.ToArray();
        // Board snapshots can briefly contain duplicated Uids during state transitions.
        // Keep the first instance so the exporter stays read-only and never crashes on duplicates.
        var boardByUid = board
            .GroupBy(card => card.Guid.ToString())
            .ToDictionary(group => group.Key, group => group.First());
        var actions = game.Actions.ToArray();
        var actionViews = actions.Select(BuildActionView).ToArray();
        var players = new[] { BuildPlayerView(data.Player1), BuildPlayerView(data.Player2) };

        var stateVector = BuildStateVector(data, playerId, board, actionViews, boardByUid);
        var actionFeatureVectors = actionViews.Select(action => BuildActionFeatureVector(data, playerId, board, boardByUid, actionViews, action)).ToArray();

        return new RlObservationFrame(
            turnCounter,
            playerId,
            BuildResult(data),
            data.WinnerId,
            players,
            board.Select(BuildCardView).ToArray(),
            actionViews,
            stateVector,
            actionFeatureVectors
        );
    }

    public static RlMctsLeafFrame ExportMctsLeaf(Game game, int turnCounter)
    {
        var data = game.Data;
        var playerId = data.ActivePlayerId;

        var board = data.Board.Cards.ToArray();
        // Board snapshots can briefly contain duplicated Uids during state transitions.
        // Keep the first instance so the exporter stays read-only and never crashes on duplicates.
        var boardByUid = board
            .GroupBy(card => card.Guid.ToString())
            .ToDictionary(group => group.Key, group => group.First());
        var actions = game.Actions.ToArray();
        var actionViews = actions.Select(BuildActionView).ToArray();
        var stateVector = BuildStateVector(data, playerId, board, actionViews, boardByUid);

        return new RlMctsLeafFrame(
            BuildResult(data),
            data.WinnerId,
            stateVector
        );
    }

    private static string BuildResult(GameData data)
    {
        if (data.Winner == null) return "Ongoing";
        return data.Winner.Id == data.Player1.Id ? "Player1Win" : "Player2Win";
    }

    private static RlPlayerView BuildPlayerView(Player player)
    {
        return new RlPlayerView(
            player.Id,
            player.Hand.Count,
            player.Deck.Count,
            player.Trash.Count,
            player.Hand.Cards.Select(BuildCardView).ToArray()
        );
    }

    private static RlCardView BuildCardView(Card card)
    {
        var buffs = card.Unit.Buffs
            .Select(buff => new RlStatusView(buff.Key, buff.Value))
            .ToArray();
        var role = card.Data.UnitType.ToString();
        // TempAtk already mutates Unit.Atk during gameplay, so do not add it again here.
        var effectiveAtk = card.Unit.Atk;
        return new RlCardView(
            card.Guid.ToString(),
            card.Data.Id,
            card.Data.Name,
            card.Owner.Id,
            role,
            card.Unit.Atk,
            effectiveAtk,
            card.Unit.Hp,
            card.Unit.MaxHp,
            card.Unit.IsPlaced,
            card.Unit.IsMoved,
            false,
            card.Unit.PosX,
            card.Unit.PosY,
            buffs
        );
    }

    private static RlActionView BuildActionView(GameAction action)
    {
        var targetType = action.Target.Type.ToString();
        return new RlActionView(
            action.Guid.ToString(),
            action.EffectId,
            action.Source.ToString(),
            targetType,
            action.Target.Guid.ToString(),
            action.Target.Guid2.ToString(),
            action.Target.PosX,
            action.Target.PosY,
            action.ToString()
        );
    }

    private static float[] BuildStateVector(GameData data, string playerId, Card[] board, RlActionView[] actions, Dictionary<string, Card> boardByUid)
    {
        var (_, enemyPlayer, enemyId) = GetPlayers(data, playerId);
        var (ownLeader, enemyLeader) = GetLeaders(board, playerId, enemyId);

        var ownBoard = board.Where(card => card.Owner.Id == playerId && card.Unit.IsPlaced).ToArray();
        var enemyBoard = board.Where(card => card.Owner.Id == enemyId && card.Unit.IsPlaced).ToArray();

        float ownAttackTotal = ownBoard.Sum(card => (float)card.Unit.Atk);
        float enemyAttackTotal = enemyBoard.Sum(card => (float)card.Unit.Atk);
        float ownHpTotal = ownBoard.Sum(card => (float)card.Unit.Hp);
        float enemyHpTotal = enemyBoard.Sum(card => (float)card.Unit.Hp);
        float ownReadyMove = ownBoard.Count(card => !card.Unit.IsMoved);
        float ownDeployable = CountDeployableCards(data, playerId, actions);
        float ownSkillActions = CountSkillActions(data, playerId, actions);
        float ownAttackActions = CountActions(data, playerId, actions, "DefaultAttack", boardByUid);
        float ownMoveActions = CountActions(data, playerId, actions, "DefaultMove", boardByUid);
        float enemyAttackersOnLeader = ownLeader is null ? 0.0f : CountAttackersOfCard(board, ownLeader);
        float ownAttackersOnEnemyLeader = enemyLeader is null ? 0.0f : CountAttackersOfCard(board, enemyLeader);
        float centerControlOwn = ownBoard.Count(card => 2 <= card.Unit.PosX && card.Unit.PosX <= 3 && 2 <= card.Unit.PosY && card.Unit.PosY <= 3);
        float centerControlEnemy = enemyBoard.Count(card => 2 <= card.Unit.PosX && card.Unit.PosX <= 3 && 2 <= card.Unit.PosY && card.Unit.PosY <= 3);
        float ownPawnProgress = CountPawnProgress(data, board, playerId);
        float enemyPawnProgress = CountPawnProgress(data, board, enemyId);
        float ownPawnLastRank = CountPawnLastRank(data, board, playerId);
        float enemyPawnLastRank = CountPawnLastRank(data, board, enemyId);
        var ownSpecialEffectFeatures = CountSpecialEffectFeatures(data, board, playerId);
        var enemySpecialEffectFeatures = CountSpecialEffectFeatures(data, board, enemyId);

        var actionCounts = EffectBuckets.ToDictionary(bucket => bucket, _ => 0.0f);
        foreach (var action in actions)
        {
            var effectId = action.EffectId;
            var bucket = EffectBuckets.Contains(effectId) && effectId != "Skill" ? effectId : "Skill";
            actionCounts[bucket] += 1.0f;
        }
        var actionTotal = Math.Max(1.0f, actions.Length);

        var resultVector = new List<float>();
        resultVector.AddRange(new float[]
        {
            NormalizeRatio(data.GetPlayer(playerId).Hand.Count, 7.0f),
            NormalizeRatio(enemyPlayer.Hand.Count, 7.0f),
            NormalizeRatio(data.GetPlayer(playerId).Deck.Count, 14.0f),
            NormalizeRatio(enemyPlayer.Deck.Count, 14.0f),
            NormalizeRatio(data.GetPlayer(playerId).Trash.Count, 14.0f),
            NormalizeRatio(enemyPlayer.Trash.Count, 14.0f),
            ownLeader == null ? 0.0f : NormalizeRatio(ownLeader.Unit.Hp, Math.Max(1.0f, ownLeader.Unit.MaxHp)),
            enemyLeader == null ? 0.0f : NormalizeRatio(enemyLeader.Unit.Hp, Math.Max(1.0f, enemyLeader.Unit.MaxHp)),
            ownLeader == null || enemyLeader == null ? 0.0f : NormalizeRatio(ownLeader.Unit.Hp - enemyLeader.Unit.Hp, Math.Max(1.0f, ownLeader.Unit.MaxHp)),
            NormalizeRatio(ownBoard.Length, 14.0f),
            NormalizeRatio(enemyBoard.Length, 14.0f),
            NormalizeRatio(ownBoard.Length - enemyBoard.Length, 14.0f),
            NormalizeRatio(ownAttackTotal, 40.0f),
            NormalizeRatio(enemyAttackTotal, 40.0f),
            NormalizeRatio(ownAttackTotal - enemyAttackTotal, 40.0f),
            NormalizeRatio(ownHpTotal, 40.0f),
            NormalizeRatio(enemyHpTotal, 40.0f),
            NormalizeRatio(ownReadyMove, 14.0f),
            NormalizeRatio(ownDeployable, 7.0f),
            NormalizeRatio(ownSkillActions, 7.0f),
            NormalizeRatio(ownAttackActions, 20.0f),
            NormalizeRatio(ownMoveActions, 20.0f),
            NormalizeRatio(enemyAttackersOnLeader, 6.0f),
            NormalizeRatio(ownAttackersOnEnemyLeader, 6.0f),
            NormalizeRatio(centerControlOwn, 4.0f),
            NormalizeRatio(centerControlEnemy, 4.0f),
            NormalizeRatio(ownPawnProgress, 1.0f),
            NormalizeRatio(enemyPawnProgress, 1.0f),
            NormalizeRatio(ownPawnLastRank, 7.0f),
            NormalizeRatio(enemyPawnLastRank, 7.0f),
        });
        resultVector.AddRange(ownSpecialEffectFeatures);
        resultVector.AddRange(enemySpecialEffectFeatures);
        resultVector.AddRange(EffectBuckets.Select(bucket => NormalizeRatio(actionCounts[bucket], actionTotal)));

        PadTo(resultVector, GlobalFeatureDim, "global");

        var boardVector = BuildBoardVector(data, playerId, board, ownLeader, enemyLeader);
        var handVector = BuildHandVector(data, playerId);
        resultVector.AddRange(boardVector);
        resultVector.AddRange(handVector);
        PadTo(resultVector, StateVectorDim, "state");
        return resultVector.ToArray();
    }

    private static float[] BuildBoardVector(GameData data, string playerId, Card[] board, Card? ownLeader, Card? enemyLeader)
    {
        var enemyId = playerId == data.Player1.Id ? data.Player2.Id : data.Player1.Id;
        var mirrorView = ShouldMirrorPerspective(ownLeader, enemyLeader);
        var ownLx = ViewX(ownLeader?.Unit.PosX ?? -1, mirrorView);
        var ownLy = ViewY(ownLeader?.Unit.PosY ?? -1, mirrorView);
        var enemyLx = ViewX(enemyLeader?.Unit.PosX ?? -1, mirrorView);
        var enemyLy = ViewY(enemyLeader?.Unit.PosY ?? -1, mirrorView);
        var vectors = new List<float>();
        var cards = board.OrderBy(card => card.Owner.Id == playerId ? 0 : 1)
            .ThenBy(card => RoleRank(RoleFromCard(card)))
            .ThenBy(card => ViewX(card.Unit.PosX, mirrorView))
            .ThenBy(card => ViewY(card.Unit.PosY, mirrorView))
            .ThenBy(card => card.Guid.ToString())
            .ToArray();

        foreach (var card in cards.Take(MaxBoardCards))
        {
            var (_, hasMoveLock, hasAttackLock, timedStatusCount) = StatusSummary(card);
            var cx = ViewX(card.Unit.PosX, mirrorView);
            var cy = ViewY(card.Unit.PosY, mirrorView);
            var role = RoleFromCard(card);
            var hp = card.Unit.Hp;
            var maxHp = Math.Max(1, card.Unit.MaxHp);
            var baseAtk = card.Data.Atk;
            var effectiveAtk = card.Unit.Atk;
            var reachableTargets = CountReadyAttackTargets(board, card);
            var adjacentEnemies = CountEnemyNeighbors(board, card);
            var incomingAttackers = CountAttackersOfCard(board, card);
            var threatensEnemyLeader = enemyLeader != null && Distance(cx, cy, enemyLx, enemyLy) is >= 0 and <= 0.2f ? 1.0f : 0.0f;
            var inCenter = 2 <= cx && cx <= 3 && 2 <= cy && cy <= 3 ? 1.0f : 0.0f;
            var rowProgress = card.Unit.IsPlaced ? NormalizeRatio(cx, Board.BoardSize - 1) : 0.0f;
            var promotionReady = PawnPromotionReady(card, mirrorView, card.Owner.Id == playerId);
            var promotionDistance = PawnPromotionDistance(card, mirrorView, card.Owner.Id == playerId);

            vectors.AddRange(new float[]
            {
                card.Owner.Id == playerId ? 1.0f : -1.0f,
                card.Unit.IsPlaced ? 1.0f : 0.0f,
                card.Unit.IsMoved ? 1.0f : 0.0f,
                0.0f,
                NormalizePos(cx),
                NormalizePos(cy),
                NormalizeRatio(hp, maxHp),
                NormalizeRatio(maxHp, 10.0f),
                NormalizeRatio(baseAtk, 10.0f),
                NormalizeRatio(effectiveAtk, 10.0f),
                hasMoveLock,
                hasAttackLock,
                NormalizeRatio(timedStatusCount, 4.0f),
                Distance(cx, cy, ownLx, ownLy),
                Distance(cx, cy, enemyLx, enemyLy),
                NormalizeRatio(reachableTargets, 6.0f),
                NormalizeRatio(adjacentEnemies, 6.0f),
                NormalizeRatio(incomingAttackers, 6.0f),
                threatensEnemyLeader,
                inCenter,
                rowProgress,
                promotionReady,
                promotionDistance,
            });
            vectors.AddRange(RoleOneHot(role));
        }

        var missingSlots = MaxBoardCards - Math.Min(cards.Length, MaxBoardCards);
        if (missingSlots > 0)
        {
            for (var i = 0; i < missingSlots * BoardTokenDim; i++) vectors.Add(0.0f);
        }
        return vectors.ToArray();
    }

    private static float[] BuildHandVector(GameData data, string playerId)
    {
        var player = data.GetPlayer(playerId);
        var hand = player.Hand.Cards.ToArray();
        var vectors = new List<float>();
        foreach (var card in hand.Take(MaxHandCards))
        {
            var tokenStart = vectors.Count;
            var role = RoleFromCard(card);
            var cardId = card.Data.Id;
            var deployable = 0.0f;
            var skillUsable = 0.0f;
            vectors.AddRange(new float[]
            {
                1.0f,
                cardId.StartsWith("Or_") ? 1.0f : 0.0f,
                cardId.StartsWith("Cl_") ? 1.0f : 0.0f,
                deployable,
                skillUsable,
            });
            vectors.AddRange(RoleOneHot(role));
            // Python hand token is 20-dim; keep the old features and pad the tail.
            PadTo(vectors, tokenStart + HandTokenDim, "hand token");
        }
        var missingSlots = MaxHandCards - Math.Min(hand.Length, MaxHandCards);
        if (missingSlots > 0)
        {
            for (var i = 0; i < missingSlots * HandTokenDim; i++)
            {
                vectors.Add(0.0f);
            }
        }

        PadTo(vectors, MaxHandCards * HandTokenDim, "hand");
        return vectors.ToArray();
    }

    private static float[] BuildActionFeatureVector(GameData data, string playerId, Card[] board, Dictionary<string, Card> boardByUid, RlActionView[] actions, RlActionView action)
    {
        var (_, _, enemyId) = GetPlayers(data, playerId);
        var ownLeader = GetLeaders(board, playerId, enemyId).own;
        var enemyLeader = GetLeaders(board, playerId, enemyId).enemy;
        var mirrorView = ShouldMirrorPerspective(ownLeader, enemyLeader);

        var effectId = action.EffectId;
        var targetType = action.TargetType;
        var source = boardByUid.TryGetValue(action.Source, out var src) ? src : null;
        var targetCard = (targetType is "Unit" or "Card") && boardByUid.TryGetValue(action.TargetGuid, out var tgt) ? tgt : null;
        var targetCard2 = targetType == "Unit2" && boardByUid.TryGetValue(action.TargetGuid2, out var tgt2) ? tgt2 : null;

        var targetX = ViewX(action.PosX, mirrorView);
        var targetY = ViewY(action.PosY, mirrorView);
        var sourceX = ViewX(source?.Unit.PosX ?? -1, mirrorView);
        var sourceY = ViewY(source?.Unit.PosY ?? -1, mirrorView);
        var enemyLx = ViewX(enemyLeader?.Unit.PosX ?? -1, mirrorView);
        var enemyLy = ViewY(enemyLeader?.Unit.PosY ?? -1, mirrorView);

        var sourceRole = source != null ? RoleFromCard(source) : "";
        var targetRole = targetCard != null ? RoleFromCard(targetCard) : "";
        var sourceAdjacentEnemies = source != null ? CountEnemyNeighbors(board, source) : 0.0f;
        var targetIncomingAttackers = targetCard != null ? CountAttackersOfCard(board, targetCard) : 0.0f;

        var moveDistanceBefore = Distance(sourceX, sourceY, enemyLx, enemyLy);
        var moveDistanceAfter = targetType == "Cell" ? Distance(targetX, targetY, enemyLx, enemyLy) : moveDistanceBefore;
        var movesCloser = moveDistanceAfter >= 0 && moveDistanceBefore >= 0 && moveDistanceAfter < moveDistanceBefore ? 1.0f : 0.0f;
        var entersLeaderZone = targetType == "Cell" && moveDistanceAfter >= 0 && moveDistanceAfter <= 0.2f ? 1.0f : 0.0f;

        var targetHp = targetCard?.Unit.Hp ?? 0.0f;
        var targetMaxHp = Math.Max(1.0f, targetCard?.Unit.MaxHp ?? 1.0f);
        var sourceAtk = source?.Unit.Atk ?? 0.0f;
        var canKillTarget = targetCard != null && sourceAtk >= targetHp && targetHp > 0 ? 1.0f : 0.0f;
        var threatensEnemyLeader = targetCard != null && targetCard.Owner.Id != playerId && targetRole == "Leader" ? 1.0f : 0.0f;
        var affectsTwoUnits = targetCard2 != null ? 1.0f : 0.0f;
        var sourceSurvivesTrade = targetCard != null && source != null && targetCard.Unit.Atk < source.Unit.Hp ? 1.0f : 0.0f;
        var targetIsLowHp = targetCard != null && targetHp <= 2.0f ? 1.0f : 0.0f;
        var sourceFromHand = source != null && !source.Unit.IsPlaced ? 1.0f : 0.0f;
        var targetValueScore = TargetValueScore(playerId, source, targetCard);

        var vectors = new List<float>();
        vectors.AddRange(EffectOneHot(effectId));
        vectors.AddRange(TargetOneHot(targetType));
        vectors.AddRange(new float[]
        {
            source == null ? 0.0f : sourceFromHand,
            source == null ? 0.0f : NormalizeRatio(sourceAtk, 10.0f),
            source == null ? 0.0f : NormalizeRatio(source.Unit.Hp, Math.Max(1.0f, source.Unit.MaxHp)),
        });
        vectors.AddRange(RoleOneHot(sourceRole));
        vectors.AddRange(new float[]
        {
            targetCard == null ? 0.0f : (targetCard.Owner.Id != playerId ? 1.0f : -1.0f),
            targetCard == null ? 0.0f : NormalizeRatio(targetCard.Unit.Atk, 10.0f),
            targetCard == null ? 0.0f : NormalizeRatio(targetHp, targetMaxHp),
            targetCard == null ? 0.0f : (targetRole == "Leader" ? 1.0f : 0.0f),
        });
        vectors.AddRange(RoleOneHot(targetRole));
        vectors.AddRange(new float[]
        {
            NormalizePos(sourceX),
            NormalizePos(sourceY),
            NormalizePos(targetX),
            NormalizePos(targetY),
            moveDistanceBefore >= 0 ? moveDistanceBefore : 0.0f,
            moveDistanceAfter >= 0 ? moveDistanceAfter : 0.0f,
            movesCloser,
            entersLeaderZone,
            canKillTarget,
            threatensEnemyLeader,
            affectsTwoUnits,
            sourceSurvivesTrade,
            targetIsLowHp,
            sourceFromHand,
            NormalizeRatio(sourceAdjacentEnemies, 6.0f),
            NormalizeRatio(targetIncomingAttackers, 6.0f),
        });

        vectors.AddRange(SameSourceCompetitionFeatures(playerId, board, boardByUid, actions, action, source, targetCard, canKillTarget, targetValueScore));
        vectors.AddRange(GlobalActionCompetitionFeatures(playerId, board, boardByUid, actions, action, source, targetCard, targetValueScore));
        if (vectors.Count != ActionFeatureDim)
        {
            throw new InvalidOperationException($"action vector length mismatch: {vectors.Count} != {ActionFeatureDim}");
        }
        return vectors.ToArray();
    }

    private static float TargetValueScore(string playerId, Card? source, Card? targetCard)
    {
        if (targetCard is null)
        {
            return 0.0f;
        }

        var targetOwner = targetCard.Owner.Id;
        var isEnemy = targetOwner != playerId;
        var role = RoleFromCard(targetCard);
        var hp = targetCard.Unit.Hp;
        var maxHp = Math.Max(1.0f, targetCard.Unit.MaxHp);
        var targetAtk = targetCard.Unit.Atk;
        var sourceAtk = source?.Unit.Atk ?? 0.0f;
        var sourceHp = source?.Unit.Hp ?? 0.0f;

        var roleValue = role switch
        {
            "Leader" => 1.00f,
            "Rook" => 0.72f,
            "Bishop" => 0.66f,
            "Knight" => 0.58f,
            "Pawn" => 0.34f,
            _ => 0.25f,
        };

        float killBonus = 0.0f;
        if (isEnemy && source != null && sourceAtk >= hp && hp > 0)
        {
            killBonus = 0.30f + 0.20f * roleValue;
        }

        float tradeBonus = 0.0f;
        if (isEnemy && source != null)
        {
            tradeBonus = targetAtk < sourceHp ? 0.12f : -0.08f;
        }

        float threatBonus = 0.0f;
        if (isEnemy && targetAtk >= 4.0f)
        {
            threatBonus += 0.08f;
        }

        var lowHpBonus = isEnemy && hp <= 2.0f ? 0.08f : 0.0f;
        var leaderBonus = isEnemy && role == "Leader" ? 0.35f : 0.0f;
        var friendlyPenalty = isEnemy ? 0.0f : -0.35f;
        var hpFactor = Math.Clamp(hp / maxHp, 0.0f, 1.0f);

        var score = 0.12f + roleValue * 0.38f + NormalizeRatio(targetAtk, 10.0f) * 0.18f;
        score += killBonus + tradeBonus + threatBonus + lowHpBonus + leaderBonus + friendlyPenalty;
        score += isEnemy ? (1.0f - hpFactor) * 0.07f : 0.0f;
        return Math.Clamp(score, 0.0f, 1.5f);
    }

    private static float[] SameSourceCompetitionFeatures(
        string playerId,
        Card[] board,
        Dictionary<string, Card> boardByUid,
        RlActionView[] actions,
        RlActionView action,
        Card? source,
        Card? targetCard,
        float canKillTarget,
        float targetValueScore)
    {
        var sourceUid = action.Source;
        var siblingActions = string.IsNullOrWhiteSpace(sourceUid)
            ? new[] { action }
            : actions.Where(candidate => candidate.Source == sourceUid).ToArray();
        if (siblingActions.Length == 0)
        {
            siblingActions = new[] { action };
        }

        var attackActions = siblingActions.Count(candidate => candidate.EffectId == "DefaultAttack");
        var moveActions = siblingActions.Count(candidate => candidate.EffectId == "DefaultMove");
        var deployActions = siblingActions.Count(candidate => candidate.EffectId == "DeployUnit");
        var skillActions = siblingActions.Count(candidate => candidate.EffectId is not ("DeployUnit" or "DefaultMove" or "DefaultAttack" or "TurnEnd"));

        var targetScores = new List<float>();
        var killScores = new List<float>();

        foreach (var sibling in siblingActions)
        {
            var siblingSource = boardByUid.TryGetValue(sibling.Source, out var candSrc) ? candSrc : source;
            var siblingTarget = GetActionTargetCard(boardByUid, sibling);
            var score = TargetValueScore(playerId, siblingSource, siblingTarget);

            if (siblingTarget != null)
            {
                targetScores.Add(score);

                var siblingTargetHp = siblingTarget.Unit.Hp;
                var siblingSourceAtk = siblingSource?.Unit.Atk ?? 0;

                if (siblingSourceAtk >= siblingTargetHp && siblingTargetHp > 0)
                {
                    killScores.Add(score);
                }

            }
        }

        var maxTargetScore = targetScores.Count > 0 ? targetScores.Max() : 0.0f;
        var maxKillScore = killScores.Count > 0 ? killScores.Max() : 0.0f;
        var betterTargetMargin = Math.Max(0.0f, maxTargetScore - targetValueScore);
        var betterKillMargin = canKillTarget < 0.5f ? Math.Max(0.0f, maxKillScore - targetValueScore) : 0.0f;

        var rankRatio = 0.0f;
        var isBestTarget = 0.0f;
        if (targetScores.Count > 0 && targetCard != null)
        {
            var sorted = targetScores.OrderByDescending(score => score).ToArray();
            var rankIndex = sorted.Count(score => score > targetValueScore + 1e-6f);
            rankRatio = sorted.Length == 1 ? 1.0f : 1.0f - (rankIndex / (float)(sorted.Length - 1));
            isBestTarget = rankIndex == 0 ? 1.0f : 0.0f;
        }

        return new[]
        {
            NormalizeRatio(siblingActions.Length, 20.0f),
            NormalizeRatio(attackActions, 10.0f),
            NormalizeRatio(moveActions, 10.0f),
            NormalizeRatio(deployActions, 10.0f),
            NormalizeRatio(skillActions, 10.0f),
            NormalizeRatio(maxTargetScore, 1.5f),
            NormalizeRatio(maxKillScore, 1.5f),
            NormalizeRatio(targetValueScore, 1.5f),
            rankRatio,
            isBestTarget,
            NormalizeRatio(betterTargetMargin, 1.5f),
            NormalizeRatio(betterKillMargin, 1.5f),
        };
    }

    private static float[] GlobalActionCompetitionFeatures(
        string playerId,
        Card[] board,
        Dictionary<string, Card> boardByUid,
        RlActionView[] actions,
        RlActionView action,
        Card? source,
        Card? targetCard,
        float targetValueScore)
    {
        if (actions.Length == 0)
        {
            return new float[8];
        }

        var effectId = action.EffectId;
        var sameEffectActions = actions.Where(candidate => candidate.EffectId == effectId).ToArray();
        var allScores = new List<float>();
        var sameEffectScores = new List<float>();

        foreach (var candidate in actions)
        {
            var candidateSource = boardByUid.TryGetValue(candidate.Source, out var candSrc) ? candSrc : null;
            var candidateTarget = GetActionTargetCard(boardByUid, candidate);
            var candidateScore = TargetValueScore(playerId, candidateSource, candidateTarget);
            allScores.Add(candidateScore);
            if (candidate.EffectId == effectId)
            {
                sameEffectScores.Add(candidateScore);
            }
        }

        var bestGlobalScore = allScores.Count > 0 ? allScores.Max() : 0.0f;
        var bestSameEffectScore = sameEffectScores.Count > 0 ? sameEffectScores.Max() : 0.0f;

        float RankRatio(IReadOnlyList<float> scores, float value)
        {
            if (scores.Count == 0)
            {
                return 0.0f;
            }
            if (scores.Count == 1)
            {
                return 1.0f;
            }
            var sorted = scores.OrderByDescending(score => score).ToArray();
            var rankIndex = sorted.Count(score => score > value + 1e-6f);
            return 1.0f - (rankIndex / (float)(sorted.Length - 1));
        }

        var globalRankRatio = RankRatio(allScores, targetValueScore);
        var sameEffectRankRatio = RankRatio(sameEffectScores, targetValueScore);
        var betterGlobalMargin = Math.Max(0.0f, bestGlobalScore - targetValueScore);
        var betterSameEffectMargin = Math.Max(0.0f, bestSameEffectScore - targetValueScore);

        return new[]
        {
            NormalizeRatio(actions.Length, 32.0f),
            NormalizeRatio(sameEffectActions.Length, 16.0f),
            NormalizeRatio(bestGlobalScore, 1.5f),
            NormalizeRatio(bestSameEffectScore, 1.5f),
            globalRankRatio,
            sameEffectRankRatio,
            NormalizeRatio(betterGlobalMargin, 1.5f),
            NormalizeRatio(betterSameEffectMargin, 1.5f),
        };
    }

    private static Card? GetActionTargetCard(Dictionary<string, Card> boardByUid, RlActionView action)
    {
        return action.TargetType is "Unit" or "Card"
            ? boardByUid.TryGetValue(action.TargetGuid, out var target) ? target : null
            : action.TargetType == "Unit2" && boardByUid.TryGetValue(action.TargetGuid2, out var target2) ? target2 : null;
    }

    private static (Player own, Player enemy, string enemyId) GetPlayers(GameData data, string playerId)
    {
        var enemyId = data.Player1.Id == playerId ? data.Player2.Id : data.Player1.Id;
        return (data.GetPlayer(playerId), data.GetPlayer(enemyId), enemyId);
    }

    private static (Card? own, Card? enemy) GetLeaders(Card[] board, string playerId, string enemyId)
    {
        Card? own = null;
        Card? enemy = null;
        foreach (var card in board)
        {
            if (!card.Unit.IsPlaced) continue;
            if (RoleFromCard(card) != "Leader") continue;
            if (card.Owner.Id == playerId) own = card;
            else if (card.Owner.Id == enemyId) enemy = card;
        }
        return (own, enemy);
    }

    private static float CountDeployableCards(GameData data, string playerId, RlActionView[] actions)
    {
        var player = data.GetPlayer(playerId);
        var deployable = 0.0f;
        foreach (var card in player.Hand.Cards)
        {
            var uid = card.Guid.ToString();
            if (actions.Any(action => action.EffectId == "DeployUnit" && action.Source == uid))
            {
                deployable += 1.0f;
            }
        }
        return deployable;
    }

    private static float CountSkillActions(GameData data, string playerId, RlActionView[] actions)
    {
        var player = data.GetPlayer(playerId);
        var handUids = player.Hand.Cards.Select(card => card.Guid.ToString()).ToHashSet();
        var count = 0.0f;
        foreach (var action in actions)
        {
            if (action.EffectId is not ("DeployUnit" or "DefaultMove" or "DefaultAttack" or "TurnEnd") && handUids.Contains(action.Source))
            {
                count += 1.0f;
            }
        }
        return count;
    }

    private static float CountActions(GameData data, string playerId, RlActionView[] actions, string effectId, Dictionary<string, Card> boardByUid)
    {
        var count = 0.0f;
        foreach (var action in actions)
        {
            if (boardByUid.TryGetValue(action.Source, out var sourceCard) && sourceCard.Owner.Id != playerId)
            {
                continue;
            }
            if (action.EffectId == effectId)
            {
                count += 1.0f;
            }
        }
        return count;
    }

    private static float CountAttackersOfCard(Card[] board, Card targetCard)
    {
        if (!targetCard.Unit.IsPlaced) return 0.0f;
        var tx = targetCard.Unit.PosX;
        var ty = targetCard.Unit.PosY;
        var targetOwner = targetCard.Owner.Id;
        var attackers = 0.0f;
        foreach (var other in board)
        {
            if (!other.Unit.IsPlaced || other.Owner.Id == targetOwner) continue;
            var ox = other.Unit.PosX;
            var oy = other.Unit.PosY;
            if (ox < 0 || oy < 0) continue;
            if (Math.Abs(ox - tx) <= 1 && Math.Abs(oy - ty) <= 1)
            {
                attackers += 1.0f;
            }
        }
        return attackers;
    }

    private static float CountReadyAttackTargets(Card[] board, Card card)
    {
        if (!card.Unit.IsPlaced) return 0.0f;
        var sourceOwner = card.Owner.Id;
        var sx = card.Unit.PosX;
        var sy = card.Unit.PosY;
        if (sx < 0 || sy < 0) return 0.0f;
        var reachable = 0.0f;
        foreach (var other in board)
        {
            if (!other.Unit.IsPlaced || other.Owner.Id == sourceOwner) continue;
            var ox = other.Unit.PosX;
            var oy = other.Unit.PosY;
            if (ox < 0 || oy < 0) continue;
            if (Math.Abs(sx - ox) <= 1 && Math.Abs(sy - oy) <= 1)
            {
                reachable += 1.0f;
            }
        }
        return reachable;
    }

    private static float CountEnemyNeighbors(Card[] board, Card card) => CountReadyAttackTargets(board, card);

    private static float CountPawnProgress(GameData data, Card[] board, string ownerId)
    {
        var pawns = 0.0f;
        var progressSum = 0.0f;
        var ownerIsPlayer1 = ownerId == data.Player1.Id;
        foreach (var card in board)
        {
            if (card.Owner.Id != ownerId) continue;
            if (RoleFromCard(card) != "Pawn" || !card.Unit.IsPlaced) continue;
            if (card.Unit.PosX < 0) continue;
            pawns += 1.0f;
            progressSum += ownerIsPlayer1
                ? NormalizeRatio(card.Unit.PosX, Board.BoardSize - 1)
                : NormalizeRatio((Board.BoardSize - 1) - card.Unit.PosX, Board.BoardSize - 1);
        }
        return pawns == 0.0f ? 0.0f : progressSum / pawns;
    }

    private static float CountPawnLastRank(GameData data, Card[] board, string ownerId)
    {
        var ownerIsPlayer1 = ownerId == data.Player1.Id;
        var lastRank = ownerIsPlayer1 ? Board.BoardSize - 1 : 0;
        var count = 0.0f;
        foreach (var card in board)
        {
            if (card.Owner.Id != ownerId) continue;
            if (RoleFromCard(card) != "Pawn" || !card.Unit.IsPlaced) continue;
            if (card.Unit.PosX == lastRank) count += 1.0f;
        }
        return count;
    }

    private static float[] CountSpecialEffectFeatures(GameData data, Card[] board, string ownerId)
    {
        var enemyId = ownerId == data.Player1.Id ? data.Player2.Id : data.Player1.Id;
        var enemyZone = ownerId == data.Player1.Id ? Board.BoardSize - 1 : 0;
        var enemyCards = board
            .Where(card => card.Unit.IsPlaced && card.Owner.Id == enemyId)
            .ToArray();
        var enemyPositions = enemyCards
            .Select(card => (card.Unit.PosX, card.Unit.PosY))
            .ToHashSet();

        float orNCount = 0.0f;
        float orNBestEnemyCount = 0.0f;
        float orNChainReady = 0.0f;
        float clBCount = 0.0f;
        float clBBestEnemyCount = 0.0f;
        float clPCount = 0.0f;
        float clPReady = 0.0f;

        foreach (var card in board)
        {
            if (!card.Unit.IsPlaced || card.Owner.Id != ownerId) continue;
            var cardId = card.Data.Id;
            var area = data.GetMoveArea(card);
            var enemyCount = area.Count(pos => enemyPositions.Contains(pos));

            if (cardId == "Or_N")
            {
                orNCount += 1.0f;
                orNBestEnemyCount = Math.Max(orNBestEnemyCount, enemyCount);
                var attack = card.Unit.Atk;
                var killableInArea = enemyCards.Any(enemy =>
                    area.Contains((enemy.Unit.PosX, enemy.Unit.PosY)) && enemy.Unit.Hp > 0 && enemy.Unit.Hp <= attack);
                if (enemyCount >= 2.0f && killableInArea)
                {
                    orNChainReady += 1.0f;
                }
            }
            else if (cardId == "Cl_B")
            {
                clBCount += 1.0f;
                clBBestEnemyCount = Math.Max(clBBestEnemyCount, enemyCount);
            }
            else if (cardId == "Cl_P")
            {
                clPCount += 1.0f;
                if (card.Unit.PosX == enemyZone)
                {
                    clPReady += 1.0f;
                }
            }
        }

        return new[]
        {
            NormalizeRatio(orNCount, 7.0f),
            NormalizeRatio(orNBestEnemyCount, 6.0f),
            NormalizeRatio(orNChainReady, 7.0f),
            NormalizeRatio(clBCount, 7.0f),
            NormalizeRatio(clBBestEnemyCount, 6.0f),
            NormalizeRatio(clPCount, 7.0f),
            NormalizeRatio(clPReady, 7.0f),
        };
    }

    private static float PawnPromotionReady(Card card, bool mirrorView, bool ownerIsSelf)
    {
        if (RoleFromCard(card) != "Pawn" || !card.Unit.IsPlaced)
        {
            return 0.0f;
        }

        var cx = ViewX(card.Unit.PosX, mirrorView);
        if (cx < 0)
        {
            return 0.0f;
        }

        var readyRank = ownerIsSelf ? Board.BoardSize - 1 : 0;
        return cx == readyRank ? 1.0f : 0.0f;
    }

    private static float PawnPromotionDistance(Card card, bool mirrorView, bool ownerIsSelf)
    {
        if (RoleFromCard(card) != "Pawn" || !card.Unit.IsPlaced)
        {
            return 0.0f;
        }

        var cx = ViewX(card.Unit.PosX, mirrorView);
        if (cx < 0)
        {
            return 0.0f;
        }

        var remaining = ownerIsSelf ? (Board.BoardSize - 1) - cx : cx;
        return NormalizeRatio(Math.Max(0, remaining), Board.BoardSize - 1);
    }

    private static (float attackMod, float hasMoveLock, float hasAttackLock, float timedStatusCount) StatusSummary(Card card)
    {
        var attackMod = 0.0f;
        var hasMoveLock = 0.0f;
        var hasAttackLock = 0.0f;
        var timedStatusCount = 0.0f;
        foreach (var status in card.Unit.Buffs)
        {
            timedStatusCount += 1.0f;
            if (status.Key == "TempAtk") attackMod += status.Value;
            else if (status.Key == "CantMove") hasMoveLock = 1.0f;
            else if (status.Key == "AttackLock") hasAttackLock = 1.0f;
        }
        return (attackMod, hasMoveLock, hasAttackLock, timedStatusCount);
    }

    private static string RoleFromCard(Card card) => card.Data.UnitType.ToString();

    private static int RoleRank(string role) => Array.IndexOf(RoleOrder, role) switch
    {
        -1 => 99,
        var idx => idx,
    };

    private static float[] RoleOneHot(string role) => RoleOrder.Select(name => name == role ? 1.0f : 0.0f).ToArray();

    private static float[] EffectOneHot(string effectId)
    {
        var bucket = EffectBuckets.Contains(effectId) && effectId != "Skill" ? effectId : "Skill";
        return EffectBuckets.Select(name => name == bucket ? 1.0f : 0.0f).ToArray();
    }

    private static float[] TargetOneHot(string targetType) => TargetBuckets.Select(name => name == targetType ? 1.0f : 0.0f).ToArray();

    private static float NormalizeRatio(float value, float scale) => scale == 0 ? 0.0f : value / scale;

    private static float NormalizePos(int value) => value < 0 ? -1.0f : value / 5.0f;

    private static bool ShouldMirrorPerspective(Card? ownLeader, Card? enemyLeader)
    {
        if (ownLeader is null || enemyLeader is null) return false;
        return ownLeader.Unit.PosX > enemyLeader.Unit.PosX;
    }

    private static int ViewX(int value, bool mirrorView)
    {
        if (value < 0) return value;
        return mirrorView ? Board.BoardSize - 1 - value : value;
    }

    private static int ViewY(int value, bool mirrorView)
    {
        if (value < 0) return value;
        return mirrorView ? Board.BoardSize - 1 - value : value;
    }

    private static float Distance(int x1, int y1, int x2, int y2)
    {
        if (x1 < 0 || y1 < 0 || x2 < 0 || y2 < 0) return -1.0f;
        return (Math.Abs(x1 - x2) + Math.Abs(y1 - y2)) / 10.0f;
    }
}
