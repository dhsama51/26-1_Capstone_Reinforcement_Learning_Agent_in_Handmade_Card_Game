using Game.Network;
using Game.RL_Server;
using SeaEngine.RL;

static void AssertEqual<T>(T expected, T actual, string name)
{
    if (!EqualityComparer<T>.Default.Equals(expected, actual))
    {
        throw new InvalidOperationException($"{name} mismatch: expected={expected}, actual={actual}");
    }
}

static void AssertSequenceEqual<T>(IReadOnlyList<T> expected, IReadOnlyList<T> actual, string name)
{
    if (expected.Count != actual.Count)
    {
        throw new InvalidOperationException($"{name} count mismatch: expected={expected.Count}, actual={actual.Count}");
    }

    var comparer = EqualityComparer<T>.Default;
    for (int i = 0; i < expected.Count; i++)
    {
        if (!comparer.Equals(expected[i], actual[i]))
        {
            throw new InvalidOperationException($"{name}[{i}] mismatch: expected={expected[i]}, actual={actual[i]}");
        }
    }
}

static RlObservationFrame BuildFrame()
{
    var frame = new RlObservationFrame(
        Turn: 1234,
        ActivePlayerId: "P2",
        Result: "Player1Win",
        WinnerId: "P1",
        Players: new[]
        {
            new RlPlayerView(
                "P1",
                2,
                10,
                1,
                new[]
                {
                    new RlCardView("H1", "Or_P", "Orange Pawn", "P1", "Pawn", 1, 2, 1, 1, false, false, false, -1, -1, Array.Empty<RlStatusView>()),
                    new RlCardView("H2", "Or_R", "Orange Rook", "P1", "Rook", 4, 4, 5, 5, false, false, false, -1, -1, Array.Empty<RlStatusView>()),
                }
            ),
            new RlPlayerView(
                "P2",
                3,
                9,
                0,
                new[]
                {
                    new RlCardView("H3", "Cl_P", "Charles Pawn", "P2", "Pawn", 1, 1, 1, 1, false, false, false, -1, -1, Array.Empty<RlStatusView>()),
                }
            ),
        },
        Board: new[]
        {
            new RlCardView(
                "B1", "Or_L", "Orange Leader", "P1", "Leader", 3, 5, 9, 9, true, false, false, 0, 2,
                new[] { new RlStatusView("TempAtk", 2) }
            ),
            new RlCardView(
                "B2", "Cl_L", "Charles Leader", "P2", "Leader", 4, 4, 8, 9, true, true, true, 5, 3,
                new[] { new RlStatusView("Poison", 1), new RlStatusView("Shield", 2) }
            ),
        },
        Actions: new[]
        {
            new RlActionView("A1", "DefaultMove", "B1", "Cell", "", "", 1, 2, "move"),
            new RlActionView("A2", "DefaultAttack", "B1", "Unit", "B2", "", 5, 3, "attack"),
        },
        StateVector: Enumerable.Range(0, 57 + 14 * 41 + 7 * 20).Select(i => i / 1000.0f).ToArray(),
        ActionFeatureVectors: new[]
        {
            Enumerable.Range(0, 108).Select(i => i / 100.0f).ToArray(),
            Enumerable.Range(0, 108).Select(i => (i + 1) / 100.0f).ToArray(),
        }
    );

    return frame;
}

var frame = BuildFrame();
var codec = SnapshotCodec.Instance;
var expectedSize = codec.GetSize(frame);
var buffer = new byte[expectedSize + 4096];
var writer = new PacketWriter(buffer);

codec.Write(ref writer, frame);

if (writer.Offset != expectedSize)
{
    throw new InvalidOperationException($"encoded size mismatch: expected={expectedSize}, actual={writer.Offset}");
}

var reader = new PacketReader(buffer.AsSpan(0, writer.Offset));
var decoded = codec.Read(ref reader);

if (reader.Remain != 0)
{
    throw new InvalidOperationException($"reader did not consume all bytes: remain={reader.Remain}");
}

AssertEqual(frame.Turn, decoded.Turn, nameof(frame.Turn));
AssertEqual(frame.ActivePlayerId, decoded.ActivePlayerId, nameof(frame.ActivePlayerId));
AssertEqual(frame.Result, decoded.Result, nameof(frame.Result));
AssertEqual(frame.WinnerId, decoded.WinnerId, nameof(frame.WinnerId));

AssertEqual(frame.Players.Length, decoded.Players.Length, nameof(frame.Players));
for (int i = 0; i < frame.Players.Length; i++)
{
    var lhs = frame.Players[i];
    var rhs = decoded.Players[i];
    AssertEqual(lhs.Id, rhs.Id, $"Players[{i}].Id");
    AssertEqual(lhs.HandCount, rhs.HandCount, $"Players[{i}].HandCount");
    AssertEqual(lhs.DeckCount, rhs.DeckCount, $"Players[{i}].DeckCount");
    AssertEqual(lhs.TrashCount, rhs.TrashCount, $"Players[{i}].TrashCount");
    AssertEqual(lhs.Hand.Length, rhs.Hand.Length, $"Players[{i}].Hand.Length");
    for (int j = 0; j < lhs.Hand.Length; j++)
    {
        AssertEqual(lhs.Hand[j].Uid, rhs.Hand[j].Uid, $"Players[{i}].Hand[{j}].Uid");
        AssertEqual(lhs.Hand[j].CardId, rhs.Hand[j].CardId, $"Players[{i}].Hand[{j}].CardId");
        AssertEqual(lhs.Hand[j].Name, rhs.Hand[j].Name, $"Players[{i}].Hand[{j}].Name");
    }
}

AssertEqual(frame.Board.Length, decoded.Board.Length, nameof(frame.Board));
for (int i = 0; i < frame.Board.Length; i++)
{
    var lhs = frame.Board[i];
    var rhs = decoded.Board[i];
    AssertEqual(lhs.Uid, rhs.Uid, $"Board[{i}].Uid");
    AssertEqual(lhs.CardId, rhs.CardId, $"Board[{i}].CardId");
    AssertEqual(lhs.Name, rhs.Name, $"Board[{i}].Name");
    AssertEqual(lhs.OwnerId, rhs.OwnerId, $"Board[{i}].OwnerId");
    AssertEqual(lhs.Role, rhs.Role, $"Board[{i}].Role");
    AssertEqual(lhs.Atk, rhs.Atk, $"Board[{i}].Atk");
    AssertEqual(lhs.EffectiveAtk, rhs.EffectiveAtk, $"Board[{i}].EffectiveAtk");
    AssertEqual(lhs.Hp, rhs.Hp, $"Board[{i}].Hp");
    AssertEqual(lhs.MaxHp, rhs.MaxHp, $"Board[{i}].MaxHp");
    AssertEqual(lhs.IsPlaced, rhs.IsPlaced, $"Board[{i}].IsPlaced");
    AssertEqual(lhs.IsMoved, rhs.IsMoved, $"Board[{i}].IsMoved");
    AssertEqual(lhs.IsAttacked, rhs.IsAttacked, $"Board[{i}].IsAttacked");
    AssertEqual(lhs.PosX, rhs.PosX, $"Board[{i}].PosX");
    AssertEqual(lhs.PosY, rhs.PosY, $"Board[{i}].PosY");
    AssertEqual(lhs.Statuses.Length, rhs.Statuses.Length, $"Board[{i}].Statuses.Length");
    for (int j = 0; j < lhs.Statuses.Length; j++)
    {
        AssertEqual(lhs.Statuses[j].Type, rhs.Statuses[j].Type, $"Board[{i}].Statuses[{j}].Type");
        AssertEqual(lhs.Statuses[j].Value, rhs.Statuses[j].Value, $"Board[{i}].Statuses[{j}].Value");
    }
}

AssertEqual(frame.Actions.Length, decoded.Actions.Length, nameof(frame.Actions));
for (int i = 0; i < frame.Actions.Length; i++)
{
    var lhs = frame.Actions[i];
    var rhs = decoded.Actions[i];
    AssertEqual(lhs.Uid, rhs.Uid, $"Actions[{i}].Uid");
    AssertEqual(lhs.EffectId, rhs.EffectId, $"Actions[{i}].EffectId");
    AssertEqual(lhs.Source, rhs.Source, $"Actions[{i}].Source");
    AssertEqual(lhs.TargetType, rhs.TargetType, $"Actions[{i}].TargetType");
    AssertEqual(lhs.TargetGuid, rhs.TargetGuid, $"Actions[{i}].TargetGuid");
    AssertEqual(lhs.TargetGuid2, rhs.TargetGuid2, $"Actions[{i}].TargetGuid2");
    AssertEqual(lhs.PosX, rhs.PosX, $"Actions[{i}].PosX");
    AssertEqual(lhs.PosY, rhs.PosY, $"Actions[{i}].PosY");
}

AssertSequenceEqual(frame.StateVector, decoded.StateVector, nameof(frame.StateVector));
AssertEqual(frame.ActionFeatureVectors.Length, decoded.ActionFeatureVectors.Length, nameof(frame.ActionFeatureVectors));
for (int i = 0; i < frame.ActionFeatureVectors.Length; i++)
{
    AssertSequenceEqual(frame.ActionFeatureVectors[i], decoded.ActionFeatureVectors[i], $"ActionFeatureVectors[{i}]");
}

Console.WriteLine("SnapshotCodec roundtrip OK");
