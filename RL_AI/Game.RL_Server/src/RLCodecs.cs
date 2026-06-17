using System.Runtime.InteropServices;
using System.Text;
using Game.Network;
using SeaEngine.RL;

namespace Game.RL_Server;

/// <summary>
/// IPacketCodec&lt;RlObservationFrame&gt; — RlObservationFrame 자체만 wire frame으로 직렬화.
///
/// Binary layout (little-endian):
///   result_byte   : u8
///   winner_id     : WriteString (i32 len + utf8)
///   turn          : i32
///   active_player : WriteString (i32 len + utf8)
///   state_vec_len : i32
///   state_vec     : f32 * state_vec_len      (MemoryMarshal zero-copy)
///   action_count  : i32
///   [per action]
///     uid         : WriteString (i32 len + utf8)   e.g. "A001"
///     effect_id   : u8 len + utf8                  e.g. "TurnEnd"
///     source      : WriteString (i32 len + utf8)
///     target_type : WriteString (i32 len + utf8)
///     target_guid : WriteString (i32 len + utf8)
///     target_guid2: WriteString (i32 len + utf8)
///     pos_x       : i32
///     pos_y       : i32
///     feat_count  : i16
///     feat_vec    : f32 * feat_count               (MemoryMarshal zero-copy)
///   player_count  : i32
///   [per player]
///     id          : WriteString (i32 len + utf8)
///     hand_count   : i32
///     deck_count   : i32
///     trash_count  : i32
///     [per hand card]
///       uid       : WriteString (i32 len + utf8)
///       card_id   : WriteString (i32 len + utf8)
///       name      : WriteString (i32 len + utf8)
///   board_count   : i32
///   [per board card]
///     uid         : WriteString (i32 len + utf8)
///     card_id     : WriteString (i32 len + utf8)
///     name        : WriteString (i32 len + utf8)
///     owner_id    : WriteString (i32 len + utf8)
///     role        : WriteString (i32 len + utf8)
///     atk         : i32
///     effective_atk: i32
///     hp          : i32
///     max_hp      : i32
///     is_placed   : u8
///     is_moved    : u8
///     is_attacked : u8
///     pos_x       : i32
///     pos_y       : i32
///     status_count: i32
///     [per status]
///       type      : WriteString (i32 len + utf8)
///       value     : i32
/// </summary>
public sealed class SnapshotCodec : IPacketCodec<RlObservationFrame>
{
    public static readonly SnapshotCodec Instance = new();

    private static string DecodeResult(byte result) => result switch
    {
        RLProto.RESULT_P1_WIN => "Player1Win",
        RLProto.RESULT_P2_WIN => "Player2Win",
        RLProto.RESULT_DRAW    => "Draw",
        _                      => "Ongoing",
    };

    private static void Require(bool condition, string message)
    {
        if (!condition)
        {
            throw new InvalidOperationException(message);
        }
    }

    private static int WireStringSize(string? value)
        => 4 + Encoding.UTF8.GetByteCount(value ?? string.Empty);

    public int GetSize(RlObservationFrame f)
    {
        int size = 1;                                                        // result_byte
        size += WireStringSize(f.WinnerId);                                  // winner_id
        size += 4;                                                           // turn i32
        size += WireStringSize(f.ActivePlayerId);                            // active_player

        size += 4 + f.StateVector.Length * 4;                               // state_vec

        size += 4;                                                           // action_count
        int n = Math.Min(f.Actions.Length, f.ActionFeatureVectors.Length);
        for (int i = 0; i < n; i++)
        {
            size += WireStringSize(f.Actions[i].Uid);                        // uid
            size += 1 + Encoding.UTF8.GetByteCount(f.Actions[i].EffectId ?? ""); // eff_id (1-byte len)
            size += WireStringSize(f.Actions[i].Source);                    // source
            size += WireStringSize(f.Actions[i].TargetType);                // target type
            size += WireStringSize(f.Actions[i].TargetGuid);                // target guid
            size += WireStringSize(f.Actions[i].TargetGuid2);               // target guid2
            size += 4; // pos_x
            size += 4; // pos_y
            size += 2 + f.ActionFeatureVectors[i].Length * 4;                    // feat_vec
        }
        size += 4;                                                           // player_count
        foreach (var player in f.Players)
        {
            size += WireStringSize(player.Id);
            size += 4; // hand_count
            size += 4; // deck_count
            size += 4; // trash_count
            size += 4; // hand_size
            foreach (var card in player.Hand)
            {
                size += WireStringSize(card.Uid);
                size += WireStringSize(card.CardId);
                size += WireStringSize(card.Name);
            }
        }
        size += 4;                                                           // board_count
        foreach (var card in f.Board)
        {
            size += WireStringSize(card.Uid);
            size += WireStringSize(card.CardId);
            size += WireStringSize(card.Name);
            size += WireStringSize(card.OwnerId);
            size += WireStringSize(card.Role);
            size += 4; // atk
            size += 4; // effective_atk
            size += 4; // hp
            size += 4; // max_hp
            size += 1; // is_placed
            size += 1; // is_moved
            size += 1; // is_attacked
            size += 4; // pos_x
            size += 4; // pos_y
            size += 4; // status_count
            foreach (var status in card.Statuses)
            {
                size += WireStringSize(status.Type);
                size += 4; // value
            }
        }
        return size;
    }

    public void Write(ref PacketWriter writer, RlObservationFrame f)
    {
        writer.WriteByte(RLProto.EncodeResult(f.Result));
        writer.WriteString(f.WinnerId     ?? "");
        writer.WriteInt32(f.Turn);
        writer.WriteString(f.ActivePlayerId ?? "");

        // state vector — MemoryMarshal.Cast: float[] → byte span, 복사 없음
        var svSpan = f.StateVector.AsSpan();
        writer.WriteInt32(svSpan.Length);
        writer.WriteBytes(MemoryMarshal.Cast<float, byte>(svSpan));

        // actions
        int n = Math.Min(f.Actions.Length, f.ActionFeatureVectors.Length);
        writer.WriteInt32(n);

        for (int i = 0; i < n; i++)
        {
            // uid (WriteString: i32 len + utf8)
            writer.WriteString(f.Actions[i].Uid ?? "");

            // effect_id (1-byte len + utf8)
            // PacketWriter가 ref struct이므로 stackalloc span 전달 불가 → 힙 배열 사용
            var effBytes = Encoding.UTF8.GetBytes(f.Actions[i].EffectId ?? "");
            writer.WriteByte((byte)effBytes.Length);
            writer.WriteBytes(effBytes);

            writer.WriteString(f.Actions[i].Source ?? "");
            writer.WriteString(f.Actions[i].TargetType ?? "");
            writer.WriteString(f.Actions[i].TargetGuid ?? "");
            writer.WriteString(f.Actions[i].TargetGuid2 ?? "");
            writer.WriteInt32(f.Actions[i].PosX);
            writer.WriteInt32(f.Actions[i].PosY);

            // feature vector — zero-copy
            var fvSpan = f.ActionFeatureVectors[i].AsSpan();
            writer.WriteInt16((short)fvSpan.Length);
            writer.WriteBytes(MemoryMarshal.Cast<float, byte>(fvSpan));
        }

        writer.WriteInt32(f.Players.Length);
        foreach (var player in f.Players)
        {
            writer.WriteString(player.Id ?? "");
            writer.WriteInt32(player.HandCount);
            writer.WriteInt32(player.DeckCount);
            writer.WriteInt32(player.TrashCount);
            writer.WriteInt32(player.Hand.Length);
            foreach (var card in player.Hand)
            {
                writer.WriteString(card.Uid ?? "");
                writer.WriteString(card.CardId ?? "");
                writer.WriteString(card.Name ?? "");
            }
        }

        writer.WriteInt32(f.Board.Length);
        foreach (var card in f.Board)
        {
            writer.WriteString(card.Uid ?? "");
            writer.WriteString(card.CardId ?? "");
            writer.WriteString(card.Name ?? "");
            writer.WriteString(card.OwnerId ?? "");
            writer.WriteString(card.Role ?? "");
            writer.WriteInt32(card.Atk);
            writer.WriteInt32(card.EffectiveAtk);
            writer.WriteInt32(card.Hp);
            writer.WriteInt32(card.MaxHp);
            writer.WriteByte(card.IsPlaced ? (byte)1 : (byte)0);
            writer.WriteByte(card.IsMoved ? (byte)1 : (byte)0);
            writer.WriteByte(card.IsAttacked ? (byte)1 : (byte)0);
            writer.WriteInt32(card.PosX);
            writer.WriteInt32(card.PosY);
            writer.WriteInt32(card.Statuses.Length);
            foreach (var status in card.Statuses)
            {
                writer.WriteString(status.Type ?? "");
                writer.WriteInt32(status.Value);
            }
        }
    }

    public RlObservationFrame Read(ref PacketReader reader)
    {
        var result = DecodeResult(reader.ReadByte());
        var winnerId = reader.ReadString();
        var turn = reader.ReadInt32();
        var activePlayerId = reader.ReadString();

        var stateVectorLength = reader.ReadInt32();
        Require(stateVectorLength >= 0, $"Invalid state vector length: {stateVectorLength}");
        var stateVector = new float[stateVectorLength];
        for (int i = 0; i < stateVectorLength; i++)
        {
            stateVector[i] = BitConverter.Int32BitsToSingle(reader.ReadInt32());
        }

        var actionCount = reader.ReadInt32();
        Require(actionCount >= 0, $"Invalid action count: {actionCount}");
        var actions = new RlActionView[actionCount];
        var actionFeatureVectors = new float[actionCount][];
        for (int i = 0; i < actionCount; i++)
        {
            var uid = reader.ReadString();
            var effectLen = reader.ReadByte();
            var effectId = Encoding.UTF8.GetString(reader.ReadBytes(effectLen));
            var source = reader.ReadString();
            var targetType = reader.ReadString();
            var targetGuid = reader.ReadString();
            var targetGuid2 = reader.ReadString();
            var posX = reader.ReadInt32();
            var posY = reader.ReadInt32();
            var featCount = reader.ReadInt16();
            Require(featCount >= 0, $"Invalid action feature count: {featCount}");
            var featVec = new float[featCount];
            for (int j = 0; j < featCount; j++)
            {
                featVec[j] = BitConverter.Int32BitsToSingle(reader.ReadInt32());
            }

            actions[i] = new RlActionView(
                uid,
                effectId,
                source,
                targetType,
                targetGuid,
                targetGuid2,
                posX,
                posY,
                string.Empty
            );
            actionFeatureVectors[i] = featVec;
        }

        var playerCount = reader.ReadInt32();
        Require(playerCount >= 0, $"Invalid player count: {playerCount}");
        var players = new RlPlayerView[playerCount];
        for (int i = 0; i < playerCount; i++)
        {
            var id = reader.ReadString();
            var handCount = reader.ReadInt32();
            var deckCount = reader.ReadInt32();
            var trashCount = reader.ReadInt32();
            var handSize = reader.ReadInt32();
            Require(handCount >= 0, $"Invalid hand count: {handCount}");
            Require(deckCount >= 0, $"Invalid deck count: {deckCount}");
            Require(trashCount >= 0, $"Invalid trash count: {trashCount}");
            Require(handSize >= 0, $"Invalid hand size: {handSize}");
            var hand = new RlCardView[handSize];
            for (int j = 0; j < handSize; j++)
            {
                var uid = reader.ReadString();
                var cardId = reader.ReadString();
                var name = reader.ReadString();
                hand[j] = new RlCardView(
                    uid,
                    cardId,
                    name,
                    string.Empty,
                    string.Empty,
                    0,
                    0,
                    0,
                    0,
                    false,
                    false,
                    false,
                    -1,
                    -1,
                    Array.Empty<RlStatusView>()
                );
            }

            players[i] = new RlPlayerView(id, handCount, deckCount, trashCount, hand);
        }

        var boardCount = reader.ReadInt32();
        Require(boardCount >= 0, $"Invalid board count: {boardCount}");
        var board = new RlCardView[boardCount];
        for (int i = 0; i < boardCount; i++)
        {
            var uid = reader.ReadString();
            var cardId = reader.ReadString();
            var name = reader.ReadString();
            var ownerId = reader.ReadString();
            var role = reader.ReadString();
            var atk = reader.ReadInt32();
            var effectiveAtk = reader.ReadInt32();
            var hp = reader.ReadInt32();
            var maxHp = reader.ReadInt32();
            var isPlaced = reader.ReadByte() != 0;
            var isMoved = reader.ReadByte() != 0;
            var isAttacked = reader.ReadByte() != 0;
            var posX = reader.ReadInt32();
            var posY = reader.ReadInt32();
            var statusCount = reader.ReadInt32();
            Require(statusCount >= 0, $"Invalid status count: {statusCount}");
            var statuses = new RlStatusView[statusCount];
            for (int j = 0; j < statusCount; j++)
            {
                var type = reader.ReadString();
                var value = reader.ReadInt32();
                statuses[j] = new RlStatusView(type, value);
            }

            board[i] = new RlCardView(
                uid,
                cardId,
                name,
                ownerId,
                role,
                atk,
                effectiveAtk,
                hp,
                maxHp,
                isPlaced,
                isMoved,
                isAttacked,
                posX,
                posY,
                statuses
            );
        }

        var frame = new RlObservationFrame(
            turn,
            activePlayerId,
            result,
            winnerId,
            players,
            board,
            actions,
            stateVector,
            actionFeatureVectors
        );

        return frame;
    }
}

/// <summary>
/// MCTS 배치 응답: 각 후보의 최소 leaf 정보만 담는다.
/// Layout:
///   count:i32
///   [per frame]
///     frame_size:i32
///     frame_payload: leaf payload
/// </summary>
public sealed class MctsBatchResult
{
    public SeaEngine.RL.RlMctsLeafFrame[] Frames { get; }

    public MctsBatchResult(SeaEngine.RL.RlMctsLeafFrame[] frames)
    {
        Frames = frames;
    }
}

public sealed class MctsBatchResultCodec : IPacketCodec<MctsBatchResult>
{
    public static readonly MctsBatchResultCodec Instance = new();

    private static string DecodeResult(byte result) => result switch
    {
        RLProto.RESULT_P1_WIN => "Player1Win",
        RLProto.RESULT_P2_WIN => "Player2Win",
        RLProto.RESULT_DRAW    => "Draw",
        _                      => "Ongoing",
    };

    public int GetSize(MctsBatchResult data)
    {
        int size = 4; // count
        foreach (var frame in data.Frames)
        {
            size += 4;
            size += GetLeafSize(frame);
        }
        return size;
    }

    public void Write(ref PacketWriter writer, MctsBatchResult data)
    {
        writer.WriteInt32(data.Frames.Length);
        foreach (var frame in data.Frames)
        {
            writer.WriteInt32(GetLeafSize(frame));
            WriteLeafFrame(ref writer, frame);
        }
    }

    public MctsBatchResult Read(ref PacketReader reader)
    {
        var count = reader.ReadInt32();
        if (count < 0)
            throw new InvalidOperationException($"Invalid batch count: {count}");

        var frames = new SeaEngine.RL.RlMctsLeafFrame[count];
        for (int i = 0; i < count; i++)
        {
            var frameSize = reader.ReadInt32();
            if (frameSize < 0)
                throw new InvalidOperationException($"Invalid frame size: {frameSize}");

            var frameBytes = reader.ReadBytes(frameSize);
            var frameReader = new PacketReader(frameBytes);
            frames[i] = ReadLeafFrame(ref frameReader);
        }

        return new MctsBatchResult(frames);
    }

    private static int WireStringSize(string? value)
        => 4 + Encoding.UTF8.GetByteCount(value ?? string.Empty);

    private static int GetLeafSize(SeaEngine.RL.RlMctsLeafFrame f)
    {
        int size = 1;                               // result
        size += WireStringSize(f.WinnerId);         // winner_id
        size += 4 + f.StateVector.Length * 4;        // state_vector
        return size;
    }

    private static void WriteLeafFrame(ref PacketWriter writer, SeaEngine.RL.RlMctsLeafFrame frame)
    {
        writer.WriteByte(RLProto.EncodeResult(frame.Result));
        writer.WriteString(frame.WinnerId ?? "");

        var svSpan = frame.StateVector.AsSpan();
        writer.WriteInt32(svSpan.Length);
        writer.WriteBytes(MemoryMarshal.Cast<float, byte>(svSpan));
    }

    private static SeaEngine.RL.RlMctsLeafFrame ReadLeafFrame(ref PacketReader reader)
    {
        var result = DecodeResult(reader.ReadByte());
        var winnerId = reader.ReadString();

        var stateVectorLength = reader.ReadInt32();
        if (stateVectorLength < 0)
            throw new InvalidOperationException($"Invalid state vector length: {stateVectorLength}");
        var stateVector = new float[stateVectorLength];
        for (int i = 0; i < stateVectorLength; i++)
        {
            stateVector[i] = BitConverter.Int32BitsToSingle(reader.ReadInt32());
        }
        return new SeaEngine.RL.RlMctsLeafFrame(
            result,
            winnerId,
            stateVector
        );
    }
}
