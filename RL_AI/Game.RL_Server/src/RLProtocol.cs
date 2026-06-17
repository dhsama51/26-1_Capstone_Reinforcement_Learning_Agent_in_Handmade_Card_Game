namespace Game.RL_Server;

/// <summary>
/// 공유 상수. Python의 train_client.py와 반드시 일치해야 함.
/// </summary>
public static class RLProto
{
    // Handler IDs (HANDLER_GAME_MESSAGE=6, HANDLER_PEER_ENTRANCE=7 과 충돌하지 않음)
    public const int HANDLER_INIT       = 1;
    public const int HANDLER_STEP       = 2;
    public const int HANDLER_MCTS_RESET = 3;
    public const int HANDLER_MCTS_STEP  = 4;
    public const int HANDLER_MCTS_BATCH = 5;

    // Snapshot result byte values
    public const byte RESULT_ONGOING = 0;
    public const byte RESULT_P1_WIN  = 1;
    public const byte RESULT_P2_WIN  = 2;
    public const byte RESULT_DRAW    = 3;

    public static byte EncodeResult(string result) => result switch
    {
        "Player1Win" => RESULT_P1_WIN,
        "Player2Win" => RESULT_P2_WIN,
        "Draw"       => RESULT_DRAW,
        _            => RESULT_ONGOING,
    };
}
