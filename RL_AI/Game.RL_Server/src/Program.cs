using Game.RL_Server;
using SeaEngine.CardManager;

// ---------------------------------------------------------------------------
// Usage: RLServer <port> [card_data_path]
//
//   port           : TCP 리스닝 포트 (기본 9000)
//   card_data_path : Cards.csv 경로 (기본 <binary_dir>/db/Cards.csv)
//
// M개 서버를 병렬 실행하려면 M개의 포트로 각각 프로세스를 시작:
//   dotnet run -- 9000 &
//   dotnet run -- 9001 &
//   ...
// ---------------------------------------------------------------------------

int port = 9000;
string cardDataPath = "";

if (args.Length >= 1 && int.TryParse(args[0], out int parsedPort))
    port = parsedPort;

if (args.Length >= 2)
    cardDataPath = args[1];

if (string.IsNullOrEmpty(cardDataPath))
{
    var binDir = Path.GetDirectoryName(AppContext.BaseDirectory) ?? ".";
    cardDataPath = Path.GetFullPath(Path.Combine(binDir, "db", "Cards.csv"));
}

if (!File.Exists(cardDataPath))
{
    Console.Error.WriteLine($"[RLServer] Card data not found: {cardDataPath}");
    Console.Error.WriteLine("[RLServer] Usage: RLServer <port> <card_data_path>");
    return 1;
}

Console.WriteLine($"[RLServer] Loading cards from: {cardDataPath}");
var cardLines  = File.ReadAllLines(cardDataPath);
var cardLoader = new CardLoader(cardLines);

new RLServerApp(port, cardLoader).Run();
return 0;
