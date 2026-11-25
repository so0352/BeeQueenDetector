import 'package:shared_preferences/shared_preferences.dart';

/// 端末に保存する設定（アプリを閉じても残る）。
class AppSettings {
  AppSettings({
    this.conf = 0.25,
    this.window = 10,
    this.minHits = 4,
    this.cooldownSec = 10,
    this.autoSave = true,
    this.vibrate = true,
  });

  /// 確信度のしきい値。PC のテストで 0.25: 検出 92%・誤検出 2%/画像。
  double conf;

  /// 判定に使う直近フレーム数。
  int window;

  /// 「発見」とみなす検出フレーム数。
  int minHits;

  /// 自動保存の間隔（秒）。
  int cooldownSec;

  bool autoSave;
  bool vibrate;

  static Future<AppSettings> load() async {
    final p = await SharedPreferences.getInstance();
    final d = AppSettings();
    return AppSettings(
      conf: p.getDouble('conf') ?? d.conf,
      window: p.getInt('window') ?? d.window,
      minHits: p.getInt('minHits') ?? d.minHits,
      cooldownSec: p.getInt('cooldownSec') ?? d.cooldownSec,
      autoSave: p.getBool('autoSave') ?? d.autoSave,
      vibrate: p.getBool('vibrate') ?? d.vibrate,
    );
  }

  Future<void> save() async {
    final p = await SharedPreferences.getInstance();
    await p.setDouble('conf', conf);
    await p.setInt('window', window);
    await p.setInt('minHits', minHits);
    await p.setInt('cooldownSec', cooldownSec);
    await p.setBool('autoSave', autoSave);
    await p.setBool('vibrate', vibrate);
  }
}
