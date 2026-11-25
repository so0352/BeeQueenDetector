import 'package:shared_preferences/shared_preferences.dart';

/// 端末に保存する設定（アプリを閉じても残る）。
class AppSettings {
  AppSettings({
    this.conf = 0.15,
    this.window = 20,
    this.minHits = 1,
    this.cooldownSec = 9,
    this.autoSave = false,
    this.vibrate = false,
  });

  /// 確信度のしきい値。低めにして見逃しを減らす（誤検出は「発見」の判定と目視で見分ける）。
  double conf;

  /// 判定に使う直近フレーム数。
  int window;

  /// 「発見」とみなす検出フレーム数。
  int minHits;

  /// 自動保存の間隔（秒）。
  int cooldownSec;

  bool autoSave;
  bool vibrate;

  /// 以前（変えていない項目も保存していた頃）の初期値。
  /// それと同じ値は「変えていない」とみなして捨て、新しい初期値を使う。
  static const _oldDefaults = <String, Object>{
    'conf': 0.25,
    'window': 10,
    'minHits': 4,
    'cooldownSec': 10,
    'autoSave': true,
    'vibrate': true,
  };
  static const _version = 2;

  static Future<AppSettings> load() async {
    final p = await SharedPreferences.getInstance();
    if ((p.getInt('settingsVersion') ?? 1) < _version) {
      for (final e in _oldDefaults.entries) {
        if (p.get(e.key) == e.value) await p.remove(e.key);
      }
      await p.setInt('settingsVersion', _version);
    }
    final d = AppSettings();
    final s = AppSettings(
      conf: p.getDouble('conf') ?? d.conf,
      window: p.getInt('window') ?? d.window,
      minHits: p.getInt('minHits') ?? d.minHits,
      cooldownSec: p.getInt('cooldownSec') ?? d.cooldownSec,
      autoSave: p.getBool('autoSave') ?? d.autoSave,
      vibrate: p.getBool('vibrate') ?? d.vibrate,
    );
    if (s.minHits > s.window) s.minHits = s.window;
    return s;
  }

  /// 初期値から変えた項目だけを保存する（初期値を変えたとき、触っていない項目は新しい初期値になる）。
  Future<void> save() async {
    final p = await SharedPreferences.getInstance();
    final d = AppSettings();
    Future<void> put<T>(String key, T value, T def, Future<bool> Function(String, T) set) =>
        value == def ? p.remove(key) : set(key, value);
    await put('conf', conf, d.conf, p.setDouble);
    await put('window', window, d.window, p.setInt);
    await put('minHits', minHits, d.minHits, p.setInt);
    await put('cooldownSec', cooldownSec, d.cooldownSec, p.setInt);
    await put('autoSave', autoSave, d.autoSave, p.setBool);
    await put('vibrate', vibrate, d.vibrate, p.setBool);
    await p.setInt('settingsVersion', _version);
  }
}
