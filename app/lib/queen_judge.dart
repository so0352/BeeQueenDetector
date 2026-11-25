import 'dart:collection';

/// 「女王蜂発見」の判定（PC 版 detect_stream.py と同じ考え方）。
///
/// 1 フレームの誤検出で騒がないよう、直近 [window] フレームのうち
/// [minHits] フレーム以上で女王蜂が検出されたら「発見」とみなす。
class QueenJudge {
  QueenJudge({this.window = 10, this.minHits = 4});

  int window;
  int minHits;
  final _hits = ListQueue<bool>();

  /// 1 フレーム分の結果（女王蜂が 1 匹以上検出されたか）を加え、発見状態を返す。
  bool add(bool hit) {
    _hits.addLast(hit);
    while (_hits.length > window) {
      _hits.removeFirst();
    }
    return found;
  }

  bool get found => hitCount >= minHits;

  int get hitCount => _hits.where((h) => h).length;

  void reset() => _hits.clear();
}

/// 自動保存の間隔制御（「発見」状態が続いても [cooldown] ごとに 1 枚だけ保存する）。
class SnapPolicy {
  SnapPolicy({this.cooldown = const Duration(seconds: 10)});

  Duration cooldown;
  DateTime? _last;

  bool shouldSnap(DateTime now) => _last == null || now.difference(_last!) >= cooldown;

  void markSaved(DateTime now) => _last = now;
}
