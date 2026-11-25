import 'package:find_bee_queen/queen_judge.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  test('直近 N フレーム中 K フレーム以上で「発見」', () {
    final j = QueenJudge(window: 10, minHits: 4);
    for (var i = 0; i < 3; i++) {
      expect(j.add(true), isFalse);
    }
    expect(j.add(true), isTrue); // 4 回目で発見
    for (var i = 0; i < 6; i++) {
      j.add(false); // 10 フレーム中 4 回のまま
    }
    expect(j.found, isTrue);
    expect(j.add(false), isFalse); // 最初の検出が窓から外れて 3 回に
    j.reset();
    expect(j.hitCount, 0);
  });

  test('1 フレームだけの誤検出では発見にならない', () {
    final j = QueenJudge(window: 10, minHits: 4);
    final seq = [false, true, false, false, true, false, false, false, true, false];
    expect(seq.map(j.add).any((f) => f), isFalse);
  });

  test('自動保存は間隔をあける', () {
    final p = SnapPolicy(cooldown: const Duration(seconds: 10));
    final t0 = DateTime(2026, 1, 1, 12);
    expect(p.shouldSnap(t0), isTrue);
    p.markSaved(t0);
    expect(p.shouldSnap(t0.add(const Duration(seconds: 9))), isFalse);
    expect(p.shouldSnap(t0.add(const Duration(seconds: 10))), isTrue);
  });
}
