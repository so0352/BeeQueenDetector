import 'dart:async';
import 'dart:io' show Platform;

import 'package:flutter/services.dart';
import 'package:gal/gal.dart';
import 'package:material_ui/material_ui.dart';
import 'package:ultralytics_yolo/ultralytics_yolo.dart';
import 'package:wakelock_plus/wakelock_plus.dart';

import 'queen_judge.dart';
import 'settings.dart';

/// 端末ごとのモデル（export_mobile.py で書き出し、pubspec.yaml で端末ごとに同梱）。
String get modelPath =>
    Platform.isIOS ? 'assets/models/queen.mlpackage.zip' : 'assets/models/queen.tflite';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(const FindBeeQueenApp());
}

class FindBeeQueenApp extends StatelessWidget {
  const FindBeeQueenApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: '女王蜂さがし',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.amber, brightness: Brightness.dark),
        useMaterial3: true,
      ),
      home: const DetectScreen(),
    );
  }
}

class DetectScreen extends StatefulWidget {
  const DetectScreen({super.key});

  @override
  State<DetectScreen> createState() => _DetectScreenState();
}

/// 画面上部の表示用（毎フレーム更新されるので setState ではなく ValueNotifier で更新する）。
class _Status {
  const _Status({this.found = false, this.hits = 0, this.maxConf = 0, this.fps = 0, this.ms = 0});
  final bool found;
  final int hits;
  final double maxConf;
  final double fps;
  final double ms;
}

class _DetectScreenState extends State<DetectScreen> with WidgetsBindingObserver {
  final _controller = YOLOViewController();
  final _status = ValueNotifier(const _Status());
  final _zoom = ValueNotifier(1.0);
  AppSettings? _settings;
  late QueenJudge _judge;
  late SnapPolicy _snap;
  String? _modelError;
  bool _modelReady = false;
  bool _torch = false;
  bool _saving = false;
  int _savedCount = 0;
  double _baseZoom = 1.0;
  double _fps = 0;
  double _ms = 0;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    WakelockPlus.enable(); // 巣板を映している間に画面が消えないように
    AppSettings.load().then((s) {
      if (!mounted) return;
      setState(() {
        _settings = s;
        _judge = QueenJudge(window: s.window, minHits: s.minHits);
        _snap = SnapPolicy(cooldown: Duration(seconds: s.cooldownSec));
      });
    });
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    WakelockPlus.disable();
    _status.dispose();
    _zoom.dispose();
    super.dispose();
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    // アプリから戻ったときに発見状態を引きずらない
    if (state == AppLifecycleState.resumed && _settings != null) {
      _judge.reset();
      _status.value = const _Status();
    }
  }

  void _onResult(List<YOLOResult> results) {
    final s = _settings;
    if (s == null) return;
    final queens = results.where((r) => r.confidence >= s.conf).toList();
    final wasFound = _judge.found;
    final found = _judge.add(queens.isNotEmpty);
    final maxConf = queens.fold<double>(0, (m, r) => r.confidence > m ? r.confidence : m);
    _status.value = _Status(found: found, hits: _judge.hitCount, maxConf: maxConf, fps: _fps, ms: _ms);

    if (found && !wasFound && s.vibrate) {
      HapticFeedback.heavyImpact();
    }
    final now = DateTime.now();
    if (found && s.autoSave && !_saving && _snap.shouldSnap(now)) {
      _snap.markSaved(now);
      _save(auto: true);
    }
  }

  void _onMetrics(YOLOPerformanceMetrics m) {
    _fps = m.fps;
    _ms = m.processingTimeMs;
  }

  Future<void> _save({required bool auto}) async {
    if (_saving) return;
    _saving = true;
    try {
      if (!await Gal.hasPermission(toAlbum: Platform.isAndroid) &&
          !await Gal.requestPermission(toAlbum: Platform.isAndroid)) {
        _toast('写真への保存が許可されていません（設定アプリから許可してください）');
        return;
      }
      final bytes = await _controller.capturePhoto(withOverlays: true);
      if (bytes == null) {
        if (!auto) _toast('画像を取得できませんでした');
        return;
      }
      final t = DateTime.now();
      String two(int v) => v.toString().padLeft(2, '0');
      final name = 'queen_${t.year}${two(t.month)}${two(t.day)}_${two(t.hour)}${two(t.minute)}${two(t.second)}'
          '${auto ? '_auto' : ''}';
      // Android は「findBeeQueen」アルバムにまとめる。iOS はアルバムに入れると写真の読み取り権限まで必要になるので入れない
      await Gal.putImageBytes(bytes, name: name, album: Platform.isAndroid ? 'findBeeQueen' : null);
      if (mounted) setState(() => _savedCount++);
      _toast(auto ? '女王蜂を見つけたので保存しました' : '保存しました');
    } on GalException catch (e) {
      _toast('保存に失敗しました: ${e.type.message}');
    } catch (e) {
      _toast('保存に失敗しました: $e');
    } finally {
      _saving = false;
    }
  }

  void _toast(String msg) {
    if (!mounted) return;
    ScaffoldMessenger.of(context)
      ..hideCurrentSnackBar()
      ..showSnackBar(SnackBar(content: Text(msg), duration: const Duration(seconds: 2)));
  }

  Future<void> _toggleTorch() async {
    final v = !_torch;
    await _controller.setTorchMode(v);
    if (mounted) setState(() => _torch = v);
  }

  Future<void> _setZoom(double z) async {
    final v = z.clamp(1.0, 8.0);
    _zoom.value = v;
    await _controller.setZoomLevel(v);
  }

  Future<void> _openSettings() async {
    final s = _settings!;
    await showModalBottomSheet<void>(
      context: context,
      isScrollControlled: true,
      showDragHandle: true,
      builder: (_) => _SettingsSheet(settings: s),
    );
    await s.save();
    _judge
      ..window = s.window
      ..minHits = s.minHits
      ..reset();
    _snap.cooldown = Duration(seconds: s.cooldownSec);
    await _controller.setConfidenceThreshold(s.conf);
    if (mounted) setState(() {});
  }

  @override
  Widget build(BuildContext context) {
    final s = _settings;
    return Scaffold(
      backgroundColor: Colors.black,
      body: s == null
          ? const Center(child: CircularProgressIndicator())
          : LayoutBuilder(
              builder: (context, box) => Stack(
                fit: StackFit.expand,
                children: [
                  YOLOView(
                    modelPath: modelPath,
                    task: YOLOTask.detect,
                    controller: _controller,
                    confidenceThreshold: s.conf,
                    iouThreshold: 0.5,
                    onResult: _onResult,
                    onPerformanceMetrics: _onMetrics,
                    onZoomChanged: (z) => _zoom.value = z,
                    onModelLoad: (_, _) => setState(() => _modelReady = true),
                    onModelError: (e, _, _) => setState(() => _modelError = '$e'),
                  ),
                  // ピンチでズーム、タップでピント合わせ（YOLOView 自体はジェスチャーを受け取らない）
                  GestureDetector(
                    behavior: HitTestBehavior.translucent,
                    onScaleStart: (_) => _baseZoom = _zoom.value,
                    onScaleUpdate: (d) {
                      if (d.pointerCount >= 2) _setZoom(_baseZoom * d.scale);
                    },
                    onTapUp: (d) => _controller.tapToFocus(
                      (d.localPosition.dx / box.maxWidth).clamp(0.0, 1.0),
                      (d.localPosition.dy / box.maxHeight).clamp(0.0, 1.0),
                    ),
                  ),
                  SafeArea(
                    child: Column(
                      children: [
                        _StatusBar(status: _status, settings: s, ready: _modelReady),
                        if (_modelError != null) _ErrorCard(message: _modelError!),
                        const Spacer(),
                        _BottomBar(
                          torch: _torch,
                          zoom: _zoom,
                          autoSave: s.autoSave,
                          savedCount: _savedCount,
                          onTorch: _toggleTorch,
                          onZoom: () => _setZoom(_zoom.value < 1.5 ? 2 : (_zoom.value < 2.5 ? 3 : 1)),
                          onShutter: () => _save(auto: false),
                          onAutoSave: () {
                            setState(() => s.autoSave = !s.autoSave);
                            s.save();
                            _toast(s.autoSave ? '発見時に自動で保存します' : '自動保存をオフにしました');
                          },
                          onSettings: _openSettings,
                        ),
                      ],
                    ),
                  ),
                ],
              ),
            ),
    );
  }
}

class _StatusBar extends StatelessWidget {
  const _StatusBar({required this.status, required this.settings, required this.ready});

  final ValueNotifier<_Status> status;
  final AppSettings settings;
  final bool ready;

  @override
  Widget build(BuildContext context) {
    return ValueListenableBuilder<_Status>(
      valueListenable: status,
      builder: (context, st, _) {
        final found = st.found;
        return AnimatedContainer(
          duration: const Duration(milliseconds: 150),
          margin: const EdgeInsets.fromLTRB(12, 8, 12, 0),
          padding: EdgeInsets.symmetric(horizontal: 16, vertical: found ? 14 : 8),
          decoration: BoxDecoration(
            color: found ? const Color(0xE6D50000) : const Color(0x99000000),
            borderRadius: BorderRadius.circular(14),
          ),
          child: Row(
            children: [
              Icon(found ? Icons.emoji_nature : Icons.search, color: Colors.white, size: found ? 30 : 20),
              const SizedBox(width: 10),
              Expanded(
                child: Text(
                  !ready ? 'モデルを読み込み中…' : (found ? '女王蜂発見！' : '巣板を映してください'),
                  style: TextStyle(
                    color: Colors.white,
                    fontSize: found ? 26 : 16,
                    fontWeight: found ? FontWeight.bold : FontWeight.normal,
                  ),
                ),
              ),
              Column(
                crossAxisAlignment: CrossAxisAlignment.end,
                children: [
                  if (st.maxConf > 0)
                    Text('確信度 ${(st.maxConf * 100).round()}%', style: const TextStyle(color: Colors.white)),
                  Text(
                    '${st.fps.toStringAsFixed(0)} fps  ${st.hits}/${settings.window}',
                    style: const TextStyle(color: Colors.white70, fontSize: 12),
                  ),
                ],
              ),
            ],
          ),
        );
      },
    );
  }
}

class _ErrorCard extends StatelessWidget {
  const _ErrorCard({required this.message});

  final String message;

  @override
  Widget build(BuildContext context) {
    return Card(
      color: Colors.red.shade900,
      margin: const EdgeInsets.all(12),
      child: Padding(
        padding: const EdgeInsets.all(12),
        child: Text('モデルを読み込めませんでした\n$message', style: const TextStyle(color: Colors.white)),
      ),
    );
  }
}

class _BottomBar extends StatelessWidget {
  const _BottomBar({
    required this.torch,
    required this.zoom,
    required this.autoSave,
    required this.savedCount,
    required this.onTorch,
    required this.onZoom,
    required this.onShutter,
    required this.onAutoSave,
    required this.onSettings,
  });

  final bool torch;
  final ValueNotifier<double> zoom;
  final bool autoSave;
  final int savedCount;
  final VoidCallback onTorch;
  final VoidCallback onZoom;
  final VoidCallback onShutter;
  final VoidCallback onAutoSave;
  final VoidCallback onSettings;

  @override
  Widget build(BuildContext context) {
    Widget roundButton({required Widget child, required VoidCallback onTap, required String label, bool on = false}) {
      return Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          Material(
            color: on ? Colors.amber : const Color(0x99000000),
            shape: const CircleBorder(),
            child: InkWell(
              customBorder: const CircleBorder(),
              onTap: onTap,
              child: SizedBox(
                width: 52,
                height: 52,
                child: Center(
                  child: IconTheme(
                    data: IconThemeData(color: on ? Colors.black : Colors.white),
                    child: DefaultTextStyle(
                      style: TextStyle(color: on ? Colors.black : Colors.white, fontWeight: FontWeight.bold),
                      child: child,
                    ),
                  ),
                ),
              ),
            ),
          ),
          const SizedBox(height: 4),
          Text(label, style: const TextStyle(color: Colors.white, fontSize: 11, shadows: [Shadow(blurRadius: 3)])),
        ],
      );
    }

    return Padding(
      padding: const EdgeInsets.fromLTRB(8, 0, 8, 12),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceEvenly,
        crossAxisAlignment: CrossAxisAlignment.end,
        children: [
          roundButton(
            child: Icon(torch ? Icons.flashlight_on : Icons.flashlight_off),
            onTap: onTorch,
            label: 'ライト',
            on: torch,
          ),
          ValueListenableBuilder<double>(
            valueListenable: zoom,
            builder: (_, z, _) => roundButton(child: Text('${z.toStringAsFixed(1)}x'), onTap: onZoom, label: 'ズーム'),
          ),
          // シャッター
          Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              GestureDetector(
                onTap: onShutter,
                child: Container(
                  width: 72,
                  height: 72,
                  decoration: BoxDecoration(
                    shape: BoxShape.circle,
                    color: Colors.white,
                    border: Border.all(color: Colors.black54, width: 4),
                  ),
                  child: const Icon(Icons.camera_alt, color: Colors.black87, size: 32),
                ),
              ),
              const SizedBox(height: 4),
              Text(
                savedCount > 0 ? '保存 $savedCount 枚' : '保存',
                style: const TextStyle(color: Colors.white, fontSize: 11, shadows: [Shadow(blurRadius: 3)]),
              ),
            ],
          ),
          roundButton(child: const Icon(Icons.save_alt), onTap: onAutoSave, label: '自動保存', on: autoSave),
          roundButton(child: const Icon(Icons.tune), onTap: onSettings, label: '設定'),
        ],
      ),
    );
  }
}

class _SettingsSheet extends StatefulWidget {
  const _SettingsSheet({required this.settings});

  final AppSettings settings;

  @override
  State<_SettingsSheet> createState() => _SettingsSheetState();
}

class _SettingsSheetState extends State<_SettingsSheet> {
  @override
  Widget build(BuildContext context) {
    final s = widget.settings;
    return SafeArea(
      child: Padding(
        padding: const EdgeInsets.fromLTRB(20, 0, 20, 16),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text('設定', style: Theme.of(context).textTheme.titleLarge),
            const SizedBox(height: 12),
            Text('しきい値（確信度） ${s.conf.toStringAsFixed(2)}'),
            Slider(
              value: s.conf,
              min: 0.10,
              max: 0.70,
              divisions: 12,
              label: s.conf.toStringAsFixed(2),
              onChanged: (v) => setState(() => s.conf = v),
            ),
            const Text(
              '誤検出が多いときは上げる（例 0.50）、見逃すときは下げる（例 0.15）',
              style: TextStyle(fontSize: 12, color: Colors.white70),
            ),
            const SizedBox(height: 16),
            Text('「発見」の判定: 直近 ${s.window} フレーム中 ${s.minHits} フレーム以上で検出'),
            Row(
              children: [
                const SizedBox(width: 90, child: Text('フレーム数')),
                Expanded(
                  child: Slider(
                    value: s.window.toDouble(),
                    min: 3,
                    max: 30,
                    divisions: 27,
                    label: '${s.window}',
                    onChanged: (v) => setState(() {
                      s.window = v.round();
                      if (s.minHits > s.window) s.minHits = s.window;
                    }),
                  ),
                ),
              ],
            ),
            Row(
              children: [
                const SizedBox(width: 90, child: Text('必要な回数')),
                Expanded(
                  child: Slider(
                    value: s.minHits.toDouble(),
                    min: 1,
                    max: s.window.toDouble(),
                    divisions: s.window - 1,
                    label: '${s.minHits}',
                    onChanged: (v) => setState(() => s.minHits = v.round()),
                  ),
                ),
              ],
            ),
            Text('自動保存の間隔 ${s.cooldownSec} 秒'),
            Slider(
              value: s.cooldownSec.toDouble(),
              min: 3,
              max: 60,
              divisions: 19,
              label: '${s.cooldownSec} 秒',
              onChanged: (v) => setState(() => s.cooldownSec = v.round()),
            ),
            SwitchListTile(
              contentPadding: EdgeInsets.zero,
              title: const Text('発見したら振動する'),
              value: s.vibrate,
              onChanged: (v) => setState(() => s.vibrate = v),
            ),
            Align(
              alignment: Alignment.centerRight,
              child: TextButton(
                onPressed: () => setState(() {
                  final d = AppSettings();
                  s
                    ..conf = d.conf
                    ..window = d.window
                    ..minHits = d.minHits
                    ..cooldownSec = d.cooldownSec
                    ..vibrate = d.vibrate;
                }),
                child: const Text('初期値に戻す'),
              ),
            ),
          ],
        ),
      ),
    );
  }
}
