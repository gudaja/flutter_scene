// The WebGL2 backend's bookkeeping for synchronous queries: when a program's
// link status, a shader's compile status and a framebuffer's completeness are
// asked for. Pure Dart, so it
// runs on the VM against fake GL queries.
// ignore_for_file: implementation_imports
import 'package:flutter_scene/src/gpu/web/webgl_sync_state.dart';
import 'package:flutter_test/flutter_test.dart';

/// Fake GL program: counts the queries and lets a test finish the link.
class _FakeProgram {
  bool complete = false;
  String? error;
  int completionQueries = 0;
  int linkStatusQueries = 0;
  int reflections = 0;
  int now = 0;

  ProgramLink link({required bool parallel}) => ProgramLink(
    parallel: parallel,
    isComplete: () {
      completionQueries++;
      return complete;
    },
    linkError: () {
      linkStatusQueries++;
      return error;
    },
    onLinked: () => reflections++,
    nowMicros: () => now,
  );
}

/// Fake GL shader object: counts the compile-status and info-log queries.
class _FakeShader {
  _FakeShader(this.stage, {this.compiled = true});
  final String stage;
  final bool compiled;
  int statusQueries = 0;
  int logQueries = 0;
  int sourceQueries = 0;

  late final ShaderCompile compile = ShaderCompile(
    stage: stage,
    isCompiled: () {
      statusQueries++;
      return compiled;
    },
    infoLog: () {
      logQueries++;
      return 'ERROR: 0:3: \'v_x\' : undeclared identifier';
    },
    source: () {
      sourceQueries++;
      return '#version 300 es\nvoid main() { v_x; }';
    },
  );
}

const FramebufferSignature _rgba = (
  colorFormat: 0x8058,
  colorSamples: 1,
  colorIsCube: false,
  colorMipped: false,
  depthFormat: 0x88F0,
  depthSamples: 1,
);

void main() {
  group('ProgramLink', () {
    test('T-WR81h1-1 · with parallel compile the link status is not read '
        'until the completion query reports the link done', () {
      final p = _FakeProgram();
      final link = p.link(parallel: true);
      expect(p.linkStatusQueries, 0);
      expect(link.isPending, isTrue);
      expect(link.poll(), isFalse);
      expect(p.completionQueries, 1);
      expect(p.linkStatusQueries, 0);
      expect(p.reflections, 0);

      p.complete = true;
      p.now += 5000;
      expect(link.poll(), isTrue);
      expect(p.linkStatusQueries, 1);
      expect(p.reflections, 1);
      expect(link.isLinked, isTrue);

      // A linked program answers from memory.
      p.now += 5000;
      expect(link.poll(), isTrue);
      expect(p.completionQueries, 2);
      expect(p.linkStatusQueries, 1);
      expect(p.reflections, 1);
    });

    test('T-WR81h1-2 · polls of a pending link closer than the interval '
        'issue one completion query', () {
      final p = _FakeProgram();
      final link = p.link(parallel: true);
      for (var i = 0; i < 50; i++) {
        expect(link.poll(), isFalse);
        p.now += 10; // fifty draws of the program within one frame
      }
      expect(p.completionQueries, 1);
      p.now += 4000;
      expect(link.poll(), isFalse);
      expect(p.completionQueries, 2);
    });

    test('T-WR81h1-3 · a failed link throws its log from poll and again '
        'later without querying again', () {
      final p = _FakeProgram()
        ..complete = true
        ..error = 'L0001: varying mismatch';
      final link = p.link(parallel: true);
      expect(
        link.poll,
        throwsA(
          isA<Exception>().having(
            (e) => '$e',
            'message',
            contains('Failed to link program: L0001: varying mismatch'),
          ),
        ),
      );
      expect(link.isPending, isFalse);
      expect(link.isLinked, isFalse);
      p.now += 10000;
      expect(link.poll, throwsA(isA<Exception>()));
      expect(link.finish, throwsA(isA<Exception>()));
      expect(p.completionQueries, 1);
      expect(p.linkStatusQueries, 1);
      expect(p.reflections, 0);
    });

    test('T-WR81h1-4 · finish completes a pending link at once (the blocking '
        'path of a draw that cannot wait)', () {
      final p = _FakeProgram();
      final link = p.link(parallel: true);
      link.finish();
      expect(p.completionQueries, 0);
      expect(p.linkStatusQueries, 1);
      expect(p.reflections, 1);
      expect(link.poll(), isTrue);
      expect(p.completionQueries, 0);
    });

    test('T-WR81h1-5 · without the extension the link finishes in the '
        'constructor and a failure throws there', () {
      final ok = _FakeProgram();
      final link = ok.link(parallel: false);
      expect(ok.linkStatusQueries, 1);
      expect(ok.reflections, 1);
      expect(link.poll(), isTrue);
      expect(ok.completionQueries, 0);

      final bad = _FakeProgram()..error = 'no main';
      expect(() => bad.link(parallel: false), throwsA(isA<Exception>()));
      expect(bad.reflections, 0);
    });
  });

  group('ShaderCompile', () {
    test('T-WR81h6-1 · compiling and a successful link never read the '
        'compile status', () {
      final vs = _FakeShader('vertex');
      final fs = _FakeShader('fragment');
      // The pipeline linked: both stages compiled.
      vs.compile.markCompiled();
      fs.compile.markCompiled();
      expect(vs.compile.error(), isNull);
      expect(fs.compile.error(), isNull);
      expect(vs.statusQueries + fs.statusQueries, 0);
      expect(vs.logQueries + fs.logQueries, 0);
    });

    test('T-WR81h6-2 · a failed link names the stage that failed to compile '
        'with its info log and source, querying each shader once', () {
      final vs = _FakeShader('vertex');
      final fs = _FakeShader('fragment', compiled: false);
      final log = linkFailureLog('L0001: fragment shader not compiled', [
        vs.compile,
        fs.compile,
      ]);
      expect(log, contains('Failed to compile fragment shader:'));
      expect(log, contains("'v_x' : undeclared identifier"));
      expect(log, contains('--- source ---\n#version 300 es'));
      expect(log, contains('L0001: fragment shader not compiled'));
      expect(log, isNot(contains('vertex shader')));
      expect(vs.statusQueries, 1);
      expect(fs.statusQueries, 1);
      expect(vs.logQueries, 0);
      expect(fs.logQueries, 1);

      // A second pipeline with the same broken shader: no new queries.
      linkFailureLog('again', [vs.compile, fs.compile]);
      expect(vs.statusQueries + fs.statusQueries, 2);
      expect(fs.logQueries + fs.sourceQueries, 2);
    });

    test('T-WR81h6-3 · a link failure between compiled shaders reports the '
        'program log alone', () {
      final vs = _FakeShader('vertex');
      final fs = _FakeShader('fragment');
      final log = linkFailureLog('L0010: varying v_missing not written', [
        vs.compile,
        fs.compile,
      ]);
      expect(log, 'L0010: varying v_missing not written');
      expect(vs.statusQueries, 1);
      expect(fs.statusQueries, 1);
      expect(vs.logQueries + fs.logQueries, 0);
    });
  });

  group('FramebufferCompleteness', () {
    test('T-WR81h1-6 · a complete signature is checked once; other '
        'signatures and incomplete results check every time', () {
      const complete = 0x8CD5;
      const incomplete = 0x8CD6;
      final memo = FramebufferCompleteness();
      var checks = 0;
      int check(int result) {
        checks++;
        return result;
      }

      expect(memo.status(_rgba, complete, () => check(complete)), complete);
      // A resized or reallocated target with the same formats: no query.
      expect(memo.status(_rgba, complete, () => check(complete)), complete);
      expect(checks, 1);

      final msaa = (
        colorFormat: _rgba.colorFormat,
        colorSamples: 4,
        colorIsCube: false,
        colorMipped: false,
        depthFormat: _rgba.depthFormat,
        depthSamples: 4,
      );
      expect(memo.status(msaa, complete, () => check(incomplete)), incomplete);
      expect(memo.status(msaa, complete, () => check(incomplete)), incomplete);
      expect(checks, 3);
      expect(memo.knownComplete, 1);
    });
  });
}
