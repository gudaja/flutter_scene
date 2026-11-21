/// Bookkeeping for the WebGL2 backend's synchronous queries, kept free of
/// `package:web` so it is testable on the VM.
///
/// A WebGL query that returns a value (`getProgramParameter`,
/// `checkFramebufferStatus`, ...) is a round trip to the browser's GPU
/// process: the calling thread waits until every command queued before it has
/// executed. Right after `linkProgram` that wait includes the whole link, and
/// with a frame's texture uploads queued it can freeze the main thread for
/// hundreds of milliseconds. These classes decide when such a query is needed
/// at all.
library;

/// The link of one WebGL program.
///
/// With `KHR_parallel_shader_compile` ([parallel]) the link runs in the
/// background: [poll] asks `COMPLETION_STATUS_KHR`, which does not wait, and
/// only once the link is done reads `LINK_STATUS` and the program's
/// reflection ([onLinked]), both cheap by then. [finish] is the blocking path
/// for a caller that must use the program now. Without the extension the
/// constructor finishes the link at once, as a plain `linkProgram` did.
///
/// A failed link throws an [Exception] carrying the driver's log from [poll]
/// or [finish], and again from every later call, without querying again.
final class ProgramLink {
  /// Tracks a program whose `linkProgram` was just issued.
  ///
  /// [isComplete] queries `COMPLETION_STATUS_KHR` (non-blocking).
  /// [linkError] queries `LINK_STATUS` and returns null on success or the
  /// info log on failure (and deletes the program). [onLinked] builds the
  /// program's reflection caches. [nowMicros] is a monotonic clock; polls of
  /// a pending program closer together than [pollIntervalMicros] return false
  /// without a query, so many draws of one pending program in a frame cost one
  /// query.
  ProgramLink({
    required bool parallel,
    required bool Function() isComplete,
    required String? Function() linkError,
    required void Function() onLinked,
    required int Function() nowMicros,
    this.pollIntervalMicros = 4000,
  }) : _isComplete = isComplete,
       _linkError = linkError,
       _onLinked = onLinked,
       _now = nowMicros {
    if (!parallel) finish();
  }

  final bool Function() _isComplete;
  final String? Function() _linkError;
  final void Function() _onLinked;
  final int Function() _now;

  /// Minimum spacing of two completion queries of a pending link.
  final int pollIntervalMicros;

  bool _linked = false;
  String? _error;
  int? _lastPollMicros;

  /// Whether the link finished successfully. Never queries.
  bool get isLinked => _linked;

  /// Whether the link is still running (or not yet observed as finished).
  bool get isPending => !_linked && _error == null;

  /// True once the program is linked and usable; false while the link still
  /// runs. Never blocks on the link. Throws the link failure.
  bool poll() {
    if (_linked) return true;
    _throwIfFailed();
    final now = _now();
    final last = _lastPollMicros;
    if (last != null && now - last < pollIntervalMicros) return false;
    _lastPollMicros = now;
    if (!_isComplete()) return false;
    finish();
    return true;
  }

  /// Completes the link now, waiting for it if it still runs. Throws the link
  /// failure.
  void finish() {
    if (_linked) return;
    _throwIfFailed();
    final error = _linkError();
    if (error != null) {
      _error = error;
      _throwIfFailed();
    }
    _onLinked();
    _linked = true;
  }

  void _throwIfFailed() {
    final error = _error;
    if (error != null) throw Exception('Failed to link program: $error');
  }
}

/// What decides whether a framebuffer is complete: the attachments' formats,
/// sample counts, kinds, and levels, not which textures they are.
///
/// Render targets are recreated (resized, pooled, reallocated after a quality
/// change), and each new texture means a new framebuffer, but the combination
/// of formats stays the same. Keyed on this, the completeness check runs once
/// per combination instead of once per framebuffer.
typedef FramebufferSignature = ({
  int colorFormat,
  int colorSamples,
  bool colorIsCube,
  bool colorMipped,
  int? depthFormat,
  int depthSamples,
});

/// Remembers which [FramebufferSignature]s were complete, so
/// `checkFramebufferStatus` (a synchronous GPU round trip) runs once per
/// signature. An incomplete result is not remembered: every framebuffer with
/// that signature checks again and reports its own error.
final class FramebufferCompleteness {
  final Set<FramebufferSignature> _complete = {};

  /// The status of a framebuffer with [signature]: [complete] without calling
  /// [checkStatus] when the signature is known complete, otherwise the result
  /// of [checkStatus], remembered when it equals [complete].
  int status(
    FramebufferSignature signature,
    int complete,
    int Function() checkStatus,
  ) {
    if (_complete.contains(signature)) return complete;
    final status = checkStatus();
    if (status == complete) _complete.add(signature);
    return status;
  }

  /// Signatures known complete.
  int get knownComplete => _complete.length;
}
