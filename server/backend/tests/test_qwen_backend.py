"""Tests for Qwen3 backend integration."""

from __future__ import annotations

import importlib
import sys
import types
from typing import Any

import numpy as np
import pytest

# Ensure the torch stub from conftest.py is installed
pytestmark = pytest.mark.usefixtures("torch_stub")

def _install_stubs():
    """Stub soundfile and whisperx so backend imports in the lightweight test env."""
    if "soundfile" not in sys.modules:
        soundfile_stub = types.ModuleType("soundfile")
        soundfile_stub.read = lambda *args, **kwargs: (np.zeros(16000, dtype=np.float32), 16000)
        sys.modules["soundfile"] = soundfile_stub
        
    if "whisperx" not in sys.modules:
        whisperx_stub = types.ModuleType("whisperx")
        
        class FakeModel:
            def transcribe(self, audio, batch_size=None, language=None, task=None):
                return {
                    "segments": [{"text": "qwen test", "start": 0.0, "end": 1.0}],
                    "language": "en"
                }
                
        whisperx_stub.load_qwen_model = lambda **kwargs: FakeModel()
        whisperx_stub.load_qwen_align_model = lambda **kwargs: (object(), {"metadata": "test"})
        whisperx_stub.align_qwen = lambda segments, model, metadata, audio, device, **kwargs: {
            "segments": [
                {
                    "text": "qwen test",
                    "start": 0.0,
                    "end": 1.0,
                    "words": [{"word": "qwen", "start": 0.0, "end": 0.5}, {"word": "test", "start": 0.5, "end": 1.0}]
                }
            ]
        }
        sys.modules["whisperx"] = whisperx_stub

def _import_qwen_backend():
    _install_stubs()
    return importlib.import_module("server.core.stt.backends.qwen_backend")

def test_qwen_backend_detection() -> None:
    from server.core.stt.backends.factory import detect_backend_type, is_qwen_model
    
    assert detect_backend_type("Qwen/Qwen3-ASR-0.6B") == "qwen"
    assert is_qwen_model("Qwen/Qwen3-ASR-0.6B")
    assert detect_backend_type("Qwen/Qwen3.5-0.8B-Base") == "qwen"
    # Relaxed pattern check
    assert detect_backend_type("Qwen/Anything") == "qwen"
    
    # Test HF hub snapshot path mentioned by user
    hub_path = "/models/hub/models--Qwen--Qwen3-ASR-1.7B/snapshots/7278e1e70fe206f11671096ffdd38061171dd6e5"
    assert detect_backend_type(hub_path) == "qwen"
    assert is_qwen_model(hub_path)

def test_qwen_backend_transcribe_mocked(monkeypatch) -> None:
    module = _import_qwen_backend()
    backend = module.QwenBackend()
    
    # Mock load to avoid real loading
    backend._model = types.SimpleNamespace()
    backend._model.transcribe = lambda audio, batch_size, language, task: {
        "segments": [{"text": "hello qwen", "start": 0.0, "end": 1.0}],
        "language": "en"
    }
    backend._device = "cpu"
    
    # Mock _align to avoid alignment model loading
    backend._align = lambda wx_result, audio, language: {
        "segments": [
            {
                "text": "hello qwen",
                "start": 0.0,
                "end": 1.0,
                "words": [
                    {"word": "hello", "start": 0.0, "end": 0.4, "score": 0.9},
                    {"word": "qwen", "start": 0.4, "end": 1.0, "score": 0.8}
                ]
            }
        ]
    }
    
    audio = np.zeros(16000, dtype=np.float32)
    segments, info = backend.transcribe(audio, word_timestamps=True)
    
    assert len(segments) == 1
    assert segments[0].text == "hello qwen"
    assert len(segments[0].words) == 2
    assert segments[0].words[0]["word"] == "hello"
    assert info.language == "en"

def test_qwen_backend_supports_translation() -> None:
    module = _import_qwen_backend()
    backend = module.QwenBackend()
    assert backend.supports_translation() is False
