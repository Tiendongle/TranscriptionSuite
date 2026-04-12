"""WhisperX Qwen3 STT backend.

Wraps the Qwen3 fork of the WhisperX library behind the STTBackend interface.
Executes Qwen3 ASR and forced alignment via the newly introduced Qwen API endpoints.
"""

from __future__ import annotations

import gc
import importlib
import logging
import os
import time
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from server.core.audio_utils import clear_gpu_cache
from server.core.stt.backends.base import (
    BackendSegment,
    BackendTranscriptionInfo,
    DiarizedTranscriptionResult,
    STTBackend,
)

SAMPLE_RATE = 16000

logger = logging.getLogger(__name__)

_PYANNOTE_TORCHCODEC_WARNING_RE = (
    r"torchcodec is not installed correctly so built-in audio decoding will fail\..*"
)
warnings.filterwarnings(
    "ignore",
    message=_PYANNOTE_TORCHCODEC_WARNING_RE,
    category=UserWarning,
)


def _import_whisperx_modules(
    *,
    include_diarize: bool = False,
) -> tuple[Any, Any | None]:
    """Import WhisperX and optionally diarize."""
    whisperx = importlib.import_module("whisperx")
    diarize_module = importlib.import_module("whisperx.diarize") if include_diarize else None
    return whisperx, diarize_module


class QwenBackend(STTBackend):
    """Qwen3 backend utilizing Qwen ASR pipeline + Qwen Forced Aligner + pyannote diarization."""

    def __init__(self) -> None:
        self._model: Any | None = None
        self._model_name: str | None = None
        self._device: str = "cuda"
        self._batch_size: int = 16
        self._align_model: Any | None = None
        self._align_metadata: Any | None = None
        self._align_language: str | None = None

    def load(self, model_name: str, device: str, **kwargs: Any) -> None:
        whisperx, _ = _import_whisperx_modules()

        self._batch_size = kwargs.get("batch_size", 16)
        self._device = device
        self._model_name = model_name

        logger.info(f"Loading Qwen3 model: {model_name} on {device}")
        
        self._model = whisperx.load_qwen_model(
            whisper_arch=model_name,
            device=device,
        )
        logger.info("Qwen3 model loaded")

    def unload(self) -> None:
        self._model = None
        self._model_name = None
        self._align_model = None
        self._align_metadata = None
        self._align_language = None
        clear_gpu_cache()

    def is_loaded(self) -> bool:
        return self._model is not None

    def warmup(self, *, language: str = "en") -> None:
        if self._model is None:
            return

        whisperx, _ = _import_whisperx_modules()
        warmup_path = Path(__file__).parent.parent / "warmup_audio.wav"

        if not warmup_path.exists():
            warmup_audio = np.zeros(SAMPLE_RATE, dtype=np.float32)
        else:
            warmup_audio, _ = sf.read(str(warmup_path), dtype="float32")

        # 1. Transcribe warmup audio
        wx_result: dict[str, Any] = {}
        try:
            t0 = time.perf_counter()
            wx_result = self._model.transcribe(warmup_audio, batch_size=1, language="en")
            logger.info("Qwen warmup transcribe complete (%.2fs)", time.perf_counter() - t0)
        except Exception as e:
            logger.warning(f"Qwen warmup transcribe failed: {e}")

        # 2. Load alignment model
        align_lang = language or "en"
        forced_aligner = os.environ.get("QWEN_FORCED_ALIGNER")
        
        try:
            t0 = time.perf_counter()
            self._align_model, self._align_metadata = whisperx.load_qwen_align_model(
                language_code=align_lang,
                device=self._device,
                model_name=forced_aligner,
            )
            self._align_language = align_lang
            logger.info("Qwen warmup alignment model loaded (%.2fs)", time.perf_counter() - t0)
        except Exception as e:
            logger.warning(f"Qwen warmup alignment model load failed: {e}")
            return

        # 3. Align inference
        segments = wx_result.get("segments", [])
        if not segments:
            duration = len(warmup_audio) / SAMPLE_RATE
            segments = [{"text": "warmup", "start": 0.0, "end": duration}]
            
        try:
            t0 = time.perf_counter()
            whisperx.align_qwen(
                segments,
                self._align_model,
                self._align_metadata,
                warmup_audio,
                self._device,
                return_char_alignments=False,
            )
            logger.info("Qwen warmup alignment complete (%.2fs)", time.perf_counter() - t0)
        except Exception as e:
            logger.info(f"Qwen warmup alignment skipped silently (expected if dummy segment): {e}")

        # 4. Release alignment model
        del self._align_model
        del self._align_metadata
        self._align_model = None
        self._align_metadata = None
        self._align_language = None
        clear_gpu_cache()

    def transcribe(
        self,
        audio: np.ndarray,
        *,
        audio_sample_rate: int = SAMPLE_RATE,
        language: str | None = None,
        task: str = "transcribe",
        beam_size: int = 5,
        initial_prompt: str | None = None,
        suppress_tokens: list[int] | None = None,
        vad_filter: bool = True,
        word_timestamps: bool = True,
        translation_target_language: str | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> tuple[list[BackendSegment], BackendTranscriptionInfo]:
        del audio_sample_rate, progress_callback, beam_size, initial_prompt, suppress_tokens, vad_filter
        
        if self._model is None:
            raise RuntimeError("Qwen model is not loaded")
            
        # task can be "transcribe" or "translate"

        t0 = time.perf_counter()
        # QwenAsrPipeline.transcribe
        wx_result = self._model.transcribe(
            audio,
            batch_size=self._batch_size,
            language=language,
            task=task,
        )
        logger.info("Qwen transcribe took %.2fs", time.perf_counter() - t0)

        detected_language = wx_result.get("language", language)

        if word_timestamps and wx_result.get("segments"):
            try:
                t0 = time.perf_counter()
                wx_result = self._align(wx_result, audio, detected_language)
                logger.info("Qwen alignment took %.2fs", time.perf_counter() - t0)
            except Exception as e:
                logger.warning(f"Qwen alignment failed, using raw timestamps: {e}")

        result_segments: list[BackendSegment] = []
        # Support both standard word arrays and the 'word_segments' returned natively by Qwen aligner
        for seg in wx_result.get("segments", []):
            words: list[dict[str, Any]] = []
            if word_timestamps and "words" in seg:
                words = [
                    {
                        "word": w.get("word", ""),
                        "start": w.get("start", 0.0),
                        "end": w.get("end", 0.0),
                        "probability": w.get("score", 0.0) or w.get("probability", 1.0),
                    }
                    for w in seg["words"]
                    if "start" in w and "end" in w
                ]
            result_segments.append(
                BackendSegment(
                    text=seg.get("text", ""),
                    start=seg.get("start", 0.0),
                    end=seg.get("end", 0.0),
                    words=words,
                )
            )

        backend_info = BackendTranscriptionInfo(
            language=detected_language,
            language_probability=0.0,
        )
        return result_segments, backend_info

    def transcribe_with_diarization(
        self,
        audio: np.ndarray,
        *,
        audio_sample_rate: int = SAMPLE_RATE,
        language: str | None = None,
        task: str = "transcribe",
        beam_size: int = 5,
        initial_prompt: str | None = None,
        suppress_tokens: list[int] | None = None,
        vad_filter: bool = True,
        num_speakers: int | None = None,
        hf_token: str | None = None,
        translation_target_language: str | None = None,
    ) -> DiarizedTranscriptionResult | None:
        del audio_sample_rate, beam_size, initial_prompt, suppress_tokens, vad_filter
        
        whisperx, diarize_module = _import_whisperx_modules(include_diarize=True)
        if diarize_module is None:
            raise RuntimeError("WhisperX diarization module failed to import")
        DiarizationPipeline = diarize_module.DiarizationPipeline

        if self._model is None:
            raise RuntimeError("Qwen model is not loaded")
            
        # task can be "transcribe" or "translate"

        token = hf_token or os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN")
        if not token:
            raise ValueError(
                "HuggingFace token required for diarization."
            )

        # 1. Transcribe
        logger.info("Qwen: transcribing audio")
        wx_result = self._model.transcribe(
            audio,
            batch_size=self._batch_size,
            language=language,
            task=task,
        )
        detected_language = wx_result.get("language", language)

        # 2. Align
        if wx_result.get("segments"):
            try:
                logger.info("Qwen: aligning with Forced Aligner")
                wx_result = self._align(wx_result, audio, detected_language)
            except Exception as e:
                logger.warning(f"Qwen alignment failed, continuing with raw timestamps: {e}")

        # 3. Diarize
        logger.info("Qwen: running diarization")
        diarize_model = DiarizationPipeline(use_auth_token=token, device=self._device)

        diarize_kwargs: dict[str, Any] = {}
        if num_speakers is not None:
            diarize_kwargs["min_speakers"] = num_speakers
            diarize_kwargs["max_speakers"] = num_speakers

        diarize_segments = diarize_model(audio, **diarize_kwargs)

        # 4. Assign word-level speakers
        logger.info("Qwen: assigning word speakers")
        wx_result = whisperx.assign_word_speakers(diarize_segments, wx_result)

        all_segments: list[dict[str, Any]] = []
        all_words: list[dict[str, Any]] = []
        speakers_seen: set[str] = set()

        for seg in wx_result.get("segments", []):
            speaker = seg.get("speaker", "SPEAKER_00")
            speakers_seen.add(speaker)

            seg_words: list[dict[str, Any]] = []
            if "words" in seg:
                for w in seg["words"]:
                    if "start" not in w or "end" not in w:
                        continue
                    word_dict = {
                        "word": w.get("word", ""),
                        "start": round(w.get("start", 0.0), 3),
                        "end": round(w.get("end", 0.0), 3),
                        "probability": round(w.get("score", 0.0) or w.get("probability", 1.0), 3),
                        "speaker": w.get("speaker", speaker),
                    }
                    seg_words.append(word_dict)
                    all_words.append(word_dict)

            all_segments.append(
                {
                    "text": seg.get("text", "").strip(),
                    "start": round(seg.get("start", 0.0), 3),
                    "end": round(seg.get("end", 0.0), 3),
                    "speaker": speaker,
                    "words": seg_words,
                }
            )

        num_speakers_found = len(speakers_seen)
        logger.info(
            "Qwen diarization complete: %s speakers, %s segments",
            num_speakers_found,
            len(all_segments),
        )

        return DiarizedTranscriptionResult(
            segments=all_segments,
            words=all_words,
            num_speakers=num_speakers_found,
            language=detected_language,
            language_probability=0.0,
        )

    def supports_translation(self) -> bool:
        return True

    @property
    def backend_name(self) -> str:
        return "qwen"

    def _align(
        self,
        wx_result: dict[str, Any],
        audio: np.ndarray,
        language: str | None,
    ) -> dict[str, Any]:
        """Run Qwen forced alignment."""
        whisperx, _ = _import_whisperx_modules()

        lang = language or "en"
        forced_aligner = os.environ.get("QWEN_FORCED_ALIGNER")

        if self._align_model is None or self._align_language != lang:
            t0 = time.perf_counter()
            if self._align_model is not None:
                del self._align_model
                del self._align_metadata
                gc.collect()
                clear_gpu_cache()
                
            self._align_model, self._align_metadata = whisperx.load_qwen_align_model(
                language_code=lang,
                device=self._device,
                model_name=forced_aligner,
            )
            self._align_language = lang
            logger.info("Qwen alignment model loaded (lang=%s, %.2fs)", lang, time.perf_counter() - t0)

        t0 = time.perf_counter()
        result = whisperx.align_qwen(
            wx_result["segments"],
            self._align_model,
            self._align_metadata,
            audio,
            self._device,
            return_char_alignments=False,
        )
        logger.info("Qwen alignment inference took %.2fs", time.perf_counter() - t0)
        return result
