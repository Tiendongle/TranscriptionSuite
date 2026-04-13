"""NLLB-200 Translation backend.

Implements the TranslationBackend interface using the Hugging Face Transformers library
to run NLLB-200 models for high-quality translation.
"""

from __future__ import annotations

import logging
from typing import Any

from server.core.audio_utils import clear_gpu_cache
from server.core.stt.backends.base import TranslationBackend

logger = logging.getLogger(__name__)

# Mapping from ISO 639-1 / Whisper codes to Flores-200 codes.
# This is a representative subset of the 200+ languages supported by NLLB-200.
# Source: https://github.com/facebookresearch/flores/blob/main/flores200/README.md#languages-in-flores-200
ISO_TO_FLORES = {
    "en": "eng_Latn",
    "af": "afr_Latn",
    "am": "amh_Ethi",
    "ar": "ary_Arab",  # Moroccan Arabic (or arb_Arab for Standard)
    "as": "asm_Beng",
    "az": "azj_Latn",
    "ba": "bak_Cyrl",
    "be": "bel_Cyrl",
    "bg": "bul_Cyrl",
    "bn": "ben_Beng",
    "bo": "bod_Tibt",
    "bs": "bos_Latn",
    "ca": "cat_Latn",
    "cs": "ces_Latn",
    "cy": "cym_Latn",
    "da": "dan_Latn",
    "de": "deu_Latn",
    "el": "ell_Grek",
    "es": "spa_Latn",
    "et": "est_Latn",
    "eu": "eus_Latn",
    "fa": "pes_Arab",
    "fi": "fin_Latn",
    "fo": "fao_Latn",
    "fr": "fra_Latn",
    "gl": "glg_Latn",
    "gu": "guj_Gujr",
    "ha": "hau_Latn",
    "hi": "hin_Deva",
    "hr": "hrv_Latn",
    "ht": "hat_Latn",
    "hu": "hun_Latn",
    "hy": "hye_Armn",
    "id": "ind_Latn",
    "is": "isl_Latn",
    "it": "ita_Latn",
    "he": "heb_Hebr",
    "ja": "jpn_Jpan",
    "jw": "jav_Latn",
    "ka": "kat_Geor",
    "kk": "kaz_Cyrl",
    "km": "khm_Khmr",
    "kn": "kan_Knda",
    "ko": "kor_Kore",
    "la": "lat_Latn",
    "lb": "ltz_Latn",
    "ln": "lin_Latn",
    "lo": "lao_Laoo",
    "lt": "lit_Latn",
    "lv": "lvs_Latn",
    "mg": "plt_Latn",
    "mi": "mri_Latn",
    "mk": "mkd_Cyrl",
    "ml": "mal_Mlym",
    "mn": "mon_Cyrl",
    "mr": "mar_Deva",
    "ms": "zsm_Latn",
    "mt": "mlt_Latn",
    "my": "mya_Mymr",
    "ne": "npi_Deva",
    "nl": "nld_Latn",
    "nn": "nno_Latn",
    "no": "nob_Latn",
    "oc": "oci_Latn",
    "pa": "pan_Guru",
    "pl": "pol_Latn",
    "ps": "pbt_Arab",
    "pt": "por_Latn",
    "ro": "ron_Latn",
    "ru": "rus_Cyrl",
    "sa": "san_Deva",
    "sd": "snd_Arab",
    "si": "sin_Sinh",
    "sk": "slk_Latn",
    "sl": "slv_Latn",
    "sn": "sna_Latn",
    "so": "som_Latn",
    "sq": "als_Latn",
    "sr": "srp_Cyrl",
    "su": "sun_Latn",
    "sv": "swe_Latn",
    "sw": "swh_Latn",
    "ta": "tam_Taml",
    "te": "tel_Telu",
    "tg": "tgk_Cyrl",
    "th": "tha_Thai",
    "tk": "tuk_Latn",
    "tl": "tgl_Latn",
    "tr": "tur_Latn",
    "tt": "tat_Cyrl",
    "uk": "ukr_Cyrl",
    "ur": "urd_Arab",
    "uz": "uzn_Latn",
    "vi": "vie_Latn",
    "yi": "ydd_Hebr",
    "yo": "yor_Latn",
    "zh": "zho_Hans",  # Default to Simplified
}


class NLLBBackend(TranslationBackend):
    """NLLB-200 Translation backend using Transformers pipeline."""

    def __init__(self):
        self.model_name = None
        self.translator = None
        self.device = None

    def load(self, model_name: str, device: str, **kwargs: Any) -> None:
        """Load the NLLB model for translation.

        Args:
            model_name: Hugging Face model ID (e.g. "facebook/nllb-200-distilled-600M")
            device: Target device ("cuda" or "cpu")
            **kwargs: Additional pipeline options.
        """
        import torch
        from transformers import pipeline

        self.model_name = model_name
        self.device = device

        # Map Xenova models to original Facebook models for Python backend compatibility.
        # Xenova models are optimized for transformers.js (ONNX) and often lack 
        # the standard PyTorch weights that the Python transformers library expects.
        effective_model = model_name
        if model_name.startswith("Xenova/nllb-200-"):
            effective_model = model_name.replace("Xenova/nllb-200-", "facebook/nllb-200-")
            logger.info(f"Mapping {model_name} to {effective_model} for Python backend compatibility")


        device_idx = -1
        if device == "cuda":
            device_idx = kwargs.get("gpu_device_index", 0)
        elif device == "mps":
            device_idx = "mps"

        torch_dtype = torch.float32
        if device != "cpu" and kwargs.get("compute_type") != "float32":
            # Use float16 on GPU by default for performance
            torch_dtype = torch.float16

        logger.info(f"Loading NLLB model '{model_name}' on {device}:{device_idx}")

        # Note: We don't set src_lang/tgt_lang here so that they can be
        # varied per translate() call, but NLLB pipeline requires them
        # as defaults. We'll set them to English.
        self.translator = pipeline(
            "translation",
            model=effective_model,
            device=device_idx,
            torch_dtype=torch_dtype,
            src_lang="eng_Latn",
            tgt_lang="eng_Latn",
        )

        logger.info("NLLB translation model loaded and ready")

    def unload(self) -> None:
        """Unload the model and free GPU memory."""
        if self.translator is not None:
            import torch

            del self.translator
            self.translator = None
            clear_gpu_cache()
            logger.info(f"NLLB model '{self.model_name}' unloaded")
            self.model_name = None

    def is_loaded(self) -> bool:
        """Return True if a model is currently loaded."""
        return self.translator is not None

    def translate(
        self,
        text: str,
        *,
        source_lang: str | None = None,
        target_lang: str = "en",
        **kwargs: Any,
    ) -> str:
        """Translate text using NLLB-200.

        Args:
            text: Input text.
            source_lang: ISO 639-1 source language code.
            target_lang: ISO 639-1 target language code.
        """
        if self.translator is None:
            raise RuntimeError("NLLB model is not loaded")

        if not text.strip():
            return text

        # Map ISO codes to Flores-200 codes
        src_flores = ISO_TO_FLORES.get(source_lang, "eng_Latn") if source_lang else "eng_Latn"
        tgt_flores = ISO_TO_FLORES.get(target_lang, "eng_Latn")

        # NLLB-200 pipeline uses 'src_lang' and 'tgt_lang' parameters
        result = self.translator(
            text,
            src_lang=src_flores,
            tgt_lang=tgt_flores,
            max_length=kwargs.get("max_length", 400),
        )

        if isinstance(result, list) and len(result) > 0:
            return result[0].get("translation_text", "").strip()
        return ""

    def get_supported_languages(self) -> list[dict[str, str]]:
        """Return subset of supported languages (mapping ISO to Flores)."""
        # For simplicity, we return the names if we can derive them,
        # but for now we'll just return the codes.
        return [{"code": k, "name": k} for k in ISO_TO_FLORES.keys()]

    @property
    def backend_name(self) -> str:
        return "nllb"
