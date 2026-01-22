"""
Video Processing Module
Transcribes video/audio content using OpenAI Whisper
"""
import whisper
import tempfile
import os
from typing import Dict, List, Any
from pathlib import Path
import torch


class VideoProcessor:
    """Process video files to extract audio transcription"""

    def __init__(self, model_name: str = "base"):
        """
        Initialize Whisper model

        Args:
            model_name: Whisper model size (tiny, base, small, medium, large)
        """
        self.model_name = model_name
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def _load_model(self):
        """Lazy load the Whisper model"""
        if self.model is None:
            print(f"Loading Whisper model '{self.model_name}' on {self.device}...")
            self.model = whisper.load_model(self.model_name, device=self.device)

    def process(self, file_bytes: bytes, filename: str) -> Dict[str, Any]:
        """
        Transcribe a video/audio file

        Args:
            file_bytes: Raw video/audio file bytes
            filename: Original filename (for extension detection)

        Returns:
            Dictionary containing transcription, segments, and metadata
        """
        self._load_model()

        # Create temporary file
        suffix = Path(filename).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_file.write(file_bytes)
            tmp_path = tmp_file.name

        try:
            # Transcribe with Whisper
            result = self.model.transcribe(
                tmp_path,
                verbose=False,
                language=None,  # Auto-detect language
                task="transcribe",
                word_timestamps=True,  # Get word-level timestamps
            )

            # Process segments
            segments = self._process_segments(result.get("segments", []))

            # Extract metadata
            metadata = {
                "language": result.get("language", "unknown"),
                "duration": result.get("duration", 0),
                "model": self.model_name,
                "device": self.device,
            }

            # Get full transcript
            full_text = result.get("text", "").strip()

            return {
                "text": full_text,
                "segments": segments,
                "metadata": metadata,
                "statistics": {
                    "duration": metadata["duration"],
                    "segment_count": len(segments),
                    "word_count": len(full_text.split()),
                    "char_count": len(full_text),
                },
            }

        finally:
            # Clean up temporary file
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

    def _process_segments(self, segments: List[Dict]) -> List[Dict[str, Any]]:
        """
        Process and clean up segments from Whisper output

        Args:
            segments: Raw segments from Whisper

        Returns:
            Cleaned segments with timestamps and text
        """
        processed = []

        for seg in segments:
            processed_seg = {
                "id": seg.get("id", 0),
                "start": seg.get("start", 0.0),
                "end": seg.get("end", 0.0),
                "text": seg.get("text", "").strip(),
            }

            # Add word-level timestamps if available
            if "words" in seg:
                processed_seg["words"] = [
                    {
                        "word": w.get("word", "").strip(),
                        "start": w.get("start", 0.0),
                        "end": w.get("end", 0.0),
                    }
                    for w in seg["words"]
                ]

            processed.append(processed_seg)

        return processed

    def extract_text_at_timestamp(
        self, segments: List[Dict], timestamp: float, context_window: float = 5.0
    ) -> str:
        """
        Extract text around a specific timestamp

        Args:
            segments: Transcription segments
            timestamp: Target timestamp in seconds
            context_window: Context window in seconds (default 5s)

        Returns:
            Text around the timestamp
        """
        start_time = timestamp - context_window
        end_time = timestamp + context_window

        relevant_segments = [
            seg
            for seg in segments
            if seg["start"] <= end_time and seg["end"] >= start_time
        ]

        return " ".join([seg["text"] for seg in relevant_segments])
