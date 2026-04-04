"""
PDF Lecture Note Engine: video -> keyframes + subtitles -> LLM -> illustrated PDF

Features:
- Smart frame extraction (SceneDetect + info-density scoring + perceptual hash dedup)
- Subtitle-frame alignment (timestamp + semantic matching)
- Dual rendering: LaTeX (professional) / HTML+Chrome (zero-dependency fallback)
- Concurrent batch processing for video collections
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import AppConfig
from .frames import ExtractedFrame, _get_duration, _extract_frame_at


# ── Data structures ────────────────────────────────────────────

@dataclass
class ScoredFrame:
    """A video frame with quality/info scores."""
    path: Path
    timestamp: float
    scene_score: float = 0.0
    info_score: float = 0.0
    phash: str = ""
    subtitle_text: str = ""
    caption: str = ""

    @property
    def time_str(self) -> str:
        m, s = divmod(int(self.timestamp), 60)
        h, m = divmod(m, 60)
        return f"{h:02d}:{m:02d}:{s:02d}"

    @property
    def total_score(self) -> float:
        return self.scene_score * 0.4 + self.info_score * 0.6


@dataclass
class EpisodeResult:
    """Result of processing one video episode."""
    episode: int
    title: str
    duration: float
    frames: list[ScoredFrame]
    subtitle_text: str
    notes_md: str = ""
    notes_tex: str = ""
    pdf_path: Path | None = None
    html_path: Path | None = None


@dataclass
class CollectionResult:
    """Result of processing a video collection."""
    title: str
    episodes: list[EpisodeResult]
    merged_pdf: Path | None = None
    merged_html: Path | None = None


# ── Smart Frame Extractor ──────────────────────────────────────

class SmartFrameExtractor:
    """Extract keyframes with scene detection, info-density scoring, and hash dedup."""

    def __init__(
        self,
        max_frames: int = 15,
        scene_threshold: float = 25.0,
        hash_threshold: int = 8,
        min_interval: float = 5.0,
    ):
        self.max_frames = max_frames
        self.scene_threshold = scene_threshold
        self.hash_threshold = hash_threshold
        self.min_interval = min_interval

    def extract(self, video_path: Path, output_dir: Path) -> list[ScoredFrame]:
        output_dir.mkdir(parents=True, exist_ok=True)
        duration = _get_duration(video_path)

        # Stage 1: dense candidate extraction
        candidates = self._extract_candidates(video_path, output_dir, duration)

        # Stage 2: compute info-density scores
        candidates = self._score_frames(candidates)

        # Stage 3: dedup with perceptual hash
        candidates = self._dedup_by_hash(candidates)

        # Stage 4: enforce minimum time interval
        candidates = self._enforce_interval(candidates)

        # Stage 5: pick top N by total score
        candidates.sort(key=lambda f: f.total_score, reverse=True)
        selected = candidates[:self.max_frames]
        selected.sort(key=lambda f: f.timestamp)

        return selected

    def _extract_candidates(
        self, video_path: Path, output_dir: Path, duration: float
    ) -> list[ScoredFrame]:
        frames: list[ScoredFrame] = []

        # Try SceneDetect first
        try:
            from scenedetect import open_video, SceneManager
            from scenedetect.detectors import ContentDetector

            video = open_video(str(video_path))
            sm = SceneManager()
            sm.add_detector(ContentDetector(threshold=self.scene_threshold))
            sm.detect_scenes(video, show_progress=False)
            scenes = sm.get_scene_list()

            for i, (start, end) in enumerate(scenes):
                # Prefer end of scene (progressive PPT: final state is most complete)
                t = end.get_seconds() - 0.5
                if t < 0:
                    t = start.get_seconds() + 0.5
                out = output_dir / f"scene_{i:03d}_{t:.1f}s.jpg"
                _extract_frame_at(video_path, t, out)
                if out.exists() and out.stat().st_size > 3000:
                    frames.append(ScoredFrame(
                        path=out,
                        timestamp=t,
                        scene_score=min(1.0, self.scene_threshold / 30.0),
                    ))
        except (ImportError, Exception):
            pass

        # Supplement with uniform sampling (every 20-30s)
        if duration > 0:
            interval = max(20, min(30, duration / 20))
            t = 3.0
            idx = len(frames)
            while t < duration - 3:
                # Skip if too close to an existing frame
                if any(abs(f.timestamp - t) < self.min_interval for f in frames):
                    t += interval
                    continue
                out = output_dir / f"uniform_{idx:03d}_{t:.1f}s.jpg"
                _extract_frame_at(video_path, t, out)
                if out.exists() and out.stat().st_size > 3000:
                    frames.append(ScoredFrame(
                        path=out, timestamp=t, scene_score=0.3,
                    ))
                idx += 1
                t += interval

        return frames

    def _score_frames(self, frames: list[ScoredFrame]) -> list[ScoredFrame]:
        """Score each frame based on visual information density."""
        try:
            from PIL import Image
            import numpy as np
        except ImportError:
            for f in frames:
                f.info_score = 0.5
            return frames

        for frame in frames:
            try:
                img = Image.open(frame.path)
                arr = np.array(img.convert("L"), dtype=np.float32)

                # Edge density (Laplacian variance) - measures sharpness and detail
                from PIL import ImageFilter
                edges = img.convert("L").filter(ImageFilter.FIND_EDGES)
                edge_arr = np.array(edges, dtype=np.float32)
                edge_score = min(1.0, np.std(edge_arr) / 60.0)

                # Contrast score
                contrast = min(1.0, np.std(arr) / 80.0)

                # Entropy (information content)
                hist, _ = np.histogram(arr.ravel(), bins=64, range=(0, 256))
                hist = hist / hist.sum()
                hist = hist[hist > 0]
                entropy = -np.sum(hist * np.log2(hist))
                entropy_score = min(1.0, entropy / 6.0)

                # Non-uniformity: penalize frames that are mostly one color
                unique_ratio = len(np.unique(arr.astype(np.uint8))) / 256.0

                frame.info_score = (
                    edge_score * 0.35 +
                    contrast * 0.25 +
                    entropy_score * 0.25 +
                    unique_ratio * 0.15
                )
            except Exception:
                frame.info_score = 0.3

        return frames

    def _dedup_by_hash(self, frames: list[ScoredFrame]) -> list[ScoredFrame]:
        """Remove near-duplicate frames using perceptual hashing."""
        try:
            import imagehash
            from PIL import Image
        except ImportError:
            return frames

        unique: list[ScoredFrame] = []
        seen_hashes: list[Any] = []

        for frame in sorted(frames, key=lambda f: f.timestamp):
            try:
                img = Image.open(frame.path)
                h = imagehash.phash(img, hash_size=12)
                frame.phash = str(h)

                is_dup = False
                for prev_h in seen_hashes:
                    if h - prev_h < self.hash_threshold:
                        is_dup = True
                        break

                if not is_dup:
                    seen_hashes.append(h)
                    unique.append(frame)
            except Exception:
                unique.append(frame)

        return unique

    def _enforce_interval(self, frames: list[ScoredFrame]) -> list[ScoredFrame]:
        """Ensure minimum time gap between selected frames."""
        frames.sort(key=lambda f: f.timestamp)
        filtered = []
        last_t = -999.0
        for f in frames:
            if f.timestamp - last_t >= self.min_interval:
                filtered.append(f)
                last_t = f.timestamp
            elif f.total_score > 0.7:
                filtered.append(f)
                last_t = f.timestamp
        return filtered


# ── Subtitle-Frame Aligner ─────────────────────────────────────

class SubtitleFrameAligner:
    """Align extracted frames to subtitle text by timestamp proximity."""

    def __init__(self, tolerance: float = 10.0):
        self.tolerance = tolerance

    def align(
        self, frames: list[ScoredFrame], subtitle_segments: list
    ) -> list[ScoredFrame]:
        for frame in frames:
            best_text = ""
            best_dist = float("inf")
            for seg in subtitle_segments:
                seg_start = seg.start if hasattr(seg, "start") else seg.get("start", 0)
                seg_end = seg.end if hasattr(seg, "end") else seg.get("end", 0)
                seg_text = seg.text if hasattr(seg, "text") else seg.get("text", "")
                mid = (seg_start + seg_end) / 2
                dist = abs(frame.timestamp - mid)
                if dist < best_dist and dist <= self.tolerance:
                    best_dist = dist
                    best_text = seg_text
            frame.subtitle_text = best_text
        return frames


# ── HTML Note Builder ──────────────────────────────────────────

class HTMLNoteBuilder:
    """Build illustrated lecture notes as HTML -> PDF via Chrome headless."""

    STYLE = """
<style>
  @page { size: A4; margin: 2cm; }
  body { font-family: "PingFang SC","Hiragino Sans GB","Noto Sans CJK SC",sans-serif;
         max-width: 900px; margin: 0 auto; padding: 2rem; line-height: 1.8; color: #2d2d2d; }
  h1 { color: #1a1a2e; border-bottom: 3px solid #e94560; padding-bottom: .5rem; font-size: 1.8rem; }
  h2 { color: #16213e; border-left: 4px solid #0f3460; padding-left: .8rem; margin-top: 2.5rem; }
  h3 { color: #0f3460; margin-top: 1.5rem; }
  code { background: #f0f0f0; padding: .15em .4em; border-radius: 3px; font-size: .88em;
         font-family: "SF Mono",Menlo,"Fira Code",monospace; }
  pre { background: #1e1e2e; color: #cdd6f4; padding: 1.2rem; border-radius: 8px;
        overflow-x: auto; font-size: .85em; line-height: 1.5; }
  pre code { background: none; color: inherit; padding: 0; }
  img { max-width: 100%; border-radius: 6px; box-shadow: 0 3px 12px rgba(0,0,0,.12);
        margin: 1rem auto; display: block; }
  .frame-caption { text-align: center; color: #666; font-size: .85em; margin-top: -.5rem;
                   margin-bottom: 1.5rem; }
  blockquote { background: #e8f4fd; border-left: 4px solid #2196f3; padding: .8rem 1rem;
               margin: 1rem 0; border-radius: 0 8px 8px 0; }
  .important-box { background: #fff8e1; border-left: 4px solid #ffb300; padding: 1rem;
                   border-radius: 0 8px 8px 0; margin: 1rem 0; }
  .important-box::before { content: "\\2B50  \\91CD\\70B9"; font-weight: bold; display: block;
                           margin-bottom: .5rem; color: #e65100; }
  .knowledge-box { background: #e3f2fd; border-left: 4px solid #1565c0; padding: 1rem;
                   border-radius: 0 8px 8px 0; margin: 1rem 0; }
  .knowledge-box::before { content: "\\1F4D6  \\77E5\\8BC6\\8865\\5145"; font-weight: bold;
                           display: block; margin-bottom: .5rem; color: #0d47a1; }
  .warning-box { background: #fce4ec; border-left: 4px solid #c62828; padding: 1rem;
                 border-radius: 0 8px 8px 0; margin: 1rem 0; }
  .warning-box::before { content: "\\26A0\\FE0F  \\6CE8\\610F"; font-weight: bold;
                         display: block; margin-bottom: .5rem; color: #b71c1c; }
  table { border-collapse: collapse; width: 100%; margin: 1rem 0; }
  th, td { border: 1px solid #ddd; padding: .6rem .8rem; text-align: left; }
  th { background: #0f3460; color: white; }
  tr:nth-child(even) { background: #f8f9fa; }
  .cover { text-align: center; padding: 3rem 0; page-break-after: always; }
  .cover h1 { border: none; font-size: 2.2rem; }
  .cover img { max-height: 300px; margin: 2rem auto; }
  .toc { page-break-after: always; }
  @media print { body { max-width: 100%; padding: 0; } pre { white-space: pre-wrap; } }
</style>"""

    def build_html(
        self,
        notes_md: str,
        frames: list[ScoredFrame],
        title: str,
        meta: dict[str, str] | None = None,
        cover_path: Path | None = None,
    ) -> str:
        if meta is None:
            meta = {}

        # Replace {IMAGE:N} markers with actual images
        def _replace_img(match):
            n = int(match.group(1))
            if 1 <= n <= len(frames):
                fr = frames[n - 1]
                abs_path = str(fr.path.resolve())
                cap = fr.caption or fr.subtitle_text or f"video {fr.time_str}"
                return (
                    f'\n<img src="file://{abs_path}" alt="frame {n}">\n'
                    f'<div class="frame-caption">Fig.{n} — {cap} ({fr.time_str})</div>\n'
                )
            return f"*(Fig.{n})*"

        body = re.sub(r"\{IMAGE:(\d+)\}", _replace_img, notes_md)

        # Convert highlight boxes
        body = re.sub(
            r"\{IMPORTANT\}(.*?)\{/IMPORTANT\}",
            r'<div class="important-box">\1</div>',
            body, flags=re.DOTALL,
        )
        body = re.sub(
            r"\{KNOWLEDGE\}(.*?)\{/KNOWLEDGE\}",
            r'<div class="knowledge-box">\1</div>',
            body, flags=re.DOTALL,
        )
        body = re.sub(
            r"\{WARNING\}(.*?)\{/WARNING\}",
            r'<div class="warning-box">\1</div>',
            body, flags=re.DOTALL,
        )

        # Build cover
        cover_html = ""
        if cover_path and cover_path.exists():
            cover_html = f'<img src="file://{cover_path.resolve()}" alt="cover">'
        meta_rows = "\n".join(
            f"<tr><td><strong>{k}</strong></td><td>{v}</td></tr>"
            for k, v in meta.items()
        )

        html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>{title}</title>
{self.STYLE}
</head>
<body>
<div class="cover">
<h1>{title}</h1>
{cover_html}
<table style="max-width:500px;margin:2rem auto">{meta_rows}</table>
</div>
{body}
</body>
</html>"""
        return html

    @staticmethod
    def html_to_pdf(html_path: Path, pdf_path: Path) -> bool:
        chrome = shutil.which("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")
        if not chrome:
            chrome = shutil.which("chromium") or shutil.which("google-chrome")
        if not chrome:
            return False
        cmd = [
            chrome, "--headless=new",
            f"--print-to-pdf={pdf_path}",
            "--print-to-pdf-no-header", "--no-sandbox", "--disable-gpu",
            f"file://{html_path.resolve()}",
        ]
        r = subprocess.run(cmd, capture_output=True, timeout=60)
        return pdf_path.exists() and pdf_path.stat().st_size > 5000


# ── LaTeX Note Builder ─────────────────────────────────────────

class LaTeXNoteBuilder:
    """Build illustrated lecture notes as LaTeX -> xelatex -> PDF."""

    def build_tex(
        self,
        notes_md: str,
        frames: list[ScoredFrame],
        title: str,
        meta: dict[str, str] | None = None,
        cover_path: Path | None = None,
        template_path: Path | None = None,
    ) -> str:
        if meta is None:
            meta = {}

        if template_path and template_path.exists():
            tex = template_path.read_text(encoding="utf-8")
        else:
            tex = self._default_template()

        # Fill metadata
        tex = tex.replace("[TITLE]", _tex_escape(title))
        tex = tex.replace("[DATE]", time.strftime("%Y-%m-%d"))
        tex = tex.replace("[CHANNEL]", _tex_escape(meta.get("uploader", "")))
        tex = tex.replace("[DURATION]", meta.get("duration", ""))
        tex = tex.replace("[URL]", meta.get("url", ""))

        if cover_path and cover_path.exists():
            tex = tex.replace("[COVER_PATH]", str(cover_path.resolve()))
        else:
            tex = tex.replace("[COVER_PATH]", "")

        # Build body from markdown-ish notes
        body = self._md_to_tex(notes_md, frames)

        tex = tex.replace("[BODY]", body)
        return tex

    def compile_pdf(self, tex_path: Path, work_dir: Path) -> Path | None:
        xelatex = shutil.which("xelatex")
        if not xelatex:
            return None
        cmd = [
            xelatex,
            "-interaction=nonstopmode",
            "-output-directory", str(work_dir),
            str(tex_path),
        ]
        for _ in range(2):  # Two passes for TOC
            subprocess.run(cmd, capture_output=True, timeout=120, cwd=str(work_dir))
        pdf = tex_path.with_suffix(".pdf")
        if not pdf.exists():
            pdf = work_dir / tex_path.with_suffix(".pdf").name
        return pdf if pdf.exists() else None

    def _md_to_tex(self, md: str, frames: list[ScoredFrame]) -> str:
        """Convert simplified markdown with {IMAGE:N} to LaTeX body."""
        # Fix 1: Expand inline {TAG}content{/TAG} on same line into multi-line form
        # so the line-by-line parser below can handle them correctly.
        BOX_TAGS = {
            "IMPORTANT": ("importantbox", "Key Point"),
            "KNOWLEDGE": ("knowledgebox", "Background"),
            "WARNING": ("warningbox", "Caution"),
        }
        for tag in BOX_TAGS:
            md = re.sub(
                rf"\{{{tag}\}}(.*?)\{{/{tag}\}}",
                rf"{{{tag}}}\n\1\n{{/{tag}}}",
                md, flags=re.DOTALL,
            )

        lines = md.split("\n")
        tex_lines = []
        in_code = False
        in_table = False  # Fix 2: track table state to wrap with tabular env
        is_header = False
        code_lang = ""
        open_boxes: list[str] = []  # Fix 1: track unclosed boxes

        for line in lines:
            stripped = line.strip()

            # Code blocks
            if stripped.startswith("```"):
                if in_code:
                    tex_lines.append("\\end{lstlisting}")
                    in_code = False
                else:
                    code_lang = stripped.replace("```", "").strip() or "python"
                    tex_lines.append(f"\\begin{{lstlisting}}[language={code_lang}]")
                    in_code = True
                continue

            if in_code:
                tex_lines.append(line)
                continue

            # Image markers
            img_match = re.match(r"\s*\{IMAGE:(\d+)\}", stripped)
            if img_match:
                n = int(img_match.group(1))
                if 1 <= n <= len(frames):
                    fr = frames[n - 1]
                    cap = fr.caption or fr.subtitle_text or f"video {fr.time_str}"
                    tex_lines.append("\\begin{figure}[H]")
                    tex_lines.append("\\centering")
                    tex_lines.append(
                        f"\\includegraphics[width=0.9\\textwidth]{{{fr.path.resolve()}}}"
                    )
                    tex_lines.append(
                        f"\\caption{{{_tex_escape(cap)} \\protect\\footnotemark}}"
                    )
                    tex_lines.append("\\end{figure}")
                    tex_lines.append(
                        f"\\footnotetext{{video time: {fr.time_str}}}"
                    )
                continue

            # Fix 1: Highlight box open/close tags (now handles multi-line form)
            box_handled = False
            for tag, (env, label) in BOX_TAGS.items():
                if stripped == f"{{{tag}}}":
                    tex_lines.append(f"\\begin{{{env}}}{{{label}}}")
                    open_boxes.append(env)
                    box_handled = True
                    break
                if stripped == f"{{/{tag}}}":
                    tex_lines.append(f"\\end{{{env}}}")
                    if open_boxes and open_boxes[-1] == env:
                        open_boxes.pop()
                    box_handled = True
                    break
            if box_handled:
                continue

            # Fix 2: Tables — detect rows and wrap in tabular environment
            if "|" in stripped and stripped.startswith("|"):
                cells = [c.strip() for c in stripped.split("|")[1:-1]]
                if all(set(c) <= set("-: ") for c in cells):
                    continue  # skip separator row
                if not in_table:
                    col_spec = "|" + "l|" * len(cells)
                    tex_lines.append("\\begin{center}")
                    tex_lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
                    tex_lines.append("\\hline")
                    in_table = True
                    is_header = True
                row = " & ".join(_tex_escape(c) for c in cells)
                tex_lines.append(f"{row} \\\\")
                tex_lines.append("\\hline")
                is_header = False
                continue
            elif in_table:
                tex_lines.append("\\end{tabular}")
                tex_lines.append("\\end{center}")
                in_table = False

            # Horizontal rules
            if stripped in ("---", "***", "___"):
                tex_lines.append("\\bigskip\\hrule\\bigskip")
                continue

            # Blockquotes
            if stripped.startswith("> "):
                tex_lines.append(
                    f"\\begin{{quote}}\\textit{{{_tex_escape(stripped[2:])}}}\\end{{quote}}"
                )
                continue

            # Headings
            if stripped.startswith("# ") and not stripped.startswith("## "):
                tex_lines.append(f"\\section*{{{_tex_escape(stripped[2:].strip())}}}")
                continue
            if line.startswith("## "):
                tex_lines.append(f"\\section{{{_tex_escape(line[3:].strip())}}}")
                continue
            if line.startswith("### "):
                tex_lines.append(f"\\subsection{{{_tex_escape(line[4:].strip())}}}")
                continue
            if line.startswith("#### "):
                tex_lines.append(f"\\subsubsection{{{_tex_escape(line[5:].strip())}}}")
                continue

            # Bold/italic
            processed = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", line)
            processed = re.sub(r"\*(.+?)\*", r"\\textit{\1}", processed)
            # Inline code
            processed = re.sub(r"`([^`]+)`", r"\\texttt{\1}", processed)
            # Inline math
            processed = re.sub(r"\$(.+?)\$", r"$\1$", processed)

            tex_lines.append(processed)

        # Fix 2: close any unclosed table
        if in_table:
            tex_lines.append("\\end{tabular}")
            tex_lines.append("\\end{center}")

        # Fix 1: close any unclosed boxes
        for env in reversed(open_boxes):
            tex_lines.append(f"\\end{{{env}}}")

        return "\n".join(tex_lines)

    @staticmethod
    def _default_template() -> str:
        return r"""\documentclass[a4paper,11pt]{article}
\usepackage[fontset=fandol]{ctex}
\usepackage{amsmath,amssymb}
\usepackage{graphicx}
\usepackage[margin=2.5cm]{geometry}
\usepackage[most]{tcolorbox}
\usepackage{listings}
\usepackage{hyperref}
\usepackage{booktabs}
\usepackage{float}
\usepackage{fancyhdr}
\usepackage{tikz}

\newtcolorbox{knowledgebox}[1]{enhanced,colback=blue!5!white,colframe=blue!75!black,
  colbacktitle=blue!75!black,coltitle=white,fonttitle=\bfseries,title=#1,
  attach boxed title to top left={yshift=-2mm,xshift=2mm},boxrule=1pt,sharp corners}
\newtcolorbox{importantbox}[1]{enhanced,colback=yellow!10!white,colframe=yellow!80!black,
  colbacktitle=yellow!80!black,coltitle=black,fonttitle=\bfseries,title=#1,sharp corners}
\newtcolorbox{warningbox}[1]{enhanced,colback=red!5!white,colframe=red!75!black,
  colbacktitle=red!75!black,coltitle=white,fonttitle=\bfseries,title=#1,sharp corners}

\lstset{language=Python,basicstyle=\ttfamily\small,keywordstyle=\color{blue},
  stringstyle=\color{red!60!black},commentstyle=\color{green!60!black},
  breaklines=true,frame=single,numbers=left,numberstyle=\tiny\color{gray},captionpos=b}

\pagestyle{fancy}
\fancyhead[L]{\small [TITLE]}
\fancyhead[R]{\small NoteKing}
\fancyfoot[C]{\thepage}

\begin{document}
\begin{titlepage}\centering
{\Large Course Notes\par}\vspace{1cm}
{\huge\bfseries [TITLE]\par}\vspace{.8cm}
{\large NoteKing\par}\vspace{.3cm}
{\large [DATE]\par}\vspace{1.5cm}
\end{titlepage}
\tableofcontents\newpage
[BODY]
\end{document}
"""


# ── PDF Pipeline ───────────────────────────────────────────────

class PDFPipeline:
    """Full pipeline: download -> subtitles -> frames -> LLM notes -> render PDF."""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.minimax.chat/v1",
        model: str = "MiniMax-M2.7",
        max_tokens: int = 5000,
        concurrency: int = 3,
    ):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.max_tokens = max_tokens
        self.concurrency = concurrency
        self.frame_extractor = SmartFrameExtractor()
        self.aligner = SubtitleFrameAligner()
        self.html_builder = HTMLNoteBuilder()
        self.latex_builder = LaTeXNoteBuilder()

    def process_episode(
        self,
        video_url: str,
        episode_num: int,
        title: str,
        work_dir: Path,
        output_dir: Path,
        total_episodes: int = 1,
        subtitle_text: str = "",
    ) -> EpisodeResult:
        """Process one episode: download video -> extract frames -> LLM -> PDF."""
        ep_work = work_dir / f"ep{episode_num:02d}"
        ep_work.mkdir(parents=True, exist_ok=True)
        frames_dir = ep_work / "frames"

        # Step 1: Download video
        video_path = ep_work / "video.mp4"
        if not (video_path.exists() and video_path.stat().st_size > 50000):
            self._download_video(video_url, video_path)

        duration = _get_duration(video_path) if video_path.exists() else 0

        # Step 2: Extract smart keyframes
        if video_path.exists():
            frames = self.frame_extractor.extract(video_path, frames_dir)
        else:
            frames = []

        # Step 3: Generate notes with LLM
        notes_md = self._generate_notes(
            episode_num, title, duration, subtitle_text, frames, total_episodes
        )

        # Step 4: Build HTML + PDF
        meta = {
            "Episode": f"{episode_num}/{total_episodes}",
            "Title": title,
            "Duration": f"{int(duration//60)}m{int(duration%60)}s",
        }

        html_content = self.html_builder.build_html(notes_md, frames, title, meta)
        html_path = output_dir / f"ep{episode_num:02d}_{_safe(title)}.html"
        html_path.write_text(html_content, encoding="utf-8")

        pdf_path = html_path.with_suffix(".pdf")
        HTMLNoteBuilder.html_to_pdf(html_path, pdf_path)

        result = EpisodeResult(
            episode=episode_num,
            title=title,
            duration=duration,
            frames=frames,
            subtitle_text=subtitle_text,
            notes_md=notes_md,
            html_path=html_path,
            pdf_path=pdf_path if pdf_path.exists() else None,
        )
        return result

    def process_collection(
        self,
        episodes: list[dict],
        work_dir: Path,
        output_dir: Path,
        collection_title: str = "Video Collection",
    ) -> CollectionResult:
        """Process multiple episodes, optionally concurrently."""
        output_dir.mkdir(parents=True, exist_ok=True)
        total = len(episodes)
        results: list[EpisodeResult] = [None] * total  # type: ignore

        def _do_one(idx: int) -> EpisodeResult:
            ep = episodes[idx]
            print(f"  [{idx+1}/{total}] {ep['title'][:40]}...")
            r = self.process_episode(
                video_url=ep["url"],
                episode_num=idx + 1,
                title=ep["title"],
                work_dir=work_dir,
                output_dir=output_dir / "episodes",
                total_episodes=total,
                subtitle_text=ep.get("subtitle_text", ""),
            )
            print(f"  [{idx+1}/{total}] done ({len(r.notes_md)} chars)")
            return r

        # Concurrent execution
        with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
            futures = {pool.submit(_do_one, i): i for i in range(total)}
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    print(f"  [{idx+1}/{total}] FAILED: {e}")
                    results[idx] = EpisodeResult(
                        episode=idx + 1,
                        title=episodes[idx]["title"],
                        duration=0,
                        frames=[],
                        subtitle_text="",
                        notes_md=f"*Generation failed: {e}*",
                    )

        # Merge into one big HTML
        merged_html = self._merge_html(results, collection_title, output_dir)
        merged_pdf = merged_html.with_suffix(".pdf")
        HTMLNoteBuilder.html_to_pdf(merged_html, merged_pdf)

        return CollectionResult(
            title=collection_title,
            episodes=results,
            merged_pdf=merged_pdf if merged_pdf.exists() else None,
            merged_html=merged_html,
        )

    # ── Internal helpers ────────────────────────────────────────

    def _download_video(self, url: str, output: Path) -> None:
        cmd = (
            f'yt-dlp "{url}" '
            f'-f "bestvideo[height<=720]+bestaudio/best[height<=720]" '
            f'--merge-output-format mp4 '
            f'-o "{output}" --no-warnings --quiet'
        )
        subprocess.run(cmd, shell=True, capture_output=True, timeout=300)

    def _generate_notes(
        self,
        ep_num: int,
        title: str,
        duration: float,
        subtitle_text: str,
        frames: list[ScoredFrame],
        total: int,
    ) -> str:
        frames_desc = ""
        if frames:
            frames_desc = "Available keyframes:\n" + "\n".join(
                f"  Fig.{i+1} at {f.time_str}: {f.subtitle_text[:60] or '(no subtitle)'}"
                for i, f in enumerate(frames)
            )

        mins = int(duration // 60)
        secs = int(duration % 60)

        prompt = f"""Generate detailed illustrated lecture notes for this video episode.

Episode: {ep_num}/{total} "{title}" ({mins}m{secs}s)

Transcript:
{subtitle_text[:6000] if subtitle_text else "(No transcript; generate notes based on the title and topic)"}

{frames_desc}

FORMAT REQUIREMENTS:
1. Use ## for sections, ### for subsections
2. Insert {{IMAGE:N}} where figure N should appear (e.g. {{IMAGE:1}}, {{IMAGE:2}})
3. Use highlight boxes for key concepts:
   {{IMPORTANT}}Key concept text{{/IMPORTANT}}
   {{KNOWLEDGE}}Background knowledge{{/KNOWLEDGE}}
   {{WARNING}}Common pitfall{{/WARNING}}
4. Include code blocks with ``` when code appears
5. Use LaTeX math ($...$) for formulas
6. End each section with a brief summary
7. Write in Chinese

CONTENT REQUIREMENTS:
- Detailed, structured, suitable for serious study
- Each figure reference must have a descriptive caption below it
- Include code with detailed comments when relevant
- Cover all major points from the transcript
- Add a final summary section with key takeaways"""

        return self._call_llm(prompt)

    def _call_llm(self, prompt: str, retries: int = 3) -> str:
        system = "You are a professional technical education expert. Output lecture notes directly in Markdown format with the specified markers. Do not wrap output in think tags."

        for attempt in range(retries):
            try:
                full = ""
                stream = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=self.max_tokens,
                    stream=True,
                    temperature=0.2,
                    timeout=120,
                )
                for chunk in stream:
                    delta = chunk.choices[0].delta.content or ""
                    full += delta
                # Strip <think> blocks
                full = re.sub(r"<think>.*?</think>", "", full, flags=re.DOTALL).strip()
                if len(full) > 200:
                    return full
            except Exception as e:
                if attempt < retries - 1:
                    time.sleep((attempt + 1) * 8)
                else:
                    return f"*Note generation failed after {retries} attempts: {e}*"
        return ""

    def _merge_html(
        self, episodes: list[EpisodeResult], title: str, output_dir: Path
    ) -> Path:
        """Merge all episodes into one big HTML document."""
        parts = [f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>{title}</title>
{HTMLNoteBuilder.STYLE}
</head><body>
<div class="cover"><h1>{title}</h1>
<p>Generated by NoteKing &middot; {time.strftime('%Y-%m-%d')}</p>
</div>
<div class="toc"><h2>Table of Contents</h2><ol>"""]

        for ep in episodes:
            if ep:
                parts.append(f'<li><a href="#ep{ep.episode}">{ep.title}</a></li>')
        parts.append("</ol></div>")

        for ep in episodes:
            if not ep:
                continue
            parts.append(f'<h1 id="ep{ep.episode}">Ep.{ep.episode}: {ep.title}</h1>')

            body = ep.notes_md
            # Replace image markers with actual images
            def _rep(m, ep=ep):
                n = int(m.group(1))
                if 1 <= n <= len(ep.frames):
                    fr = ep.frames[n - 1]
                    return (
                        f'\n<img src="file://{fr.path.resolve()}" alt="frame">\n'
                        f'<div class="frame-caption">Fig.{n} — {fr.time_str}</div>\n'
                    )
                return ""
            body = re.sub(r"\{IMAGE:(\d+)\}", _rep, body)
            body = re.sub(r"\{IMPORTANT\}(.*?)\{/IMPORTANT\}",
                          r'<div class="important-box">\1</div>', body, flags=re.DOTALL)
            body = re.sub(r"\{KNOWLEDGE\}(.*?)\{/KNOWLEDGE\}",
                          r'<div class="knowledge-box">\1</div>', body, flags=re.DOTALL)
            body = re.sub(r"\{WARNING\}(.*?)\{/WARNING\}",
                          r'<div class="warning-box">\1</div>', body, flags=re.DOTALL)
            parts.append(body)
            parts.append("<hr>")

        parts.append("</body></html>")

        out = output_dir / f"{_safe(title)}_full.html"
        out.write_text("\n".join(parts), encoding="utf-8")
        return out


# ── Helpers ────────────────────────────────────────────────────

def _safe(text: str, max_len: int = 40) -> str:
    return "".join(c if c.isalnum() or c in "_- " else "_" for c in text)[:max_len].strip()


def _tex_escape(text: str) -> str:
    specials = {"&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
                "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}",
                "^": r"\^{}"}
    for k, v in specials.items():
        text = text.replace(k, v)
    return text
