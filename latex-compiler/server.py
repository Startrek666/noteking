import base64
import os
import re
import subprocess
import tempfile
from pathlib import Path
from flask import Flask, request, send_file, jsonify

app = Flask(__name__)

BRANDING_PREAMBLE = r"""
\usepackage{fancyhdr}
\usepackage{lastpage}
\pagestyle{fancy}
\fancyhf{}
\fancyhead[L]{\small\textbf{NoteKing 笔记之王}}
\fancyhead[R]{\small 小红书: bcefghj}
\fancyfoot[L]{\small\texttt{github.com/bcefghj/noteking}}
\fancyfoot[C]{\small 第 \thepage\ 页 / 共 \pageref{LastPage} 页}
\fancyfoot[R]{\small NoteKing · 视频一键生成学习笔记}
\renewcommand{\headrulewidth}{0.4pt}
\renewcommand{\footrulewidth}{0.4pt}
"""


def clean_tex_content(raw: str) -> str:
    text = raw.strip()
    text = re.sub(r'^```(?:latex|tex)?\s*\n', '', text)
    text = re.sub(r'\n```\s*$', '', text)
    text = text.strip()
    if '\\documentclass' not in text:
        match = re.search(r'(\\documentclass.*)', text, re.DOTALL)
        if match:
            text = match.group(1)
    return text


def inject_branding(tex: str) -> str:
    if '\\pagestyle{fancy}' in tex:
        return tex
    if 'fancyhdr' in tex and '\\fancyhead' in tex:
        return tex
    tex = re.sub(r'\\usepackage(\[.*?\])?\{fancyhdr\}', '', tex)
    tex = re.sub(r'\\usepackage(\[.*?\])?\{lastpage\}', '', tex)
    m = re.search(r'\\begin\{document\}', tex)
    if m:
        tex = tex[:m.start()] + BRANDING_PREAMBLE + '\n' + tex[m.start():]
    return tex


def ensure_graphicx(tex: str) -> str:
    """确保 graphicx 和 float 包存在。"""
    if 'graphicx' in tex:
        return tex
    tex = re.sub(
        r'(\\usepackage\{amsmath[^}]*\})',
        r'\1\n\\usepackage{graphicx,float}',
        tex, count=1
    )
    if 'graphicx' not in tex:
        m = re.search(r'\\begin\{document\}', tex)
        if m:
            tex = tex[:m.start()] + '\\usepackage{graphicx,float}\n' + tex[m.start():]
    return tex


def fix_html_img_tags(tex: str) -> str:
    """把 LLM 错误生成的 HTML <img> 标签转换为正确的 LaTeX includegraphics 命令。"""
    pattern = re.compile(
        r'<img\s+src=["\']([^"\']+)["\'][^>]*/?>',
        re.IGNORECASE
    )

    def replace_img(m):
        src = m.group(1)
        return (
            "\\begin{figure}[H]\n"
            "\\centering\n"
            "\\includegraphics[width=0.85\\textwidth]{" + src + "}\n"
            "\\end{figure}"
        )

    return pattern.sub(replace_img, tex)


def remove_invalid_figure_captions(tex: str) -> str:
    """移除孤立的 \\caption 命令（不在 figure 环境内的）。"""
    lines = tex.split('\n')
    in_figure = False
    result = []
    for line in lines:
        if '\\begin{figure}' in line:
            in_figure = True
        if '\\end{figure}' in line:
            in_figure = False
        if '\\caption{' in line and not in_figure:
            continue
        result.append(line)
    return '\n'.join(result)


def remove_missing_image_figures(tex: str, available_files: set) -> str:
    """把引用了不存在图片的整个 figure 环境删掉，防止编译报 Unable to load picture。"""
    # 匹配完整的 figure 环境（含 H/h/t/b 等位置参数）
    figure_pattern = re.compile(
        r'\\begin\{figure\}.*?\\end\{figure\}',
        re.DOTALL
    )

    def keep_figure(m: re.Match) -> str:
        block = m.group(0)
        # 找出 block 里所有 \includegraphics{...} 或 \includegraphics[...]{...}
        imgs = re.findall(r'\\includegraphics(?:\[.*?\])?\{([^}]+)\}', block)
        if not imgs:
            return block  # 没有图片引用，保留
        # 只要有一张图不在 available_files 里就整块删掉
        for img in imgs:
            img_name = Path(img).name
            if img_name not in available_files and img not in available_files:
                return ''  # 删掉整个 figure
        return block

    return figure_pattern.sub(keep_figure, tex)


@app.route("/compile", methods=["POST"])
def compile_latex():
    data = request.get_json()
    if not data or "tex_content" not in data:
        return jsonify({"error": "缺少 tex_content"}), 400

    tex_content = clean_tex_content(data["tex_content"])
    filename = data.get("filename", "noteking_notes")
    frames_b64: dict = data.get("frames_b64", {})

    if '\\begin{document}' not in tex_content:
        return jsonify({"error": "LaTeX 内容无效：缺少 \\begin{document}"}), 422

    # 第一步：修复 LLM 可能生成的 HTML img 标签
    tex_content = fix_html_img_tags(tex_content)
    tex_content = remove_invalid_figure_captions(tex_content)
    tex_content = inject_branding(tex_content)
    if frames_b64:
        tex_content = ensure_graphicx(tex_content)

    with tempfile.TemporaryDirectory(prefix="latex_") as tmpdir:
        # 把帧图片写入编译目录，记录实际写入的文件名
        written_frames: set = set()
        for fname, b64data in frames_b64.items():
            try:
                img_bytes = base64.b64decode(b64data)
                img_path = Path(tmpdir) / fname
                img_path.write_bytes(img_bytes)
                written_frames.add(fname)
            except Exception:
                pass

        # 第二步：删掉引用了不存在图片的 figure 环境（必须在写入图片后才知道哪些存在）
        tex_content = remove_missing_image_figures(tex_content, written_frames)

        tex_path = Path(tmpdir) / "notes.tex"
        tex_path.write_text(tex_content, encoding="utf-8")

        for _ in range(2):
            result = subprocess.run(
                ["xelatex", "-interaction=nonstopmode", "-halt-on-error",
                 "-output-directory", tmpdir, str(tex_path)],
                capture_output=True, text=True, timeout=120, cwd=tmpdir,
            )

        pdf_path = Path(tmpdir) / "notes.pdf"
        if not pdf_path.exists():
            log_path = Path(tmpdir) / "notes.log"
            log_tail = ""
            if log_path.exists():
                lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
                err = [l for l in lines if l.startswith("!")]
                log_tail = "\n".join(err[:20]) if err else "\n".join(lines[-30:])
            return jsonify({"error": f"LaTeX 编译失败:\n{log_tail}"}), 422

        safe = re.sub(r'[^\w\u4e00-\u9fff\-]', '_', filename)[:80]
        return send_file(str(pdf_path), mimetype="application/pdf",
                         as_attachment=True, download_name=f"{safe}.pdf")


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9090)
