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


def strip_all_images(tex: str) -> str:
    """彻底移除所有图片相关内容，防止编译因缺失图片而中断。"""
    # 移除整个 figure 环境
    tex = re.sub(
        r'\\begin\{figure\}.*?\\end\{figure\}',
        '',
        tex, flags=re.DOTALL
    )
    # 移除独立的 \includegraphics 命令
    tex = re.sub(r'\\includegraphics(\[.*?\])?\{[^}]*\}', '', tex)
    # 移除 HTML img 标签（LLM 偶尔生成）
    tex = re.sub(r'<img\s+[^>]*/?>',  '', tex, flags=re.IGNORECASE)
    # 移除孤立的 \caption（不在 figure 内的）
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


@app.route("/compile", methods=["POST"])
def compile_latex():
    data = request.get_json()
    if not data or "tex_content" not in data:
        return jsonify({"error": "缺少 tex_content"}), 400

    tex_content = clean_tex_content(data["tex_content"])
    filename = data.get("filename", "noteking_notes")

    if '\\begin{document}' not in tex_content:
        return jsonify({"error": "LaTeX 内容无效：缺少 \\begin{document}"}), 422

    tex_content = strip_all_images(tex_content)
    tex_content = inject_branding(tex_content)

    with tempfile.TemporaryDirectory(prefix="latex_") as tmpdir:
        tex_path = Path(tmpdir) / "notes.tex"
        tex_path.write_text(tex_content, encoding="utf-8")

        for _ in range(2):
            result = subprocess.run(
                ["xelatex", "-interaction=nonstopmode",
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
