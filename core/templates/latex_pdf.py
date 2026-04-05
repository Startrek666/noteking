"""LaTeX PDF lecture notes template (text-only, no images)."""

from .base import BaseTemplate, TemplateContext, _truncate_transcript


class LaTeXPDFTemplate(BaseTemplate):
    name = "latex_pdf"
    display_name = "LaTeX PDF 讲义"
    description = "Professional LaTeX lecture notes compiled to PDF"
    file_extension = ".tex"

    def build_prompt(self, ctx: TemplateContext) -> str:
        transcript = _truncate_transcript(ctx.transcript)
        chapters_info = ""
        if ctx.has_chapters:
            chapters_info = "\n视频章节:\n"
            for ch in ctx.meta.chapters:
                t = ch.get("title", "")
                s = ch.get("start_time", 0)
                chapters_info += f"- [{s:.0f}s] {t}\n"

        return f"""请根据以下视频内容，生成一份完整的 LaTeX 讲义文档。

视频标题: {ctx.meta.title}
作者: {ctx.meta.uploader}
时长: {ctx.meta.duration:.0f} 秒
{chapters_info}
视频内容转录:
{transcript}

要求:
1. 生成完整的 .tex 文档，从 \\documentclass 到 \\end{{document}}
2. 使用 ctexart 文档类（支持中文）
3. 按教学逻辑重组内容，而非简单按字幕时间顺序排列
4. 每个主要章节结构:
   - 动机: 为什么要学这个
   - 核心概念: 主要思想
   - 机制/方法: 如何工作
   - 示例: 具体案例
   - 小结: 本节要点
5. 公式使用 $$...$$ 或 \\begin{{equation}}，并解释每个符号
6. 代码使用 \\begin{{lstlisting}} 环境
7. 重要概念用 \\textbf{{}} 强调
8. 每个主要 section 结尾加 \\subsection{{本章小结}}
9. 文档最后加 \\section{{总结与延伸}}
10. 禁止使用 \\includegraphics、\\begin{{figure}} 或任何图片引用命令
11. 不需要 graphicx 和 float 包
12. 表格用 tabular 环境，列表用 itemize/enumerate

LaTeX 文档头部必须包含以下所有包:
\\documentclass[12pt,a4paper]{{ctexart}}
\\usepackage{{amsmath,amssymb,listings,xcolor,geometry,hyperref}}
\\geometry{{margin=2.5cm}}
\\lstset{{basicstyle=\\ttfamily\\small,breaklines=true,frame=single}}
\\title{{{ctx.meta.title}}}
\\author{{笔记整理自 {ctx.meta.uploader} 的视频}}
\\date{{}}

请直接输出完整的 .tex 文档:"""
