# α、β 和 γ 散度族：中文全译本

原文为 Andrzej Cichocki 与 Shun-ichi Amari 的《Families of Alpha- Beta- and Gamma- Divergences: Flexible and Robust Measures of Similarities》，发表于 Entropy 2010 年第 12 卷，1532–1568 页[^paper]。本目录包含完整中文译文、可编辑 LaTeX、英文恢复稿与出版社原始资料。

## 源码来源

未找到公开的作者原始 `.tex` 文件。英文恢复稿由出版社公开 XML 全文[^xml]中的文字、MathML 公式和表格恢复，不能当作作者投稿源码。出版社 PDF[^pdf]与用户指定的 `runs/alpha-divergence-reconstruction-20261015/alpha.pdf` 的 SHA-256 相同，确认使用的是同一版本。

`original/entropy-12-01532.xml` 与 `original/entropy-12-01532.pdf` 原样归档。出处、散列与转换版本见 [来源清单](source-manifest.json)。原作版权归作者所有，以 CC BY 3.0 发布[^license]；本稿的改动为中文翻译、LaTeX 恢复和排版调整。作者姓名、原始参考文献题名与出版信息保留原文，便于检索。

## 阅读与编辑

- [中文 PDF](2026-10-15-α、β和γ散度族.pdf) 与 [中文 LaTeX 入口](2026-10-15-α、β和γ散度族.tex)。
- [中文分节正文](sections/) 与 [85 条原始参考文献](references.tex)。
- [英文恢复稿 PDF](original/recovered-latex/2026-10-15-families-of-alpha-beta-and-gamma-divergences.pdf) 与 [英文 LaTeX 入口](original/recovered-latex/2026-10-15-families-of-alpha-beta-and-gamma-divergences.tex)。

译文包含原文第 1–5 节及全部子节、附录 A–C、编号公式（1）—（128）、5 张表及全部参考文献。公式保留为可编辑 LaTeX，未使用公式截图。5 张长表采用横向分页；较长的行内定义改为不编号的独立公式，编号公式与交叉引用保持原编号。中文分页与原文不同。

正文按原文翻译，作者的理论结论保持原有陈述及适用条件；这里的完整性检查与编译检查不构成对原文全部结论的重新证明。

## 编译

需要 XeLaTeX、latexmk、ctex，以及 Noto Serif CJK SC 和 Noto Sans CJK SC 字体。英文恢复稿仅需相应的英文 LaTeX 与字体组件。进入本目录执行：

```bash
make
make english
```

两条命令分别编译中文稿和英文恢复稿。`make clean` 清理两份稿件的构建中间文件并保留 PDF。编辑入口文件、`sections/` 和 `references.tex` 后，重新执行相应命令即可；编译不需要联网或 Pandoc。

### 参考文献

[^paper]: Cichocki, A.; Amari, S. Families of Alpha- Beta- and Gamma- Divergences: Flexible and Robust Measures of Similarities. Entropy 2010, 12(6), 1532–1568. [DOI: 10.3390/e12061532](https://doi.org/10.3390/e12061532).

[^xml]: MDPI. 出版社 XML 全文. [entropy-12-01532.xml](https://mdpi-res.com/d_attachment/entropy/entropy-12-01532/article_deploy/entropy-12-01532.xml).

[^pdf]: MDPI. 出版社原始 PDF. [entropy-12-01532.pdf](https://mdpi-res.com/d_attachment/entropy/entropy-12-01532/article_deploy/entropy-12-01532.pdf).

[^license]: Creative Commons. Attribution 3.0 Unported. [CC BY 3.0](https://creativecommons.org/licenses/by/3.0/).
