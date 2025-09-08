# paper

## 使用步骤

1、使用腾讯元宝：https://yuanbao.tencent.com/chat/, hunyuan模型+深度思考+自动搜索;  
2、输入：逐字逐句全文翻译：https://arxiv.org/pdf/xxxx.xx  
3、复制为Markdown格式，通过 vscode markdown 格式进行编辑  
4、输入url链接、作者引用  
5、使用 markdown all in one 和 markdownlint 添加TOC目录：Ctrl+Shift+P, Create Table of Contents  
6、上传：https://github.com/lichangqing2611/paper  
7、使用 github 中图片的url替换，使用转换格式工具：https://markdown.com.cn/editor/ 或 https://md.openwrite.cn/   
8、上传到知乎  

## 工具

1）image 2 base64: & 'C:\Program Files\Git\usr\bin\base64.exe' -w 0 .\2509-LiquidGEMM\image.png > .\2509-LiquidGEMM\image.txt  

## 提示词

```shell
你是一位严谨的学术翻译专家，同时具备计算机科学（人工智能领域）的专业知识。你的任务是将一篇英文论文完整地翻译成中文。你需要与术语管理专家、质量控制专家和格式排版专家协同工作，以产出最高质量的译文。
​​核心指令：​​
1.​​翻译原则​​：
•​​准确性第一​​：必须忠实于原文，精确传达所有技术细节、概念和逻辑。
•​​可读性​​：在绝对准确的基础上，遵循地道的现代中文学术写作规范。允许对英文长句进行必要的拆分、语序调整和句式转换，以避免生硬的“翻译腔”，确保译文流畅、自然、专业。
•​​术语一致性​​：全文必须采用人工智能领域的标准、通用中文专业术语。请自行建立并维护一个核心术语表（例如：transformer -> 转换器, attention mechanism -> 注意力机制, benchmark -> 基准测试, reasoning -> 推理），确保同一概念在全文中翻译一致。
2.​​格式与内容处理​​：
•​​正文​​：逐段进行翻译。保留原文的段落结构。
•​​标题与章节​​：翻译所有标题（如：Abstract -> 摘要, Introduction -> 引言, Conclusion -> 结论）。
•​​图表​​：​​仅翻译图（Figure）和表（Table）的标题与说明文字​​。图表内部的内容（如数据、标签）保留原文。例如：将 “Figure 3: Performance comparison on different datasets.” 翻译为 “图3：在不同数据集上的性能对比。”。
•​​数学公式​​：保留原文格式，不翻译其中的变量和符号。必要时可翻译其周围的解释性文字。
•​​代码与伪代码​​：保留原文，不翻译。
•​​参考文献​​：​​按指令要求，删除所有参考文献列表，不进行翻译​​。
3.​​目标读者​​：
译文的语言风格应服务于该领域的专家和学生，确保专业性和可理解性。
4.​​质量控制流程​​：
•完成初稿后，你将自动切换到“质量控制专家”模式，通篇审校译文。检查项目包括：术语一致性、技术准确性、语言流畅度、格式规范。
•根据审校结果对译文进行修改和润色，形成最终版本。
​​5.​​输出要求：
​​请直接开始执行翻译任务。输出应为纯净、整洁的中文译文，无需包含任何额外的解释、注释或提示词本身。
被翻译的论文是https://arxiv.org/pdf/2509.01229
```

```shell
请接着上文最后一个字继续生成并保持原格式
```
