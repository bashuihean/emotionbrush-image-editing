# 方法来源、依赖和权利边界

本项目基于已有公开方法和软件实现。公开代码的版权声明只覆盖作者有权声明的原创部分。
依赖包与基础模型的许可证分别适用；安装、下载与使用权重前应阅读其仓库及模型卡。
不随本仓库再分发第三方权重、客户数据、私有分类器或客户结果图。

| 方法/项目 | 来源 | 本项目中的关系 |
|---|---|---|
| InstructPix2Pix | https://github.com/timothybrooks/instruct-pix2pix | 预训练 8 通道扩散主干 |
| DINOv2 | https://github.com/facebookresearch/dinov2 | 冻结内容局部特征 |
| BLIP-2 / QFormer | https://arxiv.org/abs/2301.12597 | 使用 Transformers QFormer 结构，不是 Qwen |
| LoRA | https://arxiv.org/abs/2106.09685 | 内容/风格低秩适配设计基础 |
| EmoEdit | https://github.com/JingyuanYY/EmoEdit | 情感编辑研究背景及数据流程参考 |
| StyleShot | https://github.com/open-mmlab/StyleShot | 风格提取思路参考，当前 StyleEncoder 为项目实现 |
| UniCombine | https://github.com/Xuan-World/UniCombine | 早期方案讨论参考；当前主干不是 MMDiT |
| BiRefNet | https://github.com/ZhengPeng7/BiRefNet | 主体软掩码，需要自行下载模型 |

实现依赖 torch、torchvision、diffusers、transformers、huggingface_hub、numpy、Pillow、tqdm 等。
这份方法来源表不是完整的第三方版权法律审查，也不把方法参考等同于复制源代码。
metrics.py 中保留的旧 ESMI 路径来自历史 EmoEditor 评测流程兼容，
不能宣称该指标为本项目原创或已核对为论文官方定义；来源/授权如有争议请联系维护者核查。
