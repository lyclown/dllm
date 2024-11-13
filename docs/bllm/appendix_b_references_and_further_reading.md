# 附录 B. 参考文献及延伸阅读

## B.1 第1章

自定义的LLM（大型语言模型）能够在特定领域的任务上优于通用LLM。Bloomberg团队通过在金融数据上从头预训练的GPT版本展示了这一点。这个定制的LLM在金融任务上超越了ChatGPT，同时在通用LLM基准测试中也保持了良好的表现：

-   **BloombergGPT: A Large Language Model for Finance (2023)** ，作者：Wu等人，<https://arxiv.org/abs/2303.17564>

现有的LLM也可以通过微调来适应特定领域，从而在特定任务上超越通用LLM。谷歌研究和DeepMind的团队在医疗领域展示了这一点：

-   **Towards Expert-Level Medical Question Answering with Large Language Models (2023)** ，作者：Singhal等人，<https://arxiv.org/abs/2305.09617>

提出最初Transformer架构的论文：

-   **Attention Is All You Need (2017)** ，作者：Vaswani等人，<https://arxiv.org/abs/1706.03762>

最早的编码器风格Transformer模型，称为BERT：

-   **BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding (2018)** ，作者：Devlin等人，<https://arxiv.org/abs/1810.04805>

描述解码器风格GPT-3模型的论文，该模型启发了现代LLM，并将作为本书中从头实现LLM的模板：

-   **Language Models are Few-Shot Learners (2020)** ，作者：Brown等人，<https://arxiv.org/abs/2005.14165>

用于图像分类的最早视觉Transformer，展示了Transformer架构不仅适用于文本输入：

-   **An images/image is Worth 16x16 Words: Transformers for images/image Recognition at Scale (2020)** ，作者：Dosovitskiy等人，<https://arxiv.org/abs/2010.11929>

两种实验性（但不太流行）的LLM架构示例，说明并非所有LLM都基于Transformer架构：

-   **RWKV: Reinventing RNNs for the Transformer Era (2023)** ，作者：Peng等人，<https://arxiv.org/abs/2305.13048>
-   **Hyena Hierarchy: Towards Larger Convolutional Language Models (2023)** ，作者：Poli等人，<https://arxiv.org/abs/2302.10866>
-   **Mamba: Linear-Time Sequence Modeling with Selective State Spaces (2023)** ，作者：Gu和Dao，<https://arxiv.org/abs/2312.00752>

Meta AI的模型是一种类似GPT的开源模型，与GPT-3和ChatGPT不同，它是公开可用的：

-   **Llama 2: Open Foundation and Fine-Tuned Chat Models (2023)** ，作者：Touvron等人，<https://arxiv.org/abs/2307.092881>

对于那些对1.5节中提到的数据集感兴趣的读者，以下论文介绍了Eleuther AI创建的公开可用的数据集The Pile：

-   **The Pile: An 800GB Dataset of Diverse Text for Language Modeling (2020)** ，作者：Gao等人，<https://arxiv.org/abs/2101.00027>

以下论文提供了InstructGPT的参考文献，InstructGPT用于GPT-3的微调，该模型在1.6节中提到，将在第7章中详细讨论：

-   **Training Language Models to Follow Instructions with Human Feedback (2022)** ，作者：Ouyang等人，<https://arxiv.org/abs/2203.02155>


## B.2 第2章

对嵌入空间、潜在空间及向量表示的概念感兴趣的读者，可以参考我书中《Machine Learning Q and AI》的第一章，了解更多信息：

-   **Machine Learning Q and AI (2023)** ，作者：Sebastian Raschka， <https://leanpub.com/machine-learning-q-and-ai>

以下论文深入探讨了如何使用字节对编码（BPE）作为一种分词方法：

-   **Neural Machine Translation of Rare Words with Subword Units (2015)** ，作者：Sennrich等人， <https://arxiv.org/abs/1508.07909>

OpenAI开源了用于训练GPT-2的字节对编码分词器代码：

-   <https://github.com/openai/gpt-2/blob/master/src/encoder.py>

OpenAI还提供了一个交互式网页界面，展示GPT模型中的字节对分词器的工作原理：

-   <https://platform.openai.com/tokenizer>

对从零开始编写和训练BPE分词器感兴趣的读者，可以参考Andrej Karpathy的GitHub仓库`minbpe`，提供了一个简洁易懂的实现：

-   **A minimal implementation of a BPE tokenizer**， <https://github.com/karpathy/minbpe>

有兴趣了解其他流行LLM所使用的分词方案的读者，可以参考以下论文：

-   **SentencePiece: A Simple and Language Independent Subword Tokenizer and Detokenizer for Neural Text Processing (2018)** ，作者：Kudo和Richardson， <https://aclanthology.org/D18-2012/>
-   **Fast WordPiece Tokenization (2020)** ，作者：Song等人， <https://arxiv.org/abs/2012.15524>

## B.3 第3章

对Bahdanau注意力机制及语言翻译中的应用感兴趣的读者，可以在以下论文中找到详细的信息：

-   **Neural Machine Translation by Jointly Learning to Align and Translate (2014)** ，作者：Bahdanau, Cho和Bengio， <https://arxiv.org/abs/1409.0473>

自注意力机制作为缩放点积注意力的概念首次提出于最初的Transformer论文：

-   **Attention Is All You Need (2017)** ，作者：Vaswani等人， <https://arxiv.org/abs/1706.03762>

FlashAttention是一个高效的自注意力机制实现，通过优化内存访问模式加速计算过程。FlashAttention在数学上与标准的自注意力机制相同，但在计算效率上进行了优化：

-   **FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness (2022)** ，作者：Dao等人， <https://arxiv.org/abs/2205.14135>
-   **FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning (2023)** ，作者：Dao， <https://arxiv.org/abs/2307.08691>

PyTorch实现了支持FlashAttention的自注意力和因果注意力功能。这些函数处于测试阶段，可能会有所更改：

-   **scaled_dot_product_attention documentation**， <https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html>

PyTorch还实现了基于`scaled_dot_product`函数的高效多头注意力类：

-   **MultiHeadAttention documentation**， <https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html>

Dropout是一种正则化技术，用于通过在训练过程中随机丢弃神经网络中的单元（及其连接）来防止过拟合：

-   **Dropout: A Simple Way to Prevent Neural Networks from Overfitting (2014)** ，作者：Srivastava等人， <https://jmlr.org/papers/v15/srivastava14a.html>

尽管基于缩放点积的多头注意力是自注意力中最常见的变体，作者发现去除value权重矩阵和投影层仍能实现较好的性能：

-   **Simplifying Transformer Blocks (2023)** ，作者：He和Hofmann， <https://arxiv.org/abs/2311.01906>

## B.4 第4章

标题为“Layer Normalization”的论文介绍了一种层归一化技术，它通过对隐藏层中的神经元输入求和并进行归一化来稳定网络的隐藏状态动态，与此前发布的方法相比显著缩短了训练时间：

-   **Layer Normalization (2016)** ，作者：Ba, Kiros和Hinton， <https://arxiv.org/abs/1607.06450>

后归一化（Post-LayerNorm）在原始Transformer模型中应用于自注意力和前馈网络之后。相比之下，前归一化（Pre-LayerNorm）在GPT-2及更先进的LLM中被应用于这些组件之前，这可以带来更稳定的训练动态，并在某些情况下提高性能，详细信息见以下论文：

-   **On Layer Normalization in the Transformer Architecture (2020)** ，作者：Xiong等人， <https://arxiv.org/abs/2002.04745>
-   **ResiDual: Transformer with Dual Residual Connections (2023)** ，作者：Tie等人， <https://arxiv.org/abs/2304.14802>

现代LLM中使用的一种流行的LayerNorm变体是RMSNorm，因为其计算效率更高。这种变体仅通过输入的均方根进行归一化，而不在平方之前减去均值，意味着数据在计算缩放之前不会居中。RMSNorm的详细介绍见以下论文：

-   **Root Mean Square Layer Normalization (2019)** ，作者：Zhang和Sennrich， <https://arxiv.org/abs/1910.07467>

GELU（Gaussian Error Linear Unit）激活函数结合了经典ReLU激活函数和正态分布累积分布函数的特性，能够对层输出进行建模，在深度学习模型中允许随机正则化和非线性，详见以下论文：

-   **Gaussian Error Linear Units (GELUs) (2016)** ，作者：Hendricks和Gimpel， <https://arxiv.org/abs/1606.08415>

GPT-2论文介绍了一系列不同规模的基于Transformer的LLM，参数量从124M、355M、774M到1.5B：

-   **Language Models are Unsupervised Multitask Learners (2019)** ，作者：Radford等人， [https://d4mucfpksywv.cloudfront.net/better-language￾models/language_models_are_unsupervised_multitask_learners.pdf]()

OpenAI的GPT-3在架构上与GPT-2基本相同，但最大的版本有1750亿参数，比GPT-2最大的模型大100倍，并且训练数据量更多。感兴趣的读者可以参考OpenAI的GPT-3官方论文以及Lambda Labs的技术概述，其计算出在单个RTX 8000消费级GPU上训练GPT-3需要665年：

-   **Language Models are Few-Shot Learners (2023)** ，作者：Brown等人， <https://arxiv.org/abs/2005.14165>
-   **OpenAI's GPT-3 Language Model: A Technical Overview**， <https://lambdalabs.com/blog/demystifying-gpt-3>

NanoGPT是一个代码仓库，包含了GPT-2模型的简洁且高效的实现，与本书中实现的模型类似。尽管本书中的代码与nanoGPT有所不同，但该仓库启发了将大型GPT Python主类实现重新组织成更小的子模块：

-   **NanoGPT, a repository for training medium-sized GPTs**， <https://github.com/karpathy/nanoGPT>

一篇有趣的博客文章显示，当上下文大小小于32,000个标记时，LLM的大部分计算开销花费在前馈层，而不是注意力层：

-   **In the long (context) run**，作者：Harm de Vries， <https://www.harmdevries.com/post/context-length/>

## B.5 第5章

作者的一段视频讲座，详细讲解了损失函数及其通过对数变换以便于数学优化的内容：

-   **L8.2 Logistic Regression Loss Function**， <https://www.youtube.com/watch?v=GxJe0DZvydM>

以下两篇论文详细介绍了预训练LLM所用的数据集、超参数及架构细节：

-   **Pythia: A Suite for Analyzing Large Language Models Across Training and Scaling (2023)** ，作者：Biderman等人， <https://arxiv.org/abs/2304.01373>
-   **OLMo: Accelerating the Science of Language Models (2024)** ，作者：Groeneveld等人， <https://arxiv.org/abs/2402.00838>

本书的配套代码包含了用于准备Project Gutenberg的60,000本公共领域书籍以供LLM训练的说明：

-   **Pretraining GPT on the Project Gutenberg Dataset**， <https://github.com/rasbt/LLMs-from-scratch/tree/main/ch05/03_bonus_pretraining_on_gutenberg>

第5章讨论了LLM的预训练，附录D涵盖了更高级的训练功能，如线性预热和余弦退火。以下论文发现类似的技术可成功应用于已预训练的LLM继续预训练，并提供了额外的提示和见解：

-   **Simple and Scalable Strategies to Continually Pre-train Large Language Models (2024)** ，作者：Ibrahim等人， <https://arxiv.org/abs/2403.08763>

BloombergGPT是一个在金融领域专门设计的大型语言模型（LLM），通过在通用和特定领域文本语料库上进行训练：

-   **BloombergGPT: A Large Language Model for Finance (2023)** ，作者：Wu等人， <https://arxiv.org/abs/2303.17564>

GaLore是一个近期的研究项目，旨在使LLM预训练更加高效。代码更改仅需在训练函数中用`galore-torch` Python包中的GaLoreAdamW优化器替换PyTorch的AdamW优化器：

-   **GaLore: Memory-Efficient LLM Training by Gradient Low-Rank Projection (2024)** ，作者：Zhao等人， <https://arxiv.org/abs/2403.03507>
-   **GaLore code repository**， <https://github.com/jiaweizzhao/GaLore>

以下论文和资源提供了公开的大规模预训练数据集，包含数百GB到TB的文本数据：

-   **Dolma: an Open Corpus of Three Trillion Tokens for LLM Pretraining Research**，作者：Soldaini等人，2024，<https://arxiv.org/abs/2402.00159>
-   **The Pile: An 800GB Dataset of Diverse Text for Language Modeling**，作者：Gao等人，2020，<https://arxiv.org/abs/2101.00027>
-   **The RefinedWeb Dataset for Falcon LLM: Outperforming Curated Corpora with Web Data, and Web Data Only**，作者：Penedo等人，2023，<https://arxiv.org/abs/2306.01116>
-   **RedPajama by Together AI**， <https://github.com/togethercomputer/RedPajama-Data>

首次提出Top-k采样的论文：

-   **Hierarchical Neural Story Generation (2018)** ，作者：Fan等人， <https://arxiv.org/abs/1805.04833>

束搜索（未在第5章中涉及）是一种替代解码算法，通过在每一步仅保留得分最高的部分序列来生成输出序列，兼顾了效率和质量：

-   **Diverse Beam Search: Decoding Diverse Solutions from Neural Sequence Models (2016)** ，作者：Vijayakumar等人， <https://arxiv.org/abs/1610.02424>