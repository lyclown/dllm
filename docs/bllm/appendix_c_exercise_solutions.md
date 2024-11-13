# 附录 C. 练习解答

完整的练习代码示例可以在补充的GitHub仓库中找到，地址为：<https://github.com/rasbt/LLMs-from-scratch>

## C.1 第2章

**练习 2.1**  
可以通过将字符串逐个输入编码器来获取各个token ID：

```python
print(tokenizer.encode("Ak"))
print(tokenizer.encode("w"))
# ...
```

输出结果为：

```csharp
[33901]
[86]
# ...
```

然后可以使用以下代码重新组合原始字符串：

```python
print(tokenizer.decode([33901, 86, 343, 86, 220, 959]))
```

返回结果为：

```arduino
'Akwirw ier'
```

**练习 2.2**  
设置`max_length=2`和`stride=2`的数据加载器代码：

```python
dataloader = create_dataloader(raw_text, batch_size=4, max_length=2, stride=2)
```

它会生成如下格式的批次：

```lua
tensor([[  40,  367],
        [2885, 1464],
        [1807, 3619],
        [ 402,  271]])
```

第二个数据加载器的代码，设置`max_length=8`和`stride=2`：

```python
dataloader = create_dataloader(raw_text, batch_size=4, max_length=8, stride=2)
```

示例批次如下：

```yaml
tensor([[   40,   367,  2885,  1464,  1807,  3619,   402,   271],
        [ 2885,  1464,  1807,  3619,   402,   271, 10899,  2138],
        [ 1807,  3619,   402,   271, 10899,  2138,   257,  7026],
        [  402,   271, 10899,  2138,   257,  7026, 15632,   438]])
```

## C.2 第3章

**练习 3.1**  
正确的权重分配如下：

```python
sa_v1.W_query = torch.nn.Parameter(sa_v2.W_query.weight.T)
sa_v1.W_key = torch.nn.Parameter(sa_v2.W_key.weight.T)
sa_v1.W_value = torch.nn.Parameter(sa_v2.W_value.weight.T)
```

**练习 3.2**  
为了实现与单头注意力类似的输出维度为2，我们需要将投影维度`d_out`设置为1：

```python
d_out = 1
mha = MultiHeadAttentionWrapper(d_in, d_out, block_size, 0.0, num_heads=2)
```

**练习 3.3**  
最小GPT-2模型的初始化如下：

```python
block_size = 1024
d_in, d_out = 768, 768
num_heads = 12
mha = MultiHeadAttention(d_in, d_out, block_size, 0.0, num_heads)
```


## C.3 第4章

**练习 4.1**  
我们可以通过以下代码计算前馈模块和注意力模块中的参数数量：

```python
block = TransformerBlock(GPT_CONFIG_124M)
total_params = sum(p.numel() for p in block.ff.parameters())
print(f"Total number of parameters in feed forward module: {total_params:,}")
total_params = sum(p.numel() for p in block.att.parameters())
print(f"Total number of parameters in attention module: {total_params:,}")
```

如我们所见，前馈模块的参数数量大约是注意力模块的两倍：

```arduino
Total number of parameters in feed forward module: 4,722,432
Total number of parameters in attention module: 2,360,064
```

**练习 4.2**  
要实例化其他GPT模型大小，我们可以修改配置字典，如下所示（此处为GPT-2 XL的配置）：

```python
GPT_CONFIG = GPT_CONFIG_124M.copy()
GPT_CONFIG["emb_dim"] = 1600
GPT_CONFIG["n_layers"] = 48
GPT_CONFIG["n_heads"] = 25
model = GPTModel(GPT_CONFIG)
```

然后，复用4.6节中的代码来计算参数数量和内存需求，得到以下结果：

```yaml
GPT-2 XL:
Total number of parameters: 1,637,792,000
Number of trainable parameters considering weight tying: 1,557,380,800
Total size of the model: 6247.68 MB
```

## C.4 第5章

**练习 5.1**  
我们可以使用`print_sampled_tokens`函数来打印“pizza”这个token（或单词）被采样的次数。以下是5.3.1节中定义的代码：  
在温度为0或0.1时，“pizza”不会被采样；当温度提高到5时，“pizza”被采样32次。估计概率为`32/1000 × 100% = 3.2%`。实际概率为4.3%，包含在重新缩放的softmax概率张量`scaled_probas[2][6]`中。

**练习 5.2**  
Top-k采样和温度缩放是基于LLM和期望的输出多样性和随机性来调整的设置。

-   使用相对较小的top-k值（例如小于10）并将温度设置在1以下时，模型输出变得不那么随机且更具确定性。这适用于需要生成更可预测、连贯的文本的情况，例如正式文件或报告。
-   对于需要较高的准确性应用场景，如技术分析、代码生成、问答和教育内容，较低的k值和温度更有帮助。
-   另一方面，较大的top-k值（例如20到40）和超过1的温度适用于头脑风暴或创意内容生成，如小说创作。

**练习 5.3**  
要在`generate`函数中强制实现确定性行为，有以下几种方法：

1.  将`top_k=None`，且不进行温度缩放；
1.  设置`top_k=1`。

**练习 5.4**  
加载主章节中保存的模型和优化器：

```python
checkpoint = torch.load("model_and_optimizer.pth")
model = GPTModel(GPT_CONFIG_124M)
model.load_state_dict(checkpoint["model_state_dict"])
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.1)
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
```

然后，调用`train_simple_function`并设置`num_epochs=1`以再训练一个epoch。

**练习 5.5**  
我们可以使用以下代码计算GPT模型的训练集和验证集损失：

```python
train_loss = calc_loss_loader(train_loader, gpt, device)
val_loss = calc_loss_loader(val_loader, gpt, device)
```

124M参数模型的结果为：

```yaml
Training loss: 3.754748503367106
Validation loss: 3.559617757797241
```

主要观察结果是训练集和验证集的表现相似。这有几种可能的解释：

1.  如果The Verdict不属于GPT-2的预训练数据集，则模型不会对训练集进行显式过拟合，并且在The Verdict的训练集和验证集部分表现相似。
1.  如果The Verdict属于GPT-2的训练数据集，则无法判断模型是否对训练数据过拟合，因为验证集也被用于训练。要评估过拟合程度，需要使用在GPT-2训练后生成的新数据集。

**练习 5.6**  
在主章节中，我们实验的是最小的GPT-2模型，仅包含124M参数，以尽量降低资源需求。要实验更大的模型，只需进行很小的代码更改。例如，将第5章中的124M模型替换为1558M模型，只需更改以下两行代码：

```python
hparams, params = download_and_load_gpt2(model_size="124M", models_dir="gpt2")
model_name = "gpt2-small (124M)"
```

更新后的代码如下：

```python
hparams, params = download_and_load_gpt2(model_size="1558M", models_dir="gpt2")
model_name = "gpt2-xl (1558M)"
```