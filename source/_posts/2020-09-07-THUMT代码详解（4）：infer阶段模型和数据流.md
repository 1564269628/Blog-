---
title: THUMT代码详解（4）：infer阶段模型和数据流
urlname: thumt-code-summary-4
toc: true
date: 2020-09-07 20:48:59
updated: 2020-09-07 20:48:59
tags: THUMT
categories: NLP
---

[简介篇地址](/post/thumt-code-summary-1)

<!--more-->

本篇将分为两个部分：一部分讲evaluation部分的模型和数据流，另一部分讲inference部分的模型和数据流。两者本质上是差不多的。

## evaluation

evaluation的入口处也在[trainer.py](https://github.com/THUNLP-MT/THUMT/blob/pytorch/thumt/bin/trainer.py)：

```py
if step % params.eval_steps == 0:
    utils.evaluate(model, sorted_key, eval_dataset,
                    params.output, references, params)
```

这里的`sorted_key`和`eval_dataset`和`references`就是[THUMT代码详解（2）：数据处理](/post/thumt-code-summary-2)中提到的：

* `sorted_key`：用于把排序过的数据复原
* `eval_dataset`：一个dataset，由`source`和`source_mask`组成
* `references`：分词后的target端的句子，用于计算bleu值

然后就是[evaluation.py](https://github.com/THUNLP-MT/THUMT/blob/pytorch/thumt/utils/evaluation.py)：

```py
def evaluate(model, sorted_key, dataset, base_dir, references, params):
    if not references:
        return

    # 各种文件的地址，目前没什么用
    base_dir = base_dir.rstrip("/")
    save_path = os.path.join(base_dir, "eval")
    record_name = os.path.join(save_path, "record")
    log_name = os.path.join(save_path, "log")
    max_to_keep = params.keep_top_checkpoint_max

    # 创建目录和文件，还是没什么用
    if dist.get_rank() == 0:
        # Create directory and copy files
        if not os.path.exists(save_path):
            print("Making dir: %s" % save_path)
            os.makedirs(save_path)

            params_pattern = os.path.join(base_dir, "*.json")
            params_files = glob.glob(params_pattern)

            for name in params_files:
                new_name = name.replace(base_dir, save_path)
                shutil.copy(name, new_name)

    # Do validation here
    global_step = get_global_step()

    if dist.get_rank() == 0:
        print("Validating model at step %d" % global_step)

    # 调用内部函数_evaluate_model，进行实际的evaluate工作
    score = _evaluate_model(model, sorted_key, dataset, references, params)
    .....
```

调用`_evaluate_model`：

```py
def _evaluate_model(model, sorted_key, dataset, references, params):
    # Create model
    # 在验证阶段不进行back-propagation
    with torch.no_grad():
        # 调整模型的模式
        model.eval()

        # 得到dataset的iterator
        iterator = iter(dataset)
        counter = 0
        pad_max = 1024

        # Buffers for synchronization
        # TODO
        size = torch.zeros([dist.get_world_size()]).long()
        t_list = [torch.empty([params.decode_batch_size, pad_max]).long()
                  for _ in range(dist.get_world_size())]
        results = []

        while True:
            try:
                features = next(iterator)
                # 这部分中继续对features进行处理，这个函数之前也见过了，就是把string转换成id
                # 所以就不细讲了
                features = lookup(features, "infer", params)
                batch_size = features["source"].shape[0]
            except:
                features = {
                    "source": torch.ones([1, 1]).long(),
                    "source_mask": torch.ones([1, 1]).float()
                }
                batch_size = 0

            t = time.time()
            counter += 1

            # Decode
            # 调用beam_search进行实际解码工作
            seqs, _ = beam_search([model], features, params)
            ......
```

`beam_search`位于[inference.py](https://github.com/THUNLP-MT/THUMT/blob/pytorch/thumt/utils/inference.py)：

```py
def beam_search(models, features, params):
    if not isinstance(models, (list, tuple)):
        raise ValueError("'models' must be a list or tuple")

    beam_size = params.beam_size
    top_beams = params.top_beams
    alpha = params.decode_alpha
    decode_ratio = params.decode_ratio
    decode_length = params.decode_length

    pad_id = params.lookup["target"][params.pad.encode("utf-8")]
    bos_id = params.lookup["target"][params.bos.encode("utf-8")]
    eos_id = params.lookup["target"][params.eos.encode("utf-8")]

    min_val = -1e9
    shape = features["source"].shape
    device = features["source"].device
    batch_size = shape[0]
    seq_length = shape[1]

    # Compute initial state if necessary
    states = []
    funcs = []

    # 对每个model，都计算出encode之后的state：
    # {
    #     "encoder_output": [batch, length_s, hidden],
    #     "enc_attn_bias": [batch, 1, 1, length_s],
    #     "decoder": ...
    # }
    # encode函数之前已经讲了很多了，这里和训练阶段没有区别，所以就不讲了==
    for model in models:
        state = model.empty_state(batch_size, device)
        states.append(model.encode(features, state))
        funcs.append(model.decode)

    # For source sequence length
    # 计算每个句子译码时的最长长度
    max_length = features["source_mask"].sum(1) * decode_ratio
    max_length = max_length.long() + decode_length
    max_step = max_length.max()
    # [batch, beam_size]
    # 把长度扩展beam_size倍
    max_length = torch.unsqueeze(max_length, 1).repeat([1, beam_size])

    # Expand the inputs
    # [batch, length] => [batch * beam_size, length]
    # [batch, 1, length]
    features["source"] = torch.unsqueeze(features["source"], 1)
    # [batch, beam_size, length]
    features["source"] = features["source"].repeat([1, beam_size, 1])
    # [batch * beam_size, length]
    features["source"] = torch.reshape(features["source"],
                                       [batch_size * beam_size, seq_length])
    features["source_mask"] = torch.unsqueeze(features["source_mask"], 1)
    features["source_mask"] = features["source_mask"].repeat([1, beam_size, 1])
    features["source_mask"] = torch.reshape(features["source_mask"],
                                       [batch_size * beam_size, seq_length])
    ......
```

这里使用了大量的[torch.repeat](https://pytorch.org/docs/stable/tensors.html#torch.Tensor.repeat)函数，其主要目的代码里已经写得很清楚了，就是把每个句子扩展`beam_size`倍，用于在beam search中使用。唯一值得注意的是，这里是整体重复而不是interleave的。

```py
def beam_search(models, features, params):
    ......
    # 把decode函数和features传过去了
    # 等到用到的时候再说
    decoding_fn = _get_inference_fn(funcs, features)

    # 把每个states里的每个tensor都重复beam遍（增加第二维）
    # 主要作用在encoder_output和enc_attn_bias上
    # 因为感觉和主体内容没什么关系所以不讲了
    # 感兴趣的话请自行阅读nest.py
    # states[0]["encoder_output"]: [batch, beam, length_s, hidden]
    # states[0]["enc_attn_bias"]: [batch, beam, 1, 1, length_s]
    states = map_structure(
        lambda x: _tile_to_beam_size(x, beam_size),
        states)

    # Initial beam search state
    # 创建一个维度是[batch_size, beam_size, 1]的矩阵，填充上<bos>
    # 用途是 TODO
    init_seqs = torch.full([batch_size, beam_size, 1], bos_id, device=device)
    init_seqs = init_seqs.long()
    # 创建一个维度是[batch_size, beam_size]的矩阵，第一列是0，其他列是无穷小
    # 用途是 TODO
    init_log_probs = init_seqs.new_tensor(
        [[0.] + [min_val] * (beam_size - 1)], dtype=torch.float32)
    init_log_probs = init_log_probs.repeat([batch_size, 1])
    # 创建一个维度是[batch_size, beam_size]的矩阵，全部为0
    # 用途是 TODO
    # score和log_probs的区别似乎是，score有一个length penalty
    init_scores = torch.zeros_like(init_log_probs)
    # 创建结束状态矩阵
    # 为每个句子维护beam_size个分最高的已完成的句子
    # 用途是 TODO （我猜，inputs是未完成的句子，finish是已完成的句子）
    fin_seqs = torch.zeros([batch_size, beam_size, 1], dtype=torch.int64,
                           device=device)
    fin_scores = torch.full([batch_size, beam_size], min_val,
                            dtype=torch.float32, device=device)
    fin_flags = torch.zeros([batch_size, beam_size], dtype=torch.bool,
                            device=device)

    # 创建BeamSearchState，把刚才的状态矩阵都放进去
    # 以及state TODO
    state = BeamSearchState(
        inputs=(init_seqs, init_log_probs, init_scores),
        state=states,
        finish=(fin_flags, fin_seqs, fin_scores),
    )

    # 接下来进入beam search每个step的循环
    # 循环结束的条件是，尚未结束的beam的最高可能得分低于目前已有的最低得分
    for time in range(max_step):
        state = _beam_search_step(time, decoding_fn, state, batch_size,
                                  beam_size, alpha, pad_id, eos_id, max_length)
        max_penalty = ((5.0 + max_step) / 6.0) ** alpha
        best_alive_score = torch.max(state.inputs[1][:, 0] / max_penalty)
        worst_finished_score = torch.min(state.finish[2])
        cond = torch.gt(worst_finished_score, best_alive_score)
        is_finished = bool(cond)

        if is_finished:
            break
    ......
```

然后我们来看`_beam_search_step`这个函数：

```py
# time：当前时间步
# func：用于解码的函数，实际上就是Transformer.decode
# state：包括inputs、state和finish三个属性
# ......
def _beam_search_step(time, func, state, batch_size, beam_size, alpha,
                      pad_id, eos_id, max_length, inf=-1e9):
    # Compute log probabilities
    # seqs: [batch, beam, time + 1]
    # log_probs: [batch, beam]
    seqs, log_probs = state.inputs[:2]
    # 合并前两维
    # flat_seqs: [batch * beam, time + 1]
    flat_seqs = _merge_first_two_dims(seqs)
    # flat_state[0]["encoder_output"]: [batch * beam, length_s, hidden]
    # flat_state[0]["enc_attn_bias"]: [batch * beam, 1, 1, length_s]
    # flat_state[0]["decoder"]["layer_0"]["k"]: [batch * beam, time, hidden]
    flat_state = map_structure(lambda x: _merge_first_two_dims(x), state.state)
    step_log_probs, next_state = func(flat_seqs, flat_state)
    ......
```

这时再把`_get_inference_fn`拿出来看看：

```py
def _get_inference_fn(model_fns, features):
    def inference_fn(inputs, state):
        # 这里的features是beam_search里已经按beam重复过的source和source_mask
        # target显然是beam search中已经decode出的sequence
        # target_mask全1（因为这个mask只在Transformer.forward中计算loss时有用）
        # target, target_mask: [batch * beam, time + 1]
        local_features = {
            "source": features["source"],
            "source_mask": features["source_mask"],
            "target": inputs,
            "target_mask": torch.ones(*inputs.shape).float().cuda()
        }

        outputs = []
        next_state = []

        for (model_fn, model_state) in zip(model_fns, state):
            # 把state输入给Transformer.decode，得到logits和新的state
            # state["encoder_output"]: [batch * beam, length_s, hidden]
            # state["enc_attn_bias"]: [batch * beam, 1, 1, length_s]
            # state["decoder"]["layer_0"]["k"]: [batch * beam, time, hidden]
            if model_state:
                # logits: [batch * beam, tvoc_size]
                # new_state["encoder_output"]: [batch * beam, length_s, hidden]（不变）
                # new_state["enc_attn_bias"]: [batch * beam, 1, 1, length_s]（不变）
                # new_state["decoder"]["layer_0"]["k"]: [batch * beam, time + 1, hidden]
                logits, new_state = model_fn(local_features, model_state)
                ......
```

这里调用的`Transformer.decode`和训练阶段的是有一些区别的。简单来说，就是attention的key只留下了当前的一个token，而query和value用的是整个句子。这是因为infer阶段只需要在一个位置上进行预测。因为这部分太多了（如果细讲，还是需要把decoder再过一遍），所以这部分放到附录中。

TODO （描述可能不够清晰）

```py
def _get_inference_fn(model_fns, features):
    def inference_fn(inputs, state):
                ......
                # 对最后一维做softmax，然后做log，得到log_prob
                # outputs: [batch * beam, tvoc_size]
                outputs.append(torch.nn.functional.log_softmax(logits,
                                                               dim=-1))
                next_state.append(new_state)
            else:
                logits = model_fn(local_features)
                outputs.append(torch.nn.functional.log_softmax(logits,
                                                               dim=-1))
                next_state.append({})

        # Ensemble
        # 对所有模型输出的log_prob取个平均值
        # log_prob: [batch * beam, tvoc_size]
        log_prob = sum(outputs) / float(len(outputs))

        return log_prob.float(), next_state

    return inference_fn
```

在decode完之后，回到`_beam_search_step`，进行下一步的处理：

```py
def _beam_search_step(time, func, state, batch_size, beam_size, alpha,
                      pad_id, eos_id, max_length, inf=-1e9):
    ......
    # step_log_probs: [batch, beam, tvoc_size]
    step_log_probs = _split_first_two_dims(step_log_probs, batch_size,
                                           beam_size)
    # 把state中的tensor的前两维展开（虽然下一次计算之前又会折叠回去）
    # next_state[0]["encoder_output"]: [batch, beam, length_s, hidden]
    # next_state[0]["enc_attn_bias"]: [batch, beam, 1, 1, length_s]
    # next_state[0]["decoder"]["layer_0"]["k"]: [batch, beam, time + 1, hidden]
    next_state = map_structure(
        lambda x: _split_first_two_dims(x, batch_size, beam_size), next_state)
    # 加法broadcast，维度[batch, beam, 1] + 维度[batch, beam, tvoc_size]
    # 结果维度为[batch, beam, tvoc_size]，相当于每个batch的每个beam在每个词上继续延伸都有一个概率
    curr_log_probs = torch.unsqueeze(log_probs, 2) + step_log_probs

    # Apply length penalty
    # 当前decode出的sequence长度是time+1，batch内的所有sequence长度都是一样的
    # （除了那些已经结束的，但是反正不用管）
    # TODO
    # 对log prob施加长度惩罚，得到scores
    length_penalty = ((5.0 + float(time + 1)) / 6.0) ** alpha
    # curr_scores: [batch, beam, tvoc_size]
    curr_scores = curr_log_probs / length_penalty
    # vocab_size = tvoc_size
    vocab_size = curr_scores.shape[-1]

    # Select top-k candidates
    # 从每个句子的所有beam的每个可能的下一个词（共beam*vocab个）中找出最好的2*beam个
    # 作为下一轮beam的候选
    # TODO：为什么是2*beam个？
    # [batch_size, beam_size * vocab_size]
    curr_scores = torch.reshape(curr_scores, [-1, beam_size * vocab_size])
    # [batch_size, 2 * beam_size]
    top_scores, top_indices = torch.topk(curr_scores, k=2*beam_size)
    # Shape: [batch_size, 2 * beam_size]
    # 新的beam是从哪一个beam延伸出来的
    beam_indices = top_indices // vocab_size
    # 新的beam对应的具体是哪个词
    symbol_indices = top_indices % vocab_size
    # Expand sequences
    # [batch_size, 2 * beam_size, time + 1]
    candidate_seqs = _gather_2d(seqs, beam_indices)
    ......
```

这里的`_gather_2d`函数的功能可能不太好理解。[torch.gather](https://pytorch.org/docs/stable/generated/torch.gather.html)这个函数的本意是从tensor中挑出一些组合在一起；在这里就是根据`beam_indices`挑出一些beam然后组合在一起。

```py
# params: [batch, beam, time + 1]
# indices: [batch, 2 * beam]
def _gather_2d(params, indices, name=None):
    batch_size = params.shape[0]
    # range_size = 2 * beam
    range_size = indices.shape[1]
    # 返回一个长度为2 * batch * beam的tensor，内容是[0, 1, ......, 2 * batch * beam - 1]
    batch_pos = torch.arange(batch_size * range_size, device=params.device)
    # batch_pos = [0, 0, ... 0, 1, 1, ..., 1, ..., batch - 1, batch - 1, ..., batch - 1]
    # 每种数字出现的次数是2 * beam次
    batch_pos = batch_pos // range_size
    # batch_pos: [batch, 2 * beam]
    # batch_pos = [[0, 0, ..., 0], [1, 1, ..., 1], ..., [batch - 1, batch - 1, ..., batch - 1]]
    batch_pos = torch.reshape(batch_pos, [batch_size, range_size])
    # TODO：这里实在想象不出来了。。。
    output = params[batch_pos, indices]

    return output
```

再次回到`_beam_search_step`：

```py
def _beam_search_step(time, func, state, batch_size, beam_size, alpha,
                      pad_id, eos_id, max_length, inf=-1e9):
    ......
    # 接下来把词连上去
    # candidate_seqs: [batch, 2 * beam, time + 2]
    candidate_seqs = torch.cat([candidate_seqs,
                                torch.unsqueeze(symbol_indices, 2)], 2)

    # Expand sequences
    # Suppress finished sequences
    # 找出那些产生了<eos>的句子
    flags = torch.eq(symbol_indices, eos_id).to(torch.bool)
    # 将已经结束的句子的分数置为无穷小
    # [batch, 2 * beam_size]
    alive_scores = top_scores + flags.to(torch.float32) * inf
    # 在每个句子中挑出beam个分数最高的句子
    # [batch, beam_size]
    alive_scores, alive_indices = torch.topk(alive_scores, beam_size)
    # 找出对应的词
    # TODO
    # alive_symbols: [batch, beam]
    alive_symbols = _gather_2d(symbol_indices, alive_indices)
    # 找出对应的beam位置
    # TODO
    # alive_indices: [batch, beam]
    alive_indices = _gather_2d(beam_indices, alive_indices)
    # 找出对应的sequence
    # TODO
    # alive_seqs: [batch, beam, time + 1]
    alive_seqs = _gather_2d(seqs, alive_indices)
    # 把新的词连接上去
    # alive_seqs: [batch_size, beam_size, time + 2]
    alive_seqs = torch.cat([alive_seqs, torch.unsqueeze(alive_symbols, 2)], 2)
    # alive_state[0]["encoder_output"]: [batch, beam, length_s, hidden]
    # alive_state[0]["enc_attn_bias"]: [batch, beam, 1, 1, length_s]
    # alive_state[0]["decoder"]["layer_0"]["k"]: [batch, beam, time + 1, hidden]
    alive_state = map_structure(
        lambda x: _gather_2d(x, alive_indices),
        next_state)
    alive_log_probs = alive_scores * length_penalty
    # Check length constraint
    # 如果句子长度超过限制，则分数设为无穷小
    length_flags = torch.le(max_length, time + 1).float()
    alive_log_probs = alive_log_probs + length_flags * inf
    alive_scores = alive_scores + length_flags * inf

    # Select finished sequences
    prev_fin_flags, prev_fin_seqs, prev_fin_scores = state.finish
    # [batch, 2 * beam_size]
    step_fin_scores = top_scores + (1.0 - flags.to(torch.float32)) * inf
    # [batch, 3 * beam_size]
    fin_flags = torch.cat([prev_fin_flags, flags], dim=1)
    fin_scores = torch.cat([prev_fin_scores, step_fin_scores], dim=1)
    # [batch, beam_size]
    fin_scores, fin_indices = torch.topk(fin_scores, beam_size)
    fin_flags = _gather_2d(fin_flags, fin_indices)
    pad_seqs = prev_fin_seqs.new_full([batch_size, beam_size, 1], pad_id)
    prev_fin_seqs = torch.cat([prev_fin_seqs, pad_seqs], dim=2)
    fin_seqs = torch.cat([prev_fin_seqs, candidate_seqs], dim=1)
    fin_seqs = _gather_2d(fin_seqs, fin_indices)

    new_state = BeamSearchState(
        inputs=(alive_seqs, alive_log_probs, alive_scores),
        state=alive_state,
        finish=(fin_flags, fin_seqs, fin_scores),
    )

    return new_state
```

```py
def beam_search(models, features, params):
    ......
    final_state = state
    alive_seqs = final_state.inputs[0]
    alive_scores = final_state.inputs[2]
    final_flags = final_state.finish[0].byte()
    final_seqs = final_state.finish[1]
    final_scores = final_state.finish[2]

    final_seqs = torch.where(final_flags[:, :, None], final_seqs, alive_seqs)
    final_scores = torch.where(final_flags, final_scores, alive_scores)

    # Append extra <eos>
    final_seqs = torch.nn.functional.pad(final_seqs, (0, 1, 0, 0, 0, 0),
                                         value=eos_id)

    return final_seqs[:, :top_beams, 1:], final_scores[:, :top_beams]
```

然后再回到`_evaluate_model`：

```py
def _evaluate_model(model, sorted_key, dataset, references, params):
            ......
            # Padding
            seqs = torch.squeeze(seqs, dim=1)
            pad_batch = params.decode_batch_size - seqs.shape[0]
            pad_length = pad_max - seqs.shape[1]
            seqs = torch.nn.functional.pad(seqs, (0, pad_length, 0, pad_batch))

            # Synchronization
            size.zero_()
            size[dist.get_rank()].copy_(torch.tensor(batch_size))
            dist.all_reduce(size)
            dist.all_gather(t_list, seqs)

            if size.sum() == 0:
                break

            if dist.get_rank() != 0:
                continue

            for i in range(params.decode_batch_size):
                for j in range(dist.get_world_size()):
                    n = size[j]
                    seq = _convert_to_string(t_list[j][i], params)

                    if i >= n:
                        continue

                    # Restore BPE segmentation
                    seq = BPE.decode(seq)

                    results.append(seq.split())

            t = time.time() - t
            print("Finished batch: %d (%.3f sec)" % (counter, t))

    model.train()

    if dist.get_rank() == 0:
        restored_results = []

        for idx in range(len(results)):
            restored_results.append(results[sorted_key[idx]])

        return bleu(restored_results, references)

    return 0.0
```

最后回到`evaluate`函数：

```py
def evaluate(model, sorted_key, dataset, base_dir, references, params):
    ......
    # 接下来的工作就是保存和替换checkpoint，暂时没什么用
    # Save records
    if dist.get_rank() == 0:
        scalar("BLEU/score", score, global_step, write_every_n_steps=1)
        print("BLEU at step %d: %f" % (global_step, score))

        # Save checkpoint to save_path
        save({"model": model.state_dict(), "step": global_step}, save_path)

        _save_log(log_name, ("BLEU", global_step, score))
        records = _read_score_record(record_name)
        record = [latest_checkpoint(save_path).split("/")[-1], score]

        added, removed, records = _add_to_record(records, record, max_to_keep)

        if added is None:
            # Remove latest checkpoint
            filename = latest_checkpoint(save_path)
            print("Removing %s" % filename)
            files = glob.glob(filename + "*")

            for name in files:
                os.remove(name)

        if removed is not None:
            filename = os.path.join(save_path, removed)
            print("Removing %s" % filename)
            files = glob.glob(filename + "*")

            for name in files:
                os.remove(name)

        _save_score_record(record_name, records)

        best_score = records[0][1]
        print("Best score at step %d: %f" % (global_step, best_score))
```

## 附录：infer阶段的decoder

