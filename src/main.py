import torch

import utility
import data
import model
import loss
from option import args
from trainer import Trainer

torch.manual_seed(args.seed)
checkpoint = utility.checkpoint(args)



from collections import OrderedDict
import numpy as np
import pandas as pd
import torch

def summary(model, x, *args, **kwargs):
    """Summarize the given input model.
    Summarized information are 1) output shape, 2) kernel shape,
    3) number of the parameters and 4) operations (Mult-Adds)
    Args:
        model (Module): Model to summarize
        x (Tensor): Input tensor of the model with [N, C, H, W] shape
                    dtype and device have to match to the model
        args, kwargs: Other argument used in `model.forward` function
    """
    def register_hook(module):      #函数内部定义了一个hook函数，它是一个回调函数，用于在模型的前向传播过程中获取模块的输入和输出，并将相关信息保存到summary字典中。
        def hook(module, inputs, outputs):
            cls_name = str(module.__class__).split(".")[-1].split("'")[0]   #h获取当前模块的类名cls_name，然后计算当前模块在summary字典中的索引module_idx。
            module_idx = len(summary)

            # Lookup name in a dict that includes parents
            for name, item in module_names.items():
                if item == module:
                    key = "{}_{}".format(module_idx, name)

            info = OrderedDict()        #函数创建一个有序字典info，用于保存当前模块的相关信息，包括模块的id、输出形状、卷积核大小、内部权重等。
            info["id"] = id(module)
            if isinstance(outputs, (list, tuple)):  #判断outputs是否是list或tuple的实例
                try:
                    info["out"] = list(outputs[0].size())
                except AttributeError:
                    # pack_padded_seq and pad_packed_seq store feature into data attribute
                    info["out"] = list(outputs[0].data.size())
            else:
                info["out"] = list(outputs.size())

            info["ksize"] = "-"
            info["inner"] = OrderedDict()
            info["params_nt"], info["params"], info["macs"] = 0, 0, 0
            for name, param in module.named_parameters():
                info["params"] += param.nelement() * param.requires_grad
                info["params_nt"] += param.nelement() * (not param.requires_grad)

                if name == "weight":
                    ksize = list(param.size())
                    # to make [in_shape, out_shape, ksize, ksize]
                    if len(ksize) > 1:
                        ksize[0], ksize[1] = ksize[1], ksize[0]
                    info["ksize"] = ksize

                    # ignore N, C when calculate Mult-Adds in ConvNd
                    if "Conv" in cls_name:
                        info["macs"] += int(param.nelement() * np.prod(info["out"][2:]))
                    else:
                        info["macs"] += param.nelement()

                # RNN modules have inner weights such as weight_ih_l0
                elif "weight" in name:
                    info["inner"][name] = list(param.size())
                    info["macs"] += param.nelement()

            # if the current module is already-used, mark as "(recursive)" 
            # check if this module has params
            #对于RNN模块，它们通常具有内部权重（如weight_ih_l0），会将这些内部权重的大小保存到info["inner"]中，并将其乘法累加次数添加到info["macs"]中。
            if list(module.named_parameters()):
                for v in summary.values():
                    if info["id"] == v["id"]:
                        info["params"] = "(recursive)"

            if info["params"] == 0:
                info["params"], info["macs"] = "-", "-"

            summary[key] = info

        # ignore Sequential and ModuleList
        if not module._modules:
            hooks.append(module.register_forward_hook(hook))

    module_names = get_names_dict(model)

    hooks = []
    summary = OrderedDict()

    #这段代码的整体作用是在模型的前向传播过程中，应用register_hook函数对每个模块进行注册，以获取模块的相关信息。
    # 在前向传播完成后，将之前注册的hook函数移除，以避免对后续操作产生影响。
    model.apply(register_hook)  
    try:
        with torch.no_grad():
            model(x) if not (kwargs or args) else model(x, *args, **kwargs)
    finally:
        for hook in hooks:
            hook.remove()

    # Use pandas to align the columns
    df = pd.DataFrame(summary).T

    df["Mult-Adds"] = pd.to_numeric(df["macs"], errors="coerce")
    df["Params"] = pd.to_numeric(df["params"], errors="coerce")
    df["Non-trainable params"] = pd.to_numeric(df["params_nt"], errors="coerce")
    df = df.rename(columns=dict(
        ksize="Kernel Shape",
        out="Output Shape",
    ))
    df_sum = df.sum()
    df.index.name = "Layer"

    df = df[["Kernel Shape", "Output Shape", "Params", "Mult-Adds"]]
    max_repr_width = max([len(row) for row in df.to_string().split("\n")])

    # option = pd.option_context(
    #     "display.max_rows", 600,
    #     "display.max_columns", 10,
    #     "display.float_format", pd.io.formats.format.EngFormatter(use_eng_prefix=True)
    # )
    # with option:
    #     # print("="*max_repr_width)
    #     # print(df.replace(np.nan, "-"))
    #     # print("-"*max_repr_width)
    #     # df_total = pd.DataFrame(
    #     #     {"Total params": (df_sum["Params"] + df_sum["params_nt"]),
    #     #     "Trainable params": df_sum["Params"],
    #     #     "Non-trainable params": df_sum["params_nt"],
    #     #     "Mult-Adds": df_sum["Mult-Adds"]
    #     #     },
    #     #     index=['Totals']
    #     # ).T
    #     # print(df_total)
    #     # print("="*max_repr_width)

    return df, df_sum

def get_names_dict(model):
    """Recursive walk to get names including path."""
    names = {}

    def _get_names(module, parent_name=""):
        for key, m in module.named_children():
            cls_name = str(m.__class__).split(".")[-1].split("'")[0]
            num_named_children = len(list(m.named_children()))
            if num_named_children > 0:
                name = parent_name + "." + key if parent_name else key
            else:
                name = parent_name + "." + cls_name + "_"+ key if parent_name else key
            names[name] = m

            if isinstance(m, torch.nn.Module):
                _get_names(m, parent_name=name)

    _get_names(model)
    return names


def mysummary(_model, checkpoint):
    _model.model.eval()
    df, df_sum = summary(_model.model, torch.zeros((1, 3, 720 // args.scale[0], 1280 // args.scale[0])).cuda())

    max_repr_width = max([len(row) for row in df.to_string().split("\n")])

    option = pd.option_context(
        "display.max_rows", None,
        "display.max_columns", None,
        "display.float_format", pd.io.formats.format.EngFormatter(use_eng_prefix=True)
    )
    with option:
        checkpoint.write_log("="*max_repr_width)        #代码使用pd.option_context()设置pandas的显示选项，包括最大行数、最大列数和浮点数的显示格式。
        checkpoint.write_log(str(df.replace(np.nan, "-")))
        checkpoint.write_log("-"*max_repr_width)
        df_total = pd.DataFrame(
            {"Total params": (df_sum["Params"] + df_sum["params_nt"]),
            "Trainable params": df_sum["Params"],
            "Non-trainable params": df_sum["params_nt"],
            "Mult-Adds": df_sum["Mult-Adds"]
            },
            index=['Totals']
        ).T
        checkpoint.write_log(str(df_total))
        checkpoint.write_log("="*max_repr_width)


def main():
    global model
    if args.data_test == ['video']:
        from videotester import VideoTester
        model = model.Model(args, checkpoint)
        t = VideoTester(args, model, checkpoint)
        t.test()
    else:
        if checkpoint.ok:
            loader = data.Data(args)
            _model = model.Model(args, checkpoint)
            mysummary(_model, checkpoint)
            _loss = loss.Loss(args, checkpoint) if not args.test_only else None
            t = Trainer(args, loader, _model, _loss, checkpoint)
            while not t.terminate():
                t.train()
                t.test()

            checkpoint.done()

if __name__ == '__main__':
    main()
