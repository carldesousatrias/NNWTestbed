import matplotlib.pyplot as plt
from utils import *

# quantization
def quantize_tensor(x, num_bits):
    qmin = 0.
    qmax = 2. ** num_bits - 1.

    min_val, max_val = torch.min(x), torch.max(x)

    scale = (max_val - min_val) / (qmax - qmin)

    initial_zero_point = torch.min(max_val - min_val).round()
    print(min_val, max_val, scale, initial_zero_point)
    zero_point = 0
    if initial_zero_point < qmin:
        zero_point = qmin
    elif initial_zero_point > qmax:
        zero_point = qmax
    else:
        zero_point = initial_zero_point
    zero_point = int(zero_point)
    q_x = zero_point + x / scale
    q_x.clamp_(qmin, qmax).round_()
    q_x = q_x.byte()
    return {'tensor': q_x, 'scale': scale, 'zero_point': zero_point}


def dequantize_tensor(q_x):
    return q_x['scale'] * (q_x['tensor'].float() - q_x['zero_point'])


def fake_quantization(x, num_bits):
    qmax = 2. ** num_bits - 1.
    min_val, max_val = torch.min(x), torch.max(x)
    scale = qmax / (max_val - min_val)
    x_q = (x - min_val) * scale
    x_q.clamp_(0, qmax).round_() #clamp = min(max(x,min_value),max_value)
    x_q.byte()
    x_f_q = x_q.float() / scale + min_val
    return x_f_q


def quantization(net,num_bits):
    with torch.no_grad():
        for name, module in net.named_modules():
            if isinstance(module, torch.nn.Conv2d)or isinstance(module, torch.nn.Linear):
                tensor = module.weight
                tensor_q = fake_quantization(tensor, num_bits)
                module.weight = nn.Parameter(tensor_q)
    return net


'''

    "quantization"
    param={"bits":.99}

'''

