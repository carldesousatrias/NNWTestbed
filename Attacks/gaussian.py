from utils import *

def adding_noise(net, power, module_name):
    '''add gausian noise to the parameter of the network'''
    for name, parameters in net.named_parameters():
        if module_name in name:
            print("noise added")
            calcul = nn.utils.parameters_to_vector(parameters)
            sigma = torch.std(calcul, unbiased=False).item()
            noise = torch.normal(mean=0, std=power*sigma, size=parameters.size())
            parameters.data += noise.to(device)
    return net


def adding_noise_global(net, power):
    '''add gausian noise to the parameter of the network'''
    for name, module in net.named_modules():
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
            parameters=module.weight.data
            calcul = nn.utils.parameters_to_vector(parameters)
            sigma = torch.std(calcul, unbiased=False).item()
            noise = torch.normal(mean=0, std=power*sigma, size=parameters.size())
            parameters.data += noise.to(device)
    return net


'''
    "noise"
    param={'name':["features.17.w"],"std":5}
    param={'name':"all","std":5}
'''