from Attacks.pruning import prune_model_l1_unstructured, prune_model_random_unstructured
from Attacks.quantization import quantization
from Attacks.gaussian import adding_noise_global, adding_noise
from Attacks.fine_tuning import finetuning
from Attacks.knowledge_distillation import knowledge_distillation
from Attacks.watermark_overwriting import overwriting



def attacks(net,TypeAttack,attackparameters):
    '''
    Apply a modification based on the ID and parameters
    :param TypeAttack: ID of the modification
    :param net: network to be altered
    :param parameters: parameters of the modification
    :return: altered NN
    '''
    if TypeAttack=="noise":
        if attackparameters["name"]=="all":
            return adding_noise_global(net,attackparameters["std"])
        for module in attackparameters["name"]:
            net=adding_noise(net,attackparameters["std"],module)
        return net
    elif TypeAttack=="l1pruning":
        return prune_model_l1_unstructured(net, attackparameters["proportion"])
    elif TypeAttack=="rdnpruning":
        return prune_model_random_unstructured(net,attackparameters["proportion"])
    elif TypeAttack=="quantization":
        return quantization(net,attackparameters["bits"])
    elif TypeAttack=="ft":
        return finetuning(net,attackparameters["epochs"],attackparameters["trainloader"])
    elif TypeAttack=="kd":
        return knowledge_distillation(net,attackparameters["epochs"],attackparameters["trainloader"],attackparameters["student"])
    elif TypeAttack=="wo":
        return overwriting(net, attackparameters["NNWmethods"], attackparameters["nbr"], attackparameters["watermarking_dict"])
    else:
        print("NotImplemented")
        return net
'''
    attackParameter = {'name': "all", "std":.1}
    network = attacks(network, "noise", attackParameter)
'''