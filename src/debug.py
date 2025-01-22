def print_param_size(model):
    for name, param in model.named_parameters():
        print(name,param.size())

def print_param_data(model):
    # for name, module in model.named_modules():
    #     print(name, module.__class__)
    for name, param in model.named_parameters():
        print(name,param.data)

def check_layer_names(model):
    for name, module in model.named_modules():
        print(f"Layer Name: {name}, Module: {module.__class__.__name__}")
