from torchvision import transforms, models

def config1():
    arch = models.resnet152
    seed = 0
    data_transforms = {
        'train': transforms.Compose([
            transforms.ColorJitter(contrast = 0.2),
            transforms.RandomAffine(degrees = 90),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((348, 348)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    lr = 0.002
    num_cycles=5
    num_epochs_per_cycle=5
    return seed, data_transforms, arch, lr, num_cycles, num_epochs_per_cycle


def config2():
    arch = models.densenet201
    seed = 0
    data_transforms = {
        'train': transforms.Compose([
            transforms.ColorJitter(contrast = 0.2),
            transforms.RandomAffine(degrees = 90),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((348, 348)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    lr = 0.002
    num_cycles=5
    num_epochs_per_cycle=5
    return seed, data_transforms, arch, lr, num_cycles, num_epochs_per_cycle


def config3():
    arch = models.resnext101_32x8d
    seed = 0
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomAffine(degrees = 90),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((348, 348)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    lr = 0.002
    num_cycles=4
    num_epochs_per_cycle=5
    return seed, data_transforms, arch, lr, num_cycles, num_epochs_per_cycle


def config4():
    arch = models.resnext101_32x8d
    seed = 7411
    data_transforms = {
        'train': transforms.Compose([
            transforms.ColorJitter(contrast = 0.2, brightness = 0.2),
            transforms.RandomAffine(degrees = 30),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((446, 446)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    lr = 0.004
    num_cycles=4
    num_epochs_per_cycle=4
    return seed, data_transforms, arch, lr, num_cycles, num_epochs_per_cycle
