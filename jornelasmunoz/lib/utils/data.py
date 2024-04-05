from utils.transforms import Normalize_01, Noise_dB
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from dataset import MNIST_MURA, FashionMURA

def load_data(params):
    '''
    Loads a dataset based on the dictionary of params passed. Also defines list of transformations for model training.
    '''
    # Read in dictionary of params
    dataset = 'MNIST' if 'MNIST'.upper() in params.get('dataset') else 'FashionMNIST'
    snr = params.get('snr')
    print("Using the following parameters:")
    for key, val in params.items():
        print(f"{key}: {val}")
    # Define transforms 
    train_transform_list =[transforms.Grayscale(),
                           # transforms.ToTensor(),
                           transforms.Resize(params['image_size'], antialias=True),
                          ]
    # For noiseless data, just normalize values between [0,1]
    # If noise, add desired SNR noise
    if snr is None: 
        train_transform_list.append(Normalize_01())
        pass
    else:
        assert isinstance(snr, int), "SNR input is not an integer"
        train_transform_list.append(Noise_dB(desired_snr=snr))
        train_transform_list.append(Normalize_01())

    train_transform = transforms.Compose(train_transform_list)
    
    target_transform = transforms.Compose(
                [   transforms.Grayscale(),
                    # transforms.ToTensor(),
                    transforms.Resize((params['image_size'],params['image_size']), antialias=True),
                    Normalize_01(),
                    ])
    # Load Data
    if dataset == 'MNIST':
        train_data = MNIST_MURA('../data/MNIST/', params, transform=train_transform, target_transform=target_transform, train=True)
        test_data  = MNIST_MURA('../data/MNIST/', params, transform=train_transform, target_transform=target_transform, train=False)
        
    elif dataset == 'FashionMNIST':
        train_data = FashionMURA('../data/FashionMNIST/', params, transform=train_transform, target_transform=target_transform, train=True)
        test_data  = FashionMURA('../data/FashionMNIST/', params, transform=train_transform, target_transform=target_transform, train=False)
    
    loaders = {}
    # Define DataLoader
    loaders['train'] = DataLoader(train_data, batch_size=params['batch_size'], shuffle=True)
    loaders['test']  = DataLoader(test_data,  batch_size=params['batch_size'], shuffle=False)
        
    return train_data, test_data, loaders