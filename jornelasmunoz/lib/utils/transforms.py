import torch

class Noise_dB(object):
    '''
    Given a desired Signal to Noise Ratio (in decibels, dB)
    returns a noisy image. 
        Low SNR = More noisy
        High SNR = Less noisy
    
    Inputs:
        desired_snr: Integer. Signal to noise ration in decibels 
    '''

    def __init__(self, desired_snr=10):
        super().__init__()
        self.snr = desired_snr

    def __call__(self, tensor):
         # Calculate the variance of the image pixels
        signal_power = torch.var(tensor)
    
        # Calculate the noise power
        noise_power = signal_power / (10**(self.snr/10))
    
        # Generate random noise matrix
        noise = torch.normal(0,torch.sqrt(noise_power), size=tensor.shape)
    
        # Add the noise to the image
        noisy_image = tensor + noise
        # noisy_image = torch.clip(noisy_image, 0, 1)

        return noisy_image

    def __repr__(self):
        return self.__class__.__name__ + '(snr = {0})'.format(self.snr)

class Normalize_01(object):
    '''Normalize tensor values between [0,1]

    Args: 
        tensor: image tensor to be normalized. 
    '''
    def __init__(self):
        super().__init__()

    def __call__(self, tensor):
        normalized_tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())
        return normalized_tensor
    def __repr__(self):
        return self.__class__.__name__ + '([0,1])'

class Noise(object):
    def __init__(self, mean=0, dev=1):
        self.mean = mean
        self.dev = dev
    
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size())*self.dev + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + 'mean = {0}, dev= {1}', format(self.mean, self.dev)