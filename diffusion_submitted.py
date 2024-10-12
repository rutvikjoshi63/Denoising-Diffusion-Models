import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

import wandb

def extract(a, t, x_shape):
    """
    Extracts the tensor at the given time step.
    Args:
        a: A tensor contains the values of all time steps.
        t: The time step to extract.
        x_shape: The reference shape.
    Returns:
        The extracted tensor.
    """
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def cosine_schedule(timesteps, s=0.008):
    """
    Defines the cosine schedule for the diffusion process
    Args:
        timesteps: The number of timesteps.
        s: The strength of the schedule.
    Returns:
        The computed alpha.
    """
    steps = timesteps + 1
    x = torch.linspace(0, steps, steps)
    alphas_cumprod = torch.cos(((x / steps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    alphas = alphas_cumprod[1:] / alphas_cumprod[:-1]
    return torch.clip(alphas, 0.001, 1)


# normalization functions
def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5


# DDPM implementation
class Diffusion(nn.Module):
    def __init__(
        self,
        model,
        *,
        image_size,
        channels=3,
        timesteps=1000,
    ):
        super().__init__()
        self.channels = channels
        self.image_size = image_size
        self.model = model
        self.num_timesteps = int(timesteps)
        
        """
        Initializes the diffusion process.
            1. Setup the schedule for the diffusion process.
            2. Define the coefficients for the diffusion process.
        Args:
            model: The model to use for the diffusion process.
            image_size: The size of the images.
            channels: The number of channels in the images.
            timesteps: The number of timesteps for the diffusion process.
        """
        ## TODO: Implement the initialization of the diffusion process ##
        # 1. define the scheduler here
        # 2. pre-compute the coefficients for the diffusion process
                # Get alpha at time step t , we r using the cosine scheduler
        self.device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
        self.alphas = cosine_schedule(timesteps=self.num_timesteps).to(self.device)
        self.alpha_bar = torch.cumprod(self.alphas,dim = 0).to(self.device)
        temp_list = []
        temp_list2 = []
        temp_list_rev = []
        temp_list2_rev = []
        for i in range(self.num_timesteps):
            # t_visualize = [int(i * self.model.num_timesteps) for i in percent]
            t= torch.full(size=(32,), fill_value=i, device=self.device)
            # forward
            temp = extract(self.alpha_bar, t, (32, 1, 1, 1)).unsqueeze(0)
            temp2 = extract(self.alphas, t, (32, 1, 1, 1)).unsqueeze(0)
            temp_list.append(temp)
            temp_list2.append(temp2)
        for t_index in reversed(range(self.num_timesteps)):
            # reverse
            t= torch.full(size=(512,), fill_value=t_index, device=self.device)
            
            temp_rev = extract(self.alpha_bar, t, (512, 1, 1, 1)).unsqueeze(0)
            temp2_rev = extract(self.alphas, t, (512, 1, 1, 1)).unsqueeze(0)
            temp_list_rev.append(temp_rev)
            temp_list2_rev.append(temp2_rev)
        
        self.alpha_t_bar = torch.vstack(temp_list)
        self.alpha_bar_t_1 = torch.ones((32, 1, 1, 1), device=self.device) # t_index = 0
        # self.alpha_bar_t_1_ = torch.ones((32, 1, 1, 1), device=self.device) # t_index != 0
        self.alpha_t = torch.vstack(temp_list2)
        self.alpha_t_bar_rev = torch.vstack(temp_list_rev)
        self.alpha_bar_t_1_rev = torch.ones((512, 1, 1, 1), device=self.device) # t_index = 0
        # self.alpha_bar_t_1_ = torch.ones((32, 1, 1, 1), device=self.device) # t_index != 0
        self.alpha_t_rev = torch.vstack(temp_list2_rev)
        # print(f"new self.alpha_t_bar: {self.alpha_t_bar.shape}")
    
        # ###########################################################

    def noise_like(self, shape, device):
        """
        Generates noise with the same shape as the input.
        Args:
            shape: The shape of the noise.
            device: The device on which to create the noise.
        Returns:
            The generated noise.
        """
        noise = lambda: torch.randn(shape, device=device)
        return noise()

    # backward diffusion
    @torch.no_grad()
    def p_sample(self, x, t, t_index):
        """
        Samples from the reverse diffusion process at time t_index.
        Args:
            x: The initial image.
            t: a tensor of the time index to sample at.
            t_index: a scalar of the index of the time step.
        Returns:
            The sampled image.
        """
        ####### TODO: Implement the p_sample function #######
        # print(f"x shape: {x.shape}") # [512, 1, 1, 1]
        # sample x_{t-1} from the gaussian distribution wrt. posterior mean and posterior variance
        # Hint: use extract function to get the coefficients at time t
        # Hint: use self.noise_like function to generate noise. DO NOT USE torch.randn
        # Begin code here
        # 1. Get the alpha at time step t
        # alphas = cosine_schedule(timesteps=self.num_timesteps)
        # alpha_bar_t = self.alpha_t_bar_rev[t[0]] #extract(self.alpha_bar, t, x.shape)
        alpha_bar_t = extract(self.alpha_bar, t, x.shape)
        # print(f"alpha_bar_t shape: {alpha_bar_t.shape}")
        # var = torch.cat((alpha_bar_t, torch.ones_like(alpha_bar_t[0:1], device=self.device)), dim=0)
        if t_index == 0:
            alpha_bar_t_1 =  torch.ones_like(alpha_bar_t, device=self.device)#extract(var, t_index, alpha_bar_t.shape) #torch.full(size=alpha_bar_t.shape, fill_value=1, device=self.device) #extract(self.alpha_bar, t, x.shape) ## make it 1  #torch.full(size=alpha_bar_t.shape,fill_value=1, device=self.device)
        else:
            alpha_bar_t_1 = extract(self.alpha_bar, torch.clamp(t - 1, min=0), x.shape).to(self.device) #TODO check Prevent negative index
        # print(f"alpha_bar_t_1 shape: {alpha_bar_t_1.shape}")
        # alpha_t = self.alpha_t_rev[t[0]].to(self.device) #TODO check
        alpha_t = extract(self.alphas, t, x.shape).to(self.device) #TODO check
        # print(f"alpha_t shape: {alpha_t.shape}")
        # 2. Get the noise from the noise distribution
        noise = self.noise_like(x.shape, self.device)
        # print(f"noise shape: {noise.shape}")
        # 3. Get the predicted noise from the UNET model (model defined in utils.py)
        noise_pred = self.model(x, t).to(self.device) # step 4
        # print(f"noise_pred shape: {noise_pred.shape}")
        # x_0_predicted =
        # 4. Compute the sampled image x_{t-1} from the reverse diffusion process
        x_0_predicted = (x - torch.sqrt(1 - alpha_bar_t) * noise_pred) / torch.sqrt(alpha_bar_t).to(self.device)
        # print(f"x_0_predicted shape: {x_0_predicted.shape}")
        x_0_predicted = torch.clamp(x_0_predicted, -1, 1)
        mean_t_bar = (torch.sqrt(alpha_t)*(1-alpha_bar_t_1)/(1-alpha_bar_t) * x + torch.sqrt(alpha_bar_t_1) *(1-alpha_t)/(1-alpha_bar_t)* x_0_predicted).to(self.device)
        sigma_t = (torch.sqrt((1-alpha_bar_t_1)/(1-alpha_bar_t) * (1-alpha_t))).to(self.device)
        # print(f"mean_t_bar shape: {mean_t_bar.shape} sigma_t shape: {sigma_t.shape}")
        
        if t_index == 0:
            return  mean_t_bar + sigma_t * 0
        else:
            return mean_t_bar + sigma_t * noise
        # ####################################################

    @torch.no_grad()
    def p_sample_loop(self, img):
        """
        Samples from the noise distribution at each time step.
        Args:
            img: The initial image that randomly sampled from the noise distribution.
        Returns:
            The sampled image.
        """
        b = img.shape[0]#.to_device(self.device)
        # print(f"b is on device: {b.device}")
        
        #### TODO: Implement the p_sample_loop function ####
        # 1. loop through the time steps from the last to the first
        # Start with a full noise image and reverse through the timesteps
        for t_index in reversed(range(self.num_timesteps)):
        # 2. inside the loop, sample x_{t-1} from the reverse diffusion process
            t = torch.full(size=(b,),fill_value= t_index, device=self.device) # Generate time index
            # print(f"t is on device: {t.device}")
            img = self.p_sample(img, t, t_index).to(self.device)  # Reverse sample to get x_{t-1}
            # print(f"img is on device: {img.device}")

        # 3. clamp and unnormalize the generted image to valid pixel range
        img = torch.clamp(img, -1.0, 1.0)
        img = unnormalize_to_zero_to_one(img)
        # print(f"img is on device: {img.device}")
        
        # Hint: to get time index, you can use torch.full()
        return img
        # ####################################################

    @torch.no_grad()
    def sample(self, batch_size):
        """
        Samples from the noise distribution at each time step.
        Args:
            batch_size: The number of images to sample.
        Returns:
            The sampled images.
        """
        self.model.eval()
        #### TODO: Implement the p_sample_loop function ####
        # Hint: use self.noise_like function to generate noise. DO NOT USE torch.randn
        # self.alpha_bar_t_list = []
        # self.alpha_t_list = []
        # for t_index in range(self.num_timesteps):
            # t = torch.full(size=(batch_size,), fill_value=t_index, device=self.device)
            # alpha_bar_t = extract(self.alpha_bar, t, (batch_size,))
            # self.alpha_bar_t_list.append(alpha_bar_t)
            # alpha_t = extract(self.alphas, t, (batch_size,))#.to(self.device) 
            
        # Start with pure Gaussian noise
        img = self.noise_like((batch_size, self.channels, self.image_size, self.image_size), device=self.device)
        # print(f"img is on device: {img.device}")
        
        # Pass img to p_sample_loop and get generated image
        img = self.p_sample_loop(img).to(self.device)
        # print(f"img is on device: {img.device}")
    
        return img

    # forward diffusion
    def q_sample(self, x_0, t, noise):
        """
        Samples from the noise distribution at time t. Simply apply alpha interpolation between x_0 and noise.
        Args:
            x_0: The initial image.
            t: The time index to sample at.
            noise: The noise tensor to sample from.
        Returns:
            The sampled image.
        """
        ###### TODO: Implement the q_sample function #######
        # alpha_t = alphas[t][:, None, None, None]
        # alpha_t_bar = extract(self.alpha_bar, t, x_0.shape) TODO imp
        alpha_t_bar = self.alpha_t_bar[t[0]]
        # print(f"self.alpha_bar is on device: {self.alpha_bar.shape} alpha_t_bar is on device: {alpha_t_bar.shape} t is on device: {t.shape} x_0.shape{x_0.shape}")
        # self.alpha_bar is on device: torch.Size([50]) alpha_t_bar is on device: torch.Size([32, 1, 1, 1]) t is on device: torch.Size([32])
        # print("alpha_t)",alpha_t)
        x_t = (torch.sqrt(alpha_t_bar) * x_0 + torch.sqrt(1 - alpha_t_bar) * noise.to(self.device))
        # x_t = torch.clamp(x_t, 0, 1)
        # print(f"x_t is on device: {x_t.device}")

        return x_t

    def p_losses(self, x_0, t, noise):
        """
        Computes the loss for the forward diffusion.
        Args:
            x_0: The initial image.
            t: The time index to compute the loss at.
            noise: The noise tensor to use.
        Returns:
            The computed loss.
        """
        ###### TODO: Implement the p_losses function #######
        # define loss function wrt. the model output and the target
        # Hint: you can use pytorch built-in loss functions: F.l1_loss
        # Get the image x_t at time step t
        x_t = (self.q_sample(x_0, t, noise)).to(self.device)
        # print(f"x_t is on device: {x_t.device}")
        # Get the predicted noise from the UNET model (model defined in utils.py)
        noise_pred = self.model(x_t, t).to(self.device) # forward definition of the model
        # print(f"noise_pred is on device: {noise_pred.device}")

        loss = F.l1_loss(noise,noise_pred)
        # print(f"loss is on device: {loss.device}")

        return loss
        # ####################################################

    def forward(self, x_0, noise):
        """
        Computes the loss for the forward diffusion.
        Args:
            x_0: The initial image.
            noise: The noise tensor to use.
        Returns:
            The computed loss.
        """
        b, c, h, w, device, img_size, = *x_0.shape, x_0.device, self.image_size
        assert h == img_size and w == img_size, f'height and width of image must be {img_size}'
        ###### TODO: Implement the forward function #######
        # self.device = x_0.device
        # Edge case when noise is not provided
        if noise is None:
            noise = self.noise_like(x_0.shape, device=x_0.device)
            # print(f"noise is on device: {noise.device}")

        # Sample a random timestep t for each image in the batch
        t = torch.randint(low=0, high=self.num_timesteps, size=(b,), device=x_0.device) # TODO ask TA
        # print(f"t is on device: {t.device}")

        # Compute the loss for the forward diffusion
        return self.p_losses(x_0, t, noise)

