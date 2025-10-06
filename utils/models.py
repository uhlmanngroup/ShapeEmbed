import math
import torch
import torch.nn as nn
import torchvision.models as models

# helper nn.Module classes
################################################################################

# helper class for a 'view layer'
############################################
class View(nn.Module):
  def __init__(self, shape):
    super().__init__()
    self.shape = shape
  def forward(self, x): return x.view(self.shape)

# helper class for a 'scale layer'
############################################
class Scale(nn.Module):
  def __init__(self, scalar):
    super().__init__()
    self._scalar = scalar
  def forward(self, x): return torch.mul(x, self._scalar)

# helper class to apply a random roll to a distance matrix
# active when in train mode, innactive otherwise
##########################################################
class RandRoll(nn.Module):
  def __init__(self, seed=42, active=True):
    super().__init__()
    self._seed = seed
    self._gen = torch.Generator()
    self._gen.manual_seed(self._seed)
    self._active = active
  def train(self, mode): self._active = mode
  def forward(self, x):
    if self._active:
      a = torch.randint(high=x.shape[-1], size=(1,), generator=self._gen).item()
      return torch.roll(x, (a, a), dims=(2,3))
    else: return x

# helper class to normalize a distance matrix using its maximum value
#####################################################################
class Normalize(nn.Module):
  def __init__(self, norm='fro'):
    super().__init__()
    match norm:
      case 'max':
        def f(x):
          s = x.shape # Batch, Chan, Height, Width
          return torch.linalg.norm( x.view(s[0], s[1], s[2]*s[3]) # B, C, H*W
                                  , ord=float('inf')
                                  , dim=-1 ) # result: B, C
      case _: f = lambda x: torch.linalg.norm(x, ord=norm, dim=(-2, -1))
    self._norm = f
  def forward(self, x):
    # get the maximum value in each distance matrix
    s = x.shape # Batch, Chan, Height, Width
    n = self._norm(x)
    # reshape the norm tensor
    n = n[:, -1] # get down to B norms
    n = n.view(s[0], 1, 1, 1) # view as B, 1, 1, 1
    n = n.repeat(1, s[1], s[2], s[3]) # back up to B, C, H, W, used for division
    return x / n

# helper class for padded convolution module
############################################
def circularPadConv2d(in_chan, out_chan, kern_sz=3, stride=1, padding=1, bias=False):
  return nn.Sequential( nn.CircularPad2d(padding)
                      , nn.Conv2d( in_chan, out_chan
                                 , kernel_size=kern_sz, stride=stride
                                 , bias=bias ) )

# helper basic block to redefine encoder resnet with padding
class PaddedBasicBlock(models.resnet.BasicBlock):
  def __init__(self, in_chan, out_chan, *args, **kwargs):
    super().__init__(in_chan, out_chan, *args, **kwargs)
    self.conv1 = circularPadConv2d(in_chan, out_chan, kern_sz=3, stride=self.stride, padding=1)
    self.conv2 = circularPadConv2d(out_chan, out_chan, kern_sz=3, stride=1, padding=1)

# redefined resnet wrapper
def my_resnet18(*, padding=True, weights = None, progress = True, **kwargs):
  weights = models.resnet.ResNet18_Weights.verify(weights)
  block = PaddedBasicBlock if padding else models.resnet.BasicBlock
  model = models.resnet._resnet(block, [2, 2, 2, 2], weights, progress, **kwargs)
  if padding:
    model.conv1 = circularPadConv2d(1, 64, 7, 2, 3)
  else:
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
  model.maxpool = nn.Sequential( nn.CircularPad2d(1)
                               , nn.MaxPool2d(kernel_size=3, stride=2, padding=0))
  model.fc = nn.Identity() # Remove final fully connected layer
  return model

# MyNet
class MyNet(nn.Module):

  # constructor
  #############
  def __init__( self, latent_dim, matrix_size, padding
              , decoder_channels=512
              , normalize_input=None
              , rescale_input=None
              , augment_input=False
              ):
    super().__init__()

    # properties
    self.matrix_size = matrix_size
    self.padding = padding
    self.latent_dim = latent_dim

    # preproc #
    # ------- #
    preproc_steps = []
    # optional normalization step
    match normalize_input:
      # For Frobenius normalization, also rescale by the number of points in the
      # distance matrix
      case 'fro':
        preproc_steps.append(Normalize('fro'))
        preproc_steps.append(Scale(matrix_size))
      # Normalize by the maximum value in the distance matrix
      case 'max':
        preproc_steps.append(Normalize('max'))
      # ignore unsupported normalization
      case _: pass
    # optional extra rescaling step
    if rescale_input: preproc_steps.append(Scale(rescale_input))
    # optional random roll augmentation step
    if augment_input: preproc_steps.append(RandRoll())
    # gather preproc steps
    self.preproc = nn.Sequential(*preproc_steps) if preproc_steps else nn.Identity()

    # encoder #
    # ------- #
    self.encoder = my_resnet18(padding=padding)

    # latent space #
    # ------------ #
    # Note: there are 512 channels in the last layer of the resnet encoder, we
    # map them to the requested latent space size
    self.z_mean = nn.Linear(512, latent_dim)
    self.z_log_var = nn.Linear(512, latent_dim)

    # decoder #
    # ------- #
    # front layer, from latent space
    self.decoder_fc = nn.Sequential( nn.Linear(self.latent_dim, decoder_channels)
                                   , nn.ReLU() )
    # structure of the decoder based on the matrix size
    n_steps = math.ceil(math.log2(matrix_size))
    decoder_steps = []
    n_chans = decoder_channels
    for i in range(n_steps, 0, -1):
      # for every step but the last, halve the number of channels
      if i != 1:
        next_chans = math.ceil(n_chans/2) # lock at 1 channel min
        decoder_steps.extend([ nn.ConvTranspose2d( n_chans, next_chans
                                                 , kernel_size=4
                                                 , stride=2
                                                 , padding=1 )
                            , nn.ReLU() ])
      # on the last step, output a single channel
      else:
        decoder_steps.append(nn.ConvTranspose2d( n_chans, 1
                                               , kernel_size=4
                                               , stride=2
                                               , padding=1 ))
      # update the number of channels for the next step
      n_chans = next_chans
    # pass the structure as a nn.Sequential block
    self.decoder_conv = nn.Sequential(*decoder_steps)
    # assemble overall decoder
    self.decoder = nn.Sequential( self.decoder_fc
                                , View((-1, decoder_channels, 1, 1))
                                , self.decoder_conv )

  ########
  def reparameterize(self, mean, log_var):
    std = torch.exp(0.5 * log_var)
    epsilon = torch.randn_like(std)
    return mean + epsilon * std

  ########
  def forward(self, x):
    # preserve original scale
    og_scale = torch.linalg.norm(x, ord='fro', dim=(-2, -1)) * self.matrix_size
    # preproc
    x_preproc_og = self.preproc(x)
    # encode (both og and flipped preprocessed x)
    x_preproc_flipped = torch.flip(x_preproc_og, dims=(2,3))
    x_encoded_og = self.encoder(x_preproc_og)
    x_encoded_flipped = self.encoder(x_preproc_flipped)
    x_encoded = torch.add(x_encoded_og, x_encoded_flipped)
    # latent space
    z_mean = self.z_mean(x_encoded)
    z_log_var = self.z_log_var(x_encoded)
    z = self.reparameterize(z_mean, z_log_var)
    # decode
    x_recon = self.decoder(z)
    return x_preproc_og, x_recon, z, z_mean, z_log_var, og_scale
