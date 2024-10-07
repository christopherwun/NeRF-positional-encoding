import os
import gdown
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

url = "https://drive.google.com/file/d/13eBK_LWxs4SkruFKH7glKK9jwQU1BkXK/view?usp=sharing"
gdown.download(url=url, output='lego_data.npz', quiet=False, fuzzy=True)

# Load input images, poses, and intrinsics
data = np.load("lego_data.npz")

# Images
images = data["images"]

# Height and width of each image
height, width = images.shape[1:3]

# Camera extrinsics (poses)
poses = data["poses"]
poses = torch.from_numpy(poses).to(device)
print(poses.shape)

# Camera intrinsics
intrinsics = data["intrinsics"]
intrinsics = torch.from_numpy(intrinsics).to(device)

# Hold one image out (for test).
test_image, test_pose = images[101], poses[101]
test_image = torch.from_numpy(test_image).to(device)

# Map images to device
images = torch.from_numpy(images[:100, ..., :3]).to(device)

plt.imshow(test_image.detach().cpu().numpy())
plt.show()

print(data)

def get_rays(height, width, intrinsics, w_R_c, w_T_c):

    """
    Compute the origin and direction of rays passing through all pixels of an image (one ray per pixel).

    Args:
    height: the height of an image.
    width: the width of an image.
    intrinsics: camera intrinsics matrix of shape (3, 3).
    w_R_c: Rotation matrix of shape (3,3) from camera to world coordinates.
    w_T_c: Translation vector of shape (3,1) that transforms

    Returns:
    ray_origins (torch.Tensor): A tensor of shape (height, width, 3) denoting the centers of
      each ray. Note that desipte that all ray share the same origin, here we ask you to return
      the ray origin for each ray as (height, width, 3).
    ray_directions (torch.Tensor): A tensor of shape (height, width, 3) denoting the
      direction of each ray.
    """

    device = intrinsics.device
    ray_directions = torch.zeros((height, width, 3), device=device)  # placeholder
    ray_origins = torch.zeros((height, width, 3), device=device)  # placeholder

    ray_origins[:,:] = w_T_c

    u, v = torch.meshgrid(torch.arange(width, dtype=torch.float32), torch.arange(height, dtype=torch.float32))
    pixels = torch.stack((u.flatten(), v.flatten(), torch.ones_like(u.flatten())))

    # print(pixels.shape)

    Kinv = torch.linalg.inv(intrinsics)

    # print(w_R_c.shape)
    # print(Kinv.shape)
    # print(pixels.shape)

    ray_directions = ((w_R_c @ Kinv @ pixels).T.reshape(height, width, 3)).transpose(0,1)
    # ray_directions = (w_R_c @ Kinv @ pixels.T).reshape(height, width, 3).transpose(0, 1)

    return ray_origins, ray_directions

def stratified_sampling(ray_origins, ray_directions, near, far, samples):

    """
    Sample 3D points on the given rays. The near and far variables indicate the bounds of sampling range.

    Args:
    ray_origins: Origin of each ray in the "bundle" as returned by the
      get_rays() function. Shape: (height, width, 3).
    ray_directions: Direction of each ray in the "bundle" as returned by the
      get_rays() function. Shape: (height, width, 3).
    near: The 'near' extent of the bounding volume.
    far:  The 'far' extent of the bounding volume.
    samples: Number of samples to be drawn along each ray.

    Returns:
    ray_points: Query 3D points along each ray. Shape: (height, width, samples, 3).
    depth_points: Sampled depth values along each ray. Shape: (height, width, samples).
    """

    height, width, _ = ray_origins.shape

    ray_points = torch.zeros((height, width, samples, 3))
    depth_points = torch.zeros((height, width, samples))

    t = torch.linspace(near, far, samples)

    for i in range(samples):
        ray_points[..., i, :] = ray_origins + t[i] * ray_directions
        depth_points[..., i] = t[i]
    

    return ray_points, depth_points

class nerf_model(nn.Module):

    """
    Define a NeRF model comprising eight fully connected layers and following the
    architecture described in the NeRF paper.
    """

    def __init__(self, filter_size=256, num_x_frequencies=6, num_d_frequencies=3):
        super().__init__()

        # for autograder compliance, please follow the given naming for your layers
        num_x_inputs = 3*(1+2*num_x_frequencies)
        num_d_inputs = 3*(1+2*num_d_frequencies)
        self.layers = nn.ModuleDict({
            'layer_1': nn.Linear(num_x_inputs, filter_size),
            'layer_2': nn.Linear(filter_size, filter_size),
            'layer_3': nn.Linear(filter_size, filter_size),
            'layer_4': nn.Linear(filter_size, filter_size),
            'layer_5': nn.Linear(filter_size, filter_size),
            'layer_6': nn.Linear(filter_size + num_x_inputs, filter_size),
            'layer_7': nn.Linear(filter_size, filter_size),
            'layer_8': nn.Linear(filter_size, filter_size),
            'layer_s': nn.Linear(filter_size, 1),
            'layer_9': nn.Linear(filter_size, filter_size),
            'layer_10': nn.Linear(filter_size + num_d_inputs, filter_size//2),
            'layer_11': nn.Linear(filter_size//2, 3),
        })
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        print("Model initialized")


    def forward(self, x, d):
        # example of forward through a layer: y = self.layers['layer_1'](x)
        y = self.relu(self.layers['layer_1'](x))
        y = self.relu(self.layers['layer_2'](y))
        y = self.relu(self.layers['layer_3'](y))
        y = self.relu(self.layers['layer_4'](y))
        y = self.relu(self.layers['layer_5'](y))

        # print("Concatenating x and y")
        yx = torch.cat((y,x), dim=-1)
        y = self.relu(self.layers['layer_6'](yx))
        y = self.relu(self.layers['layer_7'](y))
        y = self.relu(self.layers['layer_8'](y))
        sigma = self.layers['layer_s'](y)
        y = self.layers['layer_9'](y)

        # print("Concatenating y and d")
        yd = torch.cat((y,d), dim=-1)
        y = self.relu(self.layers['layer_10'](yd))
        rgb = self.sigmoid(self.layers['layer_11'](y))

        return rgb, sigma
    

def get_batches(ray_points, ray_directions, num_x_frequencies, num_d_frequencies):

    def get_chunks(inputs, chunksize = 2**15):
        """
        This fuction gets an array/list as input and returns a list of chunks of the initial array/list
        """
        return [inputs[i:i + chunksize] for i in range(0, inputs.shape[0], chunksize)]

    """
    This function returns chunks of the ray points and directions to avoid memory errors with the
    neural network. It also applies positional encoding to the input points and directions before
    dividing them into chunks, as well as normalizing and populating the directions.
    """
    # normalize ray directions and populate the ray points and directions with positional encoding
    h,w,s,_ = ray_points.shape
    norm_rays = torch.linalg.norm(ray_directions, dim=-1)

    # print(ray_directions.shape)
    # print(ray_directions[0,0,:])
    ray_directions = ray_directions / norm_rays.unsqueeze(-1)

    # print(ray_directions.shape)

    populated_directions = ray_directions.unsqueeze(2).expand(-1,-1,s,-1)
    # print(populated_directions.shape)

    # print("Encoding ray points and directions")
    encoded_ray_points = positional_encoding(ray_points, num_frequencies=num_x_frequencies).reshape(-1, 3*(1+2*num_x_frequencies))
    encoded_ray_directions = positional_encoding(populated_directions, num_frequencies=num_d_frequencies).reshape(-1, 3*(1+2*num_d_frequencies))

    # print("Dividing data into batches")
    # divide the data into batches
    ray_points_batches = get_chunks(encoded_ray_points)
    ray_directions_batches = get_chunks(encoded_ray_directions)

    # print("Done dividing data into batches")
    print("Number of batches: ", len(ray_points_batches), len(ray_directions_batches))
    print("Batch sizes: ", ray_points_batches[0].shape, ray_directions_batches[0].shape)

    return ray_points_batches, ray_directions_batches

def positional_encoding(x, num_frequencies=6, incl_input=True):

    """
    Apply positional encoding to the input.

    Args:
    x (torch.Tensor): Input tensor to be positionally encoded.
      The dimension of x is [N, D], where N is the number of input coordinates,
      and D is the dimension of the input coordinate.
    num_frequencies (optional, int): The number of frequencies used in
     the positional encoding (default: 6).
    incl_input (optional, bool): If True, concatenate the input with the
        computed positional encoding (default: True).

    Returns:
    (torch.Tensor): Positional encoding of the input tensor.
    """

    results = []
    if incl_input:
        results.append(x)
    # encode input tensor and append the encoded tensor to the list of results.
    for pow in range(num_frequencies):
      results.append(torch.sin(2**pow * np.pi * x))
      results.append(torch.cos(2**pow * np.pi * x))

    return torch.cat(results, dim=-1)

def volumetric_rendering(rgb, s, depth_points):

    """
    Differentiably renders a radiance field, given the origin of each ray in the
    "bundle", and the sampled depth values along them.

    Args:
    rgb: RGB color at each query location (X, Y, Z). Shape: (height, width, samples, 3).
    sigma: Volume density at each query location (X, Y, Z). Shape: (height, width, samples).
    depth_points: Sampled depth values along each ray. Shape: (height, width, samples).

    Returns:
    rec_image: The reconstructed image after applying the volumetric rendering to every pixel.
    Shape: (height, width, 3)
    """

    height, width, samples = s.shape

    s = torch.relu(s)

    rgb = rgb.reshape(height*width, samples, 3)
    s = s.reshape(height*width, samples)
    depth_points = depth_points.reshape(height*width, samples)

    deltas = torch.cat((depth_points[:,1:], torch.tensor([10e9]).repeat(height*width, 1)), dim=1) - depth_points

    T = torch.exp(-1 * torch.cat([torch.zeros((height*width,1)), torch.cumsum(s * deltas, dim=1)[:,:-1]], dim=1))

    bigC = torch.sum((1 - (torch.exp(-1*s * deltas))).unsqueeze(2) * T.unsqueeze(2) * rgb, dim=1)

    # reshape into right shape
    rec_image = bigC.reshape((height, width, 3))

    return rec_image

url = "https://drive.google.com/file/d/1ag6MqSh3h4KY10Mcx5fKxt9roGNLLILK/view?usp=sharing"
gdown.download(url=url, output='sanity_volumentric.pt', quiet=False, fuzzy=True)
rbd = torch.load('sanity_volumentric.pt')

r = rbd['rgb']
s = rbd['sigma']
depth_points = rbd['depth_points']
rec_image = volumetric_rendering(r, s, depth_points)

plt.figure(figsize=(10, 5))
plt.imshow(rec_image.detach().cpu().numpy())
plt.title(f'Volumentric rendering of a sphere with $\\sigma={0.2}$, on blue background')
plt.show()

def one_forward_pass(height, width, intrinsics, pose, near, far, samples, model, num_x_frequencies, num_d_frequencies):


    #compute all the rays from the image
    w_R_c = torch.Tensor(pose[:3, :3])
    w_T_c = torch.Tensor(pose[:3, 3])

    ray_origins, ray_directions = get_rays(height, width, intrinsics, w_R_c, w_T_c)

    #sample the points from the rays
    ray_points, depth_points = stratified_sampling(ray_origins, ray_directions, near, far, samples)

    #divide data into batches to avoid memory errors
    ray_points_batches, ray_directions_batches = get_batches(ray_points, ray_directions, num_x_frequencies, num_d_frequencies)

    #forward pass the batches and concatenate the outputs at the end
    rgb_outputs = []
    s_outputs = []
    for i in range(len(ray_points_batches)):
        outputs = model(ray_points_batches[i], ray_directions_batches[i])
        rgb, s = outputs

        # append to lists for concatenation later on
        rgb_outputs.append(rgb)
        s_outputs.append(s)

    # concat
    rgb = torch.cat(rgb_outputs, dim=0)
    s = torch.cat(s_outputs, dim=0)
          
    rgb = rgb.reshape(height, width, samples, -1)
    s = s.reshape(height, width, samples)

    # Apply volumetric rendering to obtain the reconstructed image
    rec_image = volumetric_rendering(rgb, s, depth_points)

    return rec_image

num_x_frequencies = 10
num_d_frequencies = 4
learning_rate  = 5e-4
iterations = 3000
samples = 64
display = 25
near = 0.667
far = 2

model = nerf_model(num_x_frequencies=num_x_frequencies,num_d_frequencies=num_d_frequencies)
model = model.to(device)

def weights_init(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
model.apply(weights_init)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

psnrs = []
iternums = []

t = time.time()
t0 = time.time()

# Train the model
for i in range(iterations+1):
    #choose randomly a picture for the forward pass
    idx = np.random.randint(0, len(images))

    curr_rgb_img = images[idx].to(device)
    pose = poses[idx].to(device)

    # Run one iteration of NeRF and get the rendered RGB image.
    rendered_rgb_img = one_forward_pass(height, width, intrinsics, pose, near, far, samples, model, num_x_frequencies, num_d_frequencies)
    rendered_rgb_img = rendered_rgb_img.to(device)

    # Compute mean-squared error between the predicted and target images. Backprop!
    loss_fn = torch.nn.MSELoss()
    loss = loss_fn(curr_rgb_img, rendered_rgb_img)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Display images/plots/stats
    if i % display == 0:
        with torch.no_grad():
            # Render the held-out view
            test_rec_image = one_forward_pass(height, width, intrinsics, test_pose, near, far, samples, model, num_x_frequencies, num_d_frequencies)
            test_rec_image = test_rec_image.to(device)

        #calculate the loss and the psnr between the original test image and the reconstructed one.
        test_loss = loss_fn(test_image, test_rec_image)
        R = torch.max(test_image)
        psnr = 10 * torch.log10(R**2 / test_loss)

        print("Iteration %d " % i, "Loss: %.4f " % loss.item(), "PSNR: %.2f " % psnr.item(), \
                "Time: %.2f secs per iter, " % ((time.time() - t) / display), "%.2f mins in total" % ((time.time() - t0)/60))

        t = time.time()
        psnrs.append(psnr.item())
        iternums.append(i)

        plt.figure(figsize=(16, 4))
        plt.subplot(141)
        plt.imshow(test_rec_image.detach().cpu().numpy())
        plt.title(f"Iteration {i}")
        plt.subplot(142)
        plt.imshow(test_image.detach().cpu().numpy())
        plt.title("Target image")
        plt.subplot(143)
        plt.plot(iternums, psnrs)
        plt.title("PSNR")
        plt.show()

plt.imsave('test_lego.png',test_rec_image.detach().cpu().numpy())
torch.save(model.state_dict(),'model_nerf.pt')
print('Done!')