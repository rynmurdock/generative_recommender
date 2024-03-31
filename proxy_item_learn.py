

USER = 100
# much >80 seems to break
N_IMAGES = 1000
N_USERS = 20000
import torch
import numpy as np
import pandas as pd
import tqdm

# From https://alexhwilliams.info/itsneuronalblog/2018/02/26/censored-lstsq/
def censored_lstsq(A, B, M):
    """Solves least squares problem subject to missing data.

    Note: uses a broadcasted solve for speed.

    Args
    ----
    A (ndarray) : m x r matrix
    B (ndarray) : m x n matrix
    M (ndarray) : m x n binary matrix (zeros indicate missing values)

    Returns
    -------
    X (ndarray) : r x n matrix that minimizes norm(M*(AX - B))
    """

    # Note: we should check A is full rank but we won't bother...

    # if B is a vector, simply drop out corresponding rows in A
    #if B.ndim == 1 or B.shape[1] == 1:
    #   return np.linalg.lstsq(A[M.to(torch.long)], B[M.to(torch.long)])[0]

    # else solve via tensor representation
    rhs = np.dot(A.T, M * B).T[:,:,None] # n x r x 1 tensor
    T = np.matmul(A.T[None,:,:], M.T[:,:,None] * A[None,:,:]) # n x r x r tensor
    res = torch.linalg.lstsq(T, torch.from_numpy(rhs))
    return res.solution.to('cpu').squeeze().T # transpose to get r x n

class Collaborator(torch.nn.Module):
    
    def __init__(self, image_features_path='image_features.pt', interactions_df_path='replace_flicker_each_to_each.hdf'):
        super().__init__()
        
        self.image_features = torch.load(image_features_path).to('cpu').to(torch.float32)[:N_IMAGES]
        
        df = pd.read_hdf(interactions_df_path, key='huh',).to_numpy()
        self.mask = (torch.from_numpy(df) != 0).to(torch.float)[:N_USERS, :N_IMAGES]
        self.interactions = (torch.from_numpy(df / 5).to(torch.float))[:N_USERS, :N_IMAGES]
        
        self.learned_user_features = torch.randn(self.interactions.shape[0], self.image_features.shape[-1])
        get_back = torch.clone(self.learned_user_features[USER])
        
        US_THRESHOLD = 1
        IM_THRESHOLD = 1
        # remove zeroed out (unrated) subjects/images or below a threshold
        
        self.im_frequency = torch.sum(self.interactions > 0, 0, keepdim=False)
        self.interactions = self.interactions[:, self.im_frequency >= IM_THRESHOLD]
        
        self.us_frequency = torch.sum(self.interactions > 0, 1, keepdim=False)
        
        self.interactions = self.interactions[torch.logical_and(self.us_frequency >= US_THRESHOLD, self.us_frequency > 0)]

        
        self.image_features = self.image_features[self.im_frequency >= IM_THRESHOLD]

        self.learned_user_features = self.learned_user_features[torch.logical_and(self.us_frequency >= US_THRESHOLD, self.us_frequency > 0)]
        self.mask = self.mask[torch.logical_and(self.us_frequency >= US_THRESHOLD, self.us_frequency > 0)]
        self.mask = self.mask[:, self.im_frequency >= IM_THRESHOLD]
        self.frequency = self.im_frequency[self.im_frequency >= IM_THRESHOLD]
        self.us_frequency = self.us_frequency[torch.logical_and(self.us_frequency >= US_THRESHOLD, self.us_frequency > 0)]

        match_images = torch.zeros(self.image_features.shape[0], self.image_features.shape[0])
        match_images = match_images.fill_diagonal_(1)
        self.item_features = torch.clone(self.image_features)

        self.anchor_interactions_and_images = torch.cat([self.interactions, match_images], 0)
        
        self.user_images_items_mask = torch.cat([self.mask.T, match_images], 1)

        print(f'{np.argwhere((self.learned_user_features == get_back).all(1))} is USER')

        assert self.interactions.shape[0] > 1 and self.interactions.shape[1] > 1, f'We need more observations than {self.interactions.shape}'
        print(f'''Shapes:
        Interactions: {self.interactions.shape}
        Image_features: {self.image_features.shape}
        User_features: {self.learned_user_features.shape}
        Image_frequency: {self.frequency.shape}
        Mask: {self.mask.shape}
        ''')
        
        
    def forward(self,):
        loss = torch.linalg.matrix_norm((self.interactions - (self.learned_user_features @ self.item_features.T)) * self.mask)
        return loss
        
with torch.no_grad():

    # TODO visualize USER's images in a folder

    c = Collaborator().to('cpu').to(torch.float32)
    
    for i in range(100):
        least_squares = censored_lstsq(c.item_features, c.interactions.T, c.mask.T)
        c.learned_user_features.data = least_squares.T.to(torch.float32)
        
        print(c.forward(), f'Learned User Embeds; iteration {i}')
        torch.save(c.learned_user_features, 'learned_user_features.pt')
        

        least_squares = censored_lstsq(torch.cat([c.learned_user_features, c.image_features], 0), c.anchor_interactions_and_images, c.user_images_items_mask.T / torch.cat([torch.ones_like(c.us_frequency.unsqueeze(1)), torch.ones_like(c.frequency.unsqueeze(1))], 0))
        c.item_features.data = least_squares.T.to(torch.float32)
        
        print(c.forward(), f'Learned Item Embeds; iteration {i}')
        
    
























