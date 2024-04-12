import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
from .Model import Model
from numpy.random import RandomState


class ConQuatE(Model):
    def __init__(self, config):
        super(ConQuatE, self).__init__(config)
        # self.emb_s_a = nn.Embedding(self.config.entTotal, self.config.hidden_size)
        # self.emb_x_a = nn.Embedding(self.config.entTotal, self.config.hidden_size)
        # self.emb_y_a = nn.Embedding(self.config.entTotal, self.config.hidden_size)
        # self.emb_z_a = nn.Embedding(self.config.entTotal, self.config.hidden_size)
        # self.ent_transfer = nn.Embedding(self.config.entTotal, self.config.hidden_size * 4)
        # self.rel_transfer = nn.Embedding(self.config.relTotal, self.config.hidden_size * 4)
        # self.rel_s_b = nn.Embedding(self.config.relTotal, self.config.hidden_size)
        # self.rel_x_b = nn.Embedding(self.config.relTotal, self.config.hidden_size)
        # self.rel_y_b = nn.Embedding(self.config.relTotal, self.config.hidden_size)
        # self.rel_z_b = nn.Embedding(self.config.relTotal, self.config.hidden_size)
        # self.rel_w = nn.Embedding(self.config.relTotal, self.config.hidden_size)
        self.ent = nn.Embedding(self.config.entTotal, self.config.hidden_size * 4)
        self.rel = nn.Embedding(self.config.relTotal, self.config.hidden_size * 4)
        self.ent_transfer = nn.Embedding(self.config.entTotal, self.config.hidden_size * 4)
        self.rel_transfer = nn.Embedding(self.config.relTotal, self.config.hidden_size * 4)
        self.rel_w = nn.Embedding(self.config.relTotal, self.config.hidden_size)

        self.para = nn.Parameter(torch.tensor([0.5]), requires_grad=True) #定义可学习参数
        self.criterion = nn.Softplus()
        self.fc = nn.Linear(100, 50, bias=False)
        self.ent_dropout = torch.nn.Dropout(self.config.ent_dropout)
        self.rel_dropout = torch.nn.Dropout(self.config.rel_dropout)
        self.bn = torch.nn.BatchNorm1d(self.config.hidden_size)
        self.init_weights()

    def init_weights(self):
            r, i, j, k = self.quaternion_init(self.config.entTotal, self.config.hidden_size)
            r, i, j, k = torch.from_numpy(r), torch.from_numpy(i), torch.from_numpy(j), torch.from_numpy(k)
            vec1 = torch.cat([r, i, j, k], dim=1)
            self.ent.weight.data = vec1.type_as(self.ent.weight.data)
            self.ent_transfer.weight.data = vec1.type_as(self.ent_transfer.weight.data)
            # self.emb_x_a.weight.data = i.type_as(self.emb_x_a.weight.data)
            # self.emb_y_a.weight.data = j.type_as(self.emb_y_a.weight.data)
            # self.emb_z_a.weight.data = k.type_as(self.emb_z_a.weight.data)

            s, x, y, z = self.quaternion_init(self.config.relTotal, self.config.hidden_size)
            s, x, y, z = torch.from_numpy(s[:,:self.config.hidden_size]), torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(z)
            vec2 = torch.cat([s, x, y, z], dim=1)
            self.rel.data = vec2.type_as(self.rel.weight.data)
            self.rel_transfer.data = vec2.type_as(self.rel_transfer.weight.data)
            # self.rel_s_b.weight.data = s.type_as(self.rel_s_b.weight.data)
            # self.rel_x_b.weight.data = x.type_as(self.rel_x_b.weight.data)
            # self.rel_y_b.weight.data = y.type_as(self.rel_y_b.weight.data)
            # self.rel_z_b.weight.data = z.type_as(self.rel_z_b.weight.data)
            nn.init.xavier_uniform_(self.rel_w.weight.data)
    
    def _calc(self, x_a, y_a, z_a, x_c, y_c, z_c, s_b, x_b, y_b, z_b, rel_w, para):
        
    
        denominator_b = torch.sqrt(s_b ** 2 + x_b ** 2 + y_b ** 2 + z_b ** 2)
        denominator_a = torch.sqrt( x_a ** 2 + y_a ** 2 + z_a ** 2)
        denominator_c = torch.sqrt( x_c ** 2 + y_c ** 2 + z_c ** 2)
        rel_r = s_b / denominator_b
        rel_i = x_b / denominator_b
        rel_j = y_b / denominator_b
        rel_k = z_b / denominator_b

        lhs_i = x_a
        lhs_j = y_a
        lhs_k = z_a


        A = - rel_i * lhs_i - rel_j * lhs_j - rel_k * lhs_k
        B = rel_r * lhs_i + rel_j * lhs_k - lhs_j * rel_k
        C = rel_r * lhs_j + rel_k * lhs_i - lhs_k * rel_i
        D = rel_r * lhs_k + rel_i * lhs_j - lhs_i * rel_j

        B1 = -A * rel_i + rel_r * B - C * rel_k + rel_j * D
        C1 = -A * rel_j + rel_r * C - D * rel_i + rel_k * B
        D1 = -A * rel_k + rel_r * D - B * rel_j + rel_i * C


        score_r = (B1 * x_c + C1 * y_c + D1 * z_c) - para*torch.norm(denominator_a*rel_w-denominator_c,p=1,dim=-1).unsqueeze(dim=-1)/x_a.size()[1]

        return -torch.sum(score_r, -1)
    
    def _calc(self, h, r):
        s_a, x_a, y_a, z_a = torch.chunk(h, 4, dim=1)
        s_b, x_b, y_b, z_b = torch.chunk(r, 4, dim=1)

        denominator_b = torch.sqrt(s_b ** 2 + x_b ** 2 + y_b ** 2 + z_b ** 2)
        s_b = s_b / denominator_b
        x_b = x_b / denominator_b
        y_b = y_b / denominator_b
        z_b = z_b / denominator_b

        A = s_a * s_b - x_a * x_b - y_a * y_b - z_a * z_b
        B = s_a * x_b + s_b * x_a + y_a * z_b - y_b * z_a
        C = s_a * y_b + s_b * y_a + z_a * x_b - z_b * x_a
        D = s_a * z_b + s_b * z_a + x_a * y_b - x_b * y_a
        
        Ar = A*s_b+B*x_b+C*y_b+D*z_b
        Br = -A*x_b+B*s_b-C*z_b+D*y_b
        Cr = -A*y_b-B*z_b+C*s_b+D*x_b
        Dr = -A*z_b-B*y_b+C*x_b+D*s_b

        return torch.cat([Ar, Br, Cr, Dr], dim=1)
    '''QuatDE'''
   
    
    def regulation(self, x):
        a, b, c, d = torch.chunk(x, 4, dim=1)
        score = torch.mean(a ** 2) + torch.mean(b ** 2) + torch.mean(c ** 2) + torch.mean(d ** 2)
        return score
    def _transfer(self, x, x_transfer, r_transfer):
        ent_transfer = self._calc(x, x_transfer)
        ent_rel_transfer = self._calc(ent_transfer, r_transfer)

        return ent_rel_transfer
    def loss(self, score,regul):
        return (
                # torch.mean(self.criterion(score * self.batch_y)) + self.config.lmbda * L3_a +   self.config.lmbda * L3_b
                torch.mean(self.criterion(score * self.batch_y)) + self.config.lmbda * regul
        )

    def forward(self):
        h = self.ent(self.batch_h)
        r = self.rel(self.batch_r)
        t = self.ent(self.batch_t)
        # (h, r, t) transfer vector
        h_transfer = self.ent_transfer(self.batch_h)
        t_transfer = self.ent_transfer(self.batch_t)
        r_transfer = self.rel_transfer(self.batch_r)
        h1 = self._transfer(h, h_transfer, r_transfer)
        t1 = self._transfer(t, t_transfer, r_transfer)

        # multiplication as QuatE
        hr = self._calc(h1, r)
        # Inner product as QuatE
        score = torch.sum(hr * t1, -1)
        x_a = x_a_transfer = self.emb_x_a(self.batch_h)
        y_a = y_a_transfer = self.emb_y_a(self.batch_h)
        z_a = z_a_transfer = self.emb_z_a(self.batch_h)
        
        s_b = self.rel_s_b(self.batch_r)
        x_b = self.rel_x_b(self.batch_r)
        y_b = self.rel_y_b(self.batch_r)
        z_b = self.rel_z_b(self.batch_r)
        rel_w = self.rel_w(self.batch_r)
        
        x_c = self.emb_x_a(self.batch_t)
        y_c = self.emb_y_a(self.batch_t)
        z_c = self.emb_z_a(self.batch_t)

        
        para = self.para
        

    
        

        # score = self._calc(x_a, y_a, z_a, x_c, y_c, z_c, s_b, x_b, y_b, z_b, rel_w, para)
        '''Regularization'''
        # L2_a = ( torch.mean( torch.abs(x_a) ** 2)
        #          + torch.mean( torch.abs(y_a) ** 2)
        #          + torch.mean( torch.abs(z_a) ** 2)
        #          + torch.mean( torch.abs(x_c) ** 2)
        #          + torch.mean( torch.abs(y_c) ** 2)
        #          + torch.mean( torch.abs(z_c) ** 2)
        #          )
        # L2_b =  (torch.mean( torch.abs(s_b) ** 2 )
        #          + torch.mean( torch.abs(x_b) ** 2 )
        #          + torch.mean( torch.abs(y_b) ** 2 )
        #          + torch.mean( torch.abs(z_b) ** 2 ))
        # L3_a =  (torch.mean(torch.abs(x_a) ** 3)
        #         + torch.mean(torch.abs(y_a) ** 3)
        #         + torch.mean(torch.abs(z_a) ** 3)
        #         + torch.mean(torch.abs(x_c) ** 3)
        #         + torch.mean(torch.abs(y_c) ** 3)
        #         + torch.mean(torch.abs(z_c) ** 3)) ** (1/3)
        # L3_b =  (torch.mean(torch.abs(s_b) ** 3)
        #         + torch.mean(torch.abs(y_b) ** 3)
        #         + torch.mean(torch.abs(z_b) ** 3)
        #         + torch.mean(torch.abs(x_b) ** 3)) ** (1/3)
       
        regul=self.regulation(h) + self.regulation(r) + self.regulation(t) + \
                self.regulation(h_transfer) + self.regulation(r_transfer) + self.regulation(t_transfer)

        return self.loss(score, regul)

    def predict(self):
        # x_a = x_a_transfer = self.emb_x_a(self.batch_h)
        # y_a = y_a_transfer = self.emb_y_a(self.batch_h)
        # z_a = z_a_transfer = self.emb_z_a(self.batch_h)


        # x_c = x_c_transfer = self.emb_x_a(self.batch_t)
        # y_c = y_c_transfer = self.emb_y_a(self.batch_t)
        # z_c = z_c_transfer = self.emb_z_a(self.batch_t)

        # s_b = s_b_transfer = self.rel_s_b(self.batch_r)
        # x_b = x_b_transfer = self.rel_x_b(self.batch_r)
        # y_b = y_b_transfer = self.rel_y_b(self.batch_r)
        # z_b = z_b_transfer = self.rel_z_b(self.batch_r)
        h = self.ent(self.batch_h)
        r = self.rel(self.batch_r)
        t = self.ent(self.batch_t)
      
        
        h_transfer = self.ent_transfer(self.batch_h)
        t_transfer = self.ent_transfer(self.batch_t)
        r_transfer = self.rel_transfer(self.batch_r)
        s_a, x_a, y_a, z_a = torch.chunk(h1, 4, dim=1)

        s_c, x_c, y_c, z_c = torch.chunk(t1,4,dim=1)
        denominator_a = torch.sqrt( s_a ** 2 + x_a ** 2 + y_a ** 2 + z_a ** 2)
        
        denominator_c = torch.sqrt( s_c **2 + x_c ** 2 + y_c ** 2 + z_c ** 2)
        h1 = self._transfer(h, h_transfer, r_transfer)
        t1 = self._transfer(t, t_transfer, r_transfer)
        hr = self._calc(h1,r)
        rel_w = self.rel_w(self.batch_r)
        para = self.para

        score = torch.sum(hr * t1- para*torch.norm(denominator_a*rel_w-denominator_c,p=1,dim=-1).unsqueeze(dim=-1)/x_a.size()[1], -1)
        # score = self._calc(x_a, y_a, z_a, x_c, y_c, z_c, s_b, x_b, y_b, z_b, rel_w, para)
        return score.cpu().data.numpy()

    def quaternion_init(self, in_features, out_features, criterion='he'):

        fan_in = in_features
        fan_out = out_features

        if criterion == 'glorot':
            s = 1. / np.sqrt(2 * (fan_in + fan_out))
        elif criterion == 'he':
            s = 1. / np.sqrt(2 * fan_in)
        else:
            raise ValueError('Invalid criterion: ', criterion)
        rng = RandomState(123)

        # Generating randoms and purely imaginary quaternions :
        kernel_shape = (in_features, out_features)

        number_of_weights = np.prod(kernel_shape)
        v_i = np.random.uniform(0.0, 1.0, number_of_weights)
        v_j = np.random.uniform(0.0, 1.0, number_of_weights)
        v_k = np.random.uniform(0.0, 1.0, number_of_weights)

        # Purely imaginary quaternions unitary
        for i in range(0, number_of_weights):
            norm = np.sqrt(v_i[i] ** 2 + v_j[i] ** 2 + v_k[i] ** 2) + 0.0001
            v_i[i] /= norm
            v_j[i] /= norm
            v_k[i] /= norm
        v_i = v_i.reshape(kernel_shape)
        v_j = v_j.reshape(kernel_shape)
        v_k = v_k.reshape(kernel_shape)

        modulus = rng.uniform(low=-s, high=s, size=kernel_shape)
        phase = rng.uniform(low=-np.pi, high=np.pi, size=kernel_shape)

        weight_r = modulus * np.cos(phase)
        weight_i = modulus * v_i * np.sin(phase)
        weight_j = modulus * v_j * np.sin(phase)
        weight_k = modulus * v_k * np.sin(phase)

        return (weight_r, weight_i, weight_j, weight_k)