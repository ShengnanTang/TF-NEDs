import torch
import torch.nn.functional as F
import torch.nn as nn
import controldiffeq
from vector_fields import *

from utils.ADWT_1D import DWT

from utils.draw import plot_heatmap_seaborn, plot_signal_components
class NeuralGCDE(nn.Module):
    def __init__(self, args, func_f, func_g, input_channels, hidden_channels, output_channels, initial, device, atol, rtol, solver):
        super(NeuralGCDE, self).__init__()
        self.num_node = args.num_nodes
        self.input_dim = input_channels
        self.hidden_dim = hidden_channels
        self.output_dim = output_channels
        self.horizon = args.horizon
        self.num_layers = args.num_layers

        self.default_graph = args.default_graph
        self.node_embeddings = nn.Parameter(torch.randn(self.num_node, args.embed_dim), requires_grad=True)
        
        self.func_f = func_f
        self.func_g = func_g
        self.solver = solver
        self.atol = atol
        self.rtol = rtol
        self.args = args
        #predictor
        self.end_conv = nn.Conv2d(1, args.horizon * self.output_dim, kernel_size=(1, self.hidden_dim), bias=True)

        self.init_type = 'fc'
        if self.init_type == 'fc':
            self.initial_h = torch.nn.Linear(self.input_dim, self.hidden_dim)
            self.initial_z = torch.nn.Linear(self.input_dim, self.hidden_dim)
        elif self.init_type == 'conv':
            self.start_conv_h = nn.Conv2d(in_channels=input_channels,
                                            out_channels=hidden_channels,
                                            kernel_size=(1,1))
            self.start_conv_z = nn.Conv2d(in_channels=input_channels,
                                            out_channels=hidden_channels,
                                            kernel_size=(1,1))
        
        self.M = nn.Parameter(torch.randn(args.lag // 2 ,4))  
        nn.init.xavier_uniform_(self.M)
        self.Ldwt = DWT(configs=args,target_len=args.lag)
        self.mlp = nn.Sequential(
            nn.Linear(3*args.input_dim, 48),
            nn.ReLU(),
            nn.Linear(48, 48),
            nn.ReLU(), 
            nn.Linear(48, args.input_dim),# ← 这个逗号
        )

        
    def forward(self, times, coeffs):
        #source: B, T_1, N, D
        #target: B, T_2, N, D
        #supports = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec1.transpose(0,1))), dim=1)

        spline = controldiffeq.NaturalCubicSpline(times, coeffs)
        interpolated_x = torch.stack([spline.evaluate(t) for t in times], dim=-2)
        enc_x = interpolated_x[:,:,:,1]
        yl,yh = self.Ldwt(enc_x,1)

        yl_upsam = self.Ldwt((yl,(torch.zeros_like(yh[0]),)),0)
        yh_upsam = self.Ldwt((torch.zeros_like(yl),yh),0)
        # plot_signal_components(yh_upsam,yl_upsam,'/root/tangshengnan/STG-NCDE-main/pic/pic.pdf')
        # exit()
        yh_upsam = yh_upsam.unsqueeze(-1).permute(0,2,1,3)
        
        # if self.args.dataset == 'PEMSD4':
        #     yh_times = torch.linspace(0, 11, 12,device=self.args.device)
        # if self.args.dataset == 'PEMSD4':
        #     yh_times = torch.linspace(0, 11, 12,device=self.args.device)
        # elif self.args.dataset == 'token':
        #     yh_times = torch.linspace(0, 6, 7,device=self.args.device)
        # elif self.args.dataset == 'ETTh1':
        #     yh_times = torch.linspace(0, self.args.lag -1 , self.args.lag,device=self.device)
        # elif self.args.dataset == 'ETTh2':
        #     yh_times = torch.linspace(0, self.args.lag -1 , self.args.lag,device=self.device)
        if self.args.dataset in ['PEMSD4', 'PEMSD7', 'PEMSD8','PEMSD3']:
            yh_times = torch.linspace(0, 11, 12, device=self.args.device)
        elif self.args.dataset == 'token':
            yh_times = torch.linspace(0, 6, 7, device=self.args.device)
        elif self.args.dataset in ['ETTh1', 'ETTh2','ETTm1','ETTm2']:
            yh_times = torch.linspace(0, self.args.lag - 1, self.args.lag, device=self.device)

        augmented_X_tra = []
        augmented_X_tra.append(times.unsqueeze(0).unsqueeze(0).repeat(yh_upsam.shape[0],yh_upsam.shape[2],1).unsqueeze(-1).transpose(1,2))
        augmented_X_tra.append(torch.Tensor(yh_upsam[..., :]))
        yh_upsam = torch.cat(augmented_X_tra, dim=3)

        yh_coeffs = controldiffeq.natural_cubic_spline_coeffs(yh_times, yh_upsam.transpose(1,2))

        yh_spline = controldiffeq.NaturalCubicSpline(yh_times, yh_coeffs)


        if self.init_type == 'fc':

            h0 = self.initial_h(spline.evaluate(times[0]))
            z0 = self.initial_z(spline.evaluate(times[0]))
        elif self.init_type == 'conv':
            h0 = self.start_conv_h(spline.evaluate(times[0]).transpose(1,2).unsqueeze(-1)).transpose(1,2).squeeze()
            z0 = self.start_conv_z(spline.evaluate(times[0]).transpose(1,2).unsqueeze(-1)).transpose(1,2).squeeze()


        z_t = controldiffeq.cdeint_gde_dev(dX_dt=spline.derivative, #dh_dt
                                   h0=h0,
                                   z0=z0,
                                   func_f=self.func_f,
                                   func_g=self.func_g,
                                   t=times,
                                   M=self.M,
                                   mlp = self.mlp,
                                   yh_spline=yh_spline,
                                   method=self.solver,
                                   atol=self.atol,
                                   rtol=self.rtol)

        # init_state = self.encoder.init_hidden(source.shape[0])
        # output, _ = self.encoder(source, init_state, self.node_embeddings)      #B, T, N, hidden
        # output = output[:, -1:, :, :]        
                         #B, 1, N, hidden
        z_T = z_t[-1:,...].transpose(0,1)

        #CNN based predictor
        # print(z_T.shape)
        output = self.end_conv(z_T)
  

              #B, T*C, N, 1
        output = output.squeeze(-1).reshape(-1, self.horizon, self.output_dim, self.num_node)

        
        output = output.permute(0, 1, 3, 2)                             #B, T, N, C

        return output