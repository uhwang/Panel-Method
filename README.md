# Panel-Method
Low Order Panel Method <br> 
$$\varphi_S_i={\frac{\lambda_j}{2\pi}\int_{j}{\ln(r_{ij})ds_j}}$$ <br>
$$\varphi_S=\sum_{j=1}^{N}{\frac{\gamma_j}{2\pi}\int_{j}{\ln(r_{ij})ds_j}}$$<br>
$$\varphi_U = U\cos(\alpha) x + U\sin(\alpha)y $$<br>
$$\varphi = \varphi_U + \varphi_s $$ <br>
$$\varphi = U\cos(\alpha) x + U\sin(\alpha)y + \sum_{j=1}^{N}{\frac{\gamma_j}{2\pi}\int_{j}{\ln(r_{ij})ds_j}} $$ <br>
$$\frac{\partial \phi}{\partial n}=0 $$<br>
$$\frac{\partial \phi}{\partial n} = \frac{\partial \phi_U}{\partial n}+\frac{\partial \phi_S}{\partial n}$$<br>

$$\frac{\partial \phi_U}{\partial n}=U\cos(\alpha)\frac{\partial x}{n} + U\sin(\alpha)\frac{\partial y}{n}$$ <br>

$$\frac{\partial \phi_S}{\partial n}=\sum_{j=1}^{N}{\frac{\gamma_j}{2\pi}\int_{j}{\frac{\partial}{\partial n}\ln(r_{ij})ds_j}}$$<br>