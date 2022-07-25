import torch

t = torch.rand(size=[3, 2])
t_a = t[:, 0:1]
t_b = t[:, 1:]

t_c = torch.cat([t_a, t_b], axis=1)

print(t)
print(t_a)
print(t_b)
print(t_c)
