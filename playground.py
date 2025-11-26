import torch
from collections import OrderedDict
from torch import nn

m = nn.Linear(2, 2)
params = OrderedDict(m.named_parameters())

# replacing the dict value: does NOT change the module
params['weight'] = params['weight'] - 1.0
print(torch.equal(m.weight, params['weight']))  # False

# in-place mutation: DOES change the module
params = OrderedDict(m.named_parameters())
params['weight'].data.copy_(params['weight'].data - 1.0)
print(torch.equal(m.weight, params['weight']))  # True

print(not True and not False)
print(not False and not False)
