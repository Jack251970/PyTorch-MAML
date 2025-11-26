# python
import os.path

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

a = [f"Turbine_Data_Penmanshiel_{i:02d}_2022-01-01_-_2023-01-01.csv"
     for i in [1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]]
for item in a:
    if os.path.exists(os.path.join('./.materials/Penmanshiel_SCADA_2022_WT01-15/', item)):
        print(f"{item} Exists")
    else:
        print(f"{item} Not Exists")
