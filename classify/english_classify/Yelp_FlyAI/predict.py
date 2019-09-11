'''
实现模型的调用
'''
from flyai.dataset import Dataset

from model import Model

data = Dataset()
model = Model(data)
p = model.predict(
    text='gute lage im stadtzentrum. shoppingmeile und sehensw  rdigkeiten, sowie gute pubs in laufweite. das hotel ist neu, gut gepflegt und hat bem  htes nettes personal. ideal f  r einen kurztrip nach edinburgh. l  ngere aufenthalte eher nicht, da die zimmer recht klein sind.')
print(p)
