from transformers import AutoModel, AutoTokenizer , AutoConfig
from myOpenDelta.opendelta import AdapterModel
from optimization import *
#checkpoint = "microsoft/unixcoder_base"

checkpoint = 'Salesforce/codet5p-220m'

set_seed(42)
config = AutoConfig.from_pretrained(checkpoint, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
model = AutoModel.from_pretrained(checkpoint, trust_remote_code=True)
delta = AdapterModel(model , bottleneck_dim=[24])


print(model )



