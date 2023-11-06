 
---

<div align="center">    
 
# EnformerTX     
<!-- 
[![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)](https://www.nature.com/articles/nature14539)
[![Conference](http://img.shields.io/badge/NeurIPS-2019-4b44ce.svg)](https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018)
[![Conference](http://img.shields.io/badge/ICLR-2019-4b44ce.svg)](https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018)
[![Conference](http://img.shields.io/badge/AnyConference-year-4b44ce.svg)](https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018)  
 -->
<!--
ARXIV   
[![Paper](http://img.shields.io/badge/arxiv-math.co:1480.1111-B31B1B.svg)](https://www.nature.com/articles/nature14539)
-->
<!-- ![CI testing](https://github.com/PyTorchLightning/deep-learning-project-template/workflows/CI%20testing/badge.svg?branch=master&event=push) 
-->


<!--  
Conference   
-->   
</div>
 
## Description   
Finetune Enformer on expression data. 

## How to run   
First, install dependencies   
```bash
# clone project   
git clone https://github.com/wconnell/enformer-finetune

# install project   
cd enformer-finetune
pip install -e .   
pip install -r requirements.txt
 ```   
Next, download data and test finetuning. The module ships with testing data files under `tests/data/{train,val}.bed`.
 ```bash
# download
bash download-data.sh
# modify options in `config.yaml`...
# launch training
python main.py fit --config config.yaml
```

<!-- 
### Citation   
```
@article{YourName,
  title={Your Title},
  author={Your team},
  journal={Location},
  year={Year}
}
```    
-->
