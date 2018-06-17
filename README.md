## Adversarial Examples

PyTorch implementation following algorithms:

- Fast Gradient Sign Method (FGSM) [1]
- Basic Iterative Method (BIM) [2]

### Prerequisites 

- Python 3.5.2
- PyTorch 0.4.0
- torchvision 0.2.1
- NumPy 1.14.3

### Usage 

    $ python main.py
    
### Dataset

- [MNIST](http://yann.lecun.com/exdb/mnist/)

### Results

#### Clean
<table border=0>
    <tr>		
        <td align="center">Label: 0 </td>
        <td align="center">Label: 1 </td>
        <td align="center">Label: 2 </td>
        <td align="center">Label: 3 </td>
        <td align="center">Label: 4 </td>
        <td align="center">Label: 5 </td>
        <td align="center">Label: 6 </td>
        <td align="center">Label: 7 </td>
        <td align="center">Label: 8 </td>
        <td align="center">Label: 9 </td>
		</tr>
    <tr>
    	  <td align="center"> <img src="https://raw.githubusercontent.com/ej0cl6/pytorch-adversarial-examples/master/results/FGSM/0_clean.png" width="100%"> </td>
        <td align="center"> <img src="https://raw.githubusercontent.com/ej0cl6/pytorch-adversarial-examples/master/results/FGSM/1_clean.png" width="100%"> </td>
        <td align="center"> <img src="https://raw.githubusercontent.com/ej0cl6/pytorch-adversarial-examples/master/results/FGSM/2_clean.png" width="100%"> </td>
        <td align="center"> <img src="https://raw.githubusercontent.com/ej0cl6/pytorch-adversarial-examples/master/results/FGSM/3_clean.png" width="100%"> </td>
        <td align="center"> <img src="https://raw.githubusercontent.com/ej0cl6/pytorch-adversarial-examples/master/results/FGSM/4_clean.png" width="100%"> </td>
        <td align="center"> <img src="https://raw.githubusercontent.com/ej0cl6/pytorch-adversarial-examples/master/results/FGSM/5_clean.png" width="100%"> </td>
        <td align="center"> <img src="https://raw.githubusercontent.com/ej0cl6/pytorch-adversarial-examples/master/results/FGSM/6_clean.png" width="100%"> </td>
        <td align="center"> <img src="https://raw.githubusercontent.com/ej0cl6/pytorch-adversarial-examples/master/results/FGSM/7_clean.png" width="100%"> </td>
        <td align="center"> <img src="https://raw.githubusercontent.com/ej0cl6/pytorch-adversarial-examples/master/results/FGSM/8_clean.png" width="100%"> </td>
        <td align="center"> <img src="https://raw.githubusercontent.com/ej0cl6/pytorch-adversarial-examples/master/results/FGSM/9_clean.png" width="100%"> </td>
    </tr>
</table>

#### FGSM (eps=0.15)
<table border=0>
    <tr>		
        <td align="center">Label: 6 </td>
        <td align="center">Label: 5 </td>
        <td align="center">Label: 1 </td>
        <td align="center">Label: 2 </td>
        <td align="center">Label: 9 </td>
        <td align="center">Label: 3 </td>
        <td align="center">Label: 5 </td>
        <td align="center">Label: 2 </td>
        <td align="center">Label: 4 </td>
        <td align="center">Label: 7 </td>
		</tr>
    <tr>
    	  <td align="center"> <img src="https://raw.githubusercontent.com/ej0cl6/pytorch-adversarial-examples/master/results/FGSM/0_adversarial.png" width="100%"> </td>
        <td align="center"> <img src="https://raw.githubusercontent.com/ej0cl6/pytorch-adversarial-examples/master/results/FGSM/1_adversarial.png" width="100%"> </td>
        <td align="center"> <img src="https://raw.githubusercontent.com/ej0cl6/pytorch-adversarial-examples/master/results/FGSM/2_adversarial.png" width="100%"> </td>
        <td align="center"> <img src="https://raw.githubusercontent.com/ej0cl6/pytorch-adversarial-examples/master/results/FGSM/3_adversarial.png" width="100%"> </td>
        <td align="center"> <img src="https://raw.githubusercontent.com/ej0cl6/pytorch-adversarial-examples/master/results/FGSM/4_adversarial.png" width="100%"> </td>
        <td align="center"> <img src="https://raw.githubusercontent.com/ej0cl6/pytorch-adversarial-examples/master/results/FGSM/5_adversarial.png" width="100%"> </td>
        <td align="center"> <img src="https://raw.githubusercontent.com/ej0cl6/pytorch-adversarial-examples/master/results/FGSM/6_adversarial.png" width="100%"> </td>
        <td align="center"> <img src="https://raw.githubusercontent.com/ej0cl6/pytorch-adversarial-examples/master/results/FGSM/7_adversarial.png" width="100%"> </td>
        <td align="center"> <img src="https://raw.githubusercontent.com/ej0cl6/pytorch-adversarial-examples/master/results/FGSM/8_adversarial.png" width="100%"> </td>
        <td align="center"> <img src="https://raw.githubusercontent.com/ej0cl6/pytorch-adversarial-examples/master/results/FGSM/9_adversarial.png" width="100%"> </td>
    </tr>
</table>

#### BIM (eps=0.15, eps_iter=0.01, n_iter=50)
<table border=0>
    <tr>		
        <td align="center">Label: 6 </td>
        <td align="center">Label: 8 </td>
        <td align="center">Label: 3 </td>
        <td align="center">Label: 2 </td>
        <td align="center">Label: 9 </td>
        <td align="center">Label: 3 </td>
        <td align="center">Label: 5 </td>
        <td align="center">Label: 2 </td>
        <td align="center">Label: 4 </td>
        <td align="center">Label: 7 </td>
    </tr>
    <tr>
    	  <td align="center"> <img src="https://raw.githubusercontent.com/ej0cl6/pytorch-adversarial-examples/master/results/BIM/0_adversarial.png" width="100%"> </td>
        <td align="center"> <img src="https://raw.githubusercontent.com/ej0cl6/pytorch-adversarial-examples/master/results/BIM/1_adversarial.png" width="100%"> </td>
        <td align="center"> <img src="https://raw.githubusercontent.com/ej0cl6/pytorch-adversarial-examples/master/results/BIM/2_adversarial.png" width="100%"> </td>
        <td align="center"> <img src="https://raw.githubusercontent.com/ej0cl6/pytorch-adversarial-examples/master/results/BIM/3_adversarial.png" width="100%"> </td>
        <td align="center"> <img src="https://raw.githubusercontent.com/ej0cl6/pytorch-adversarial-examples/master/results/BIM/4_adversarial.png" width="100%"> </td>
        <td align="center"> <img src="https://raw.githubusercontent.com/ej0cl6/pytorch-adversarial-examples/master/results/BIM/5_adversarial.png" width="100%"> </td>
        <td align="center"> <img src="https://raw.githubusercontent.com/ej0cl6/pytorch-adversarial-examples/master/results/BIM/6_adversarial.png" width="100%"> </td>
        <td align="center"> <img src="https://raw.githubusercontent.com/ej0cl6/pytorch-adversarial-examples/master/results/BIM/7_adversarial.png" width="100%"> </td>
        <td align="center"> <img src="https://raw.githubusercontent.com/ej0cl6/pytorch-adversarial-examples/master/results/BIM/8_adversarial.png" width="100%"> </td>
        <td align="center"> <img src="https://raw.githubusercontent.com/ej0cl6/pytorch-adversarial-examples/master/results/BIM/9_adversarial.png" width="100%"> </td>
    </tr>
</table>


### Reference

[1] Ian J. Goodfellow, Jonathon Shlens, Christian Szegedy. 
    Explaining and Harnessing Adversarial Examples.
    ICLR, 2015

[2] Alexey Kurakin, Ian J. Goodfellow, Samy Bengio.
    Adversarial Examples in the Physical World.
    arXiv, 2016

### Author

Kuan-Hao Huang / [@ej0cl6](http://ej0cl6.github.io/)
