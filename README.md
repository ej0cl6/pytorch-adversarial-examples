## Adversarial Examples

PyTorch implementation following algorithms:

- Fast Gradient Sign Method (FGSM) [1]
- Basic Iterative Method (BIM) [2]
- DeepFool [3]

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
    	<td align="center"> <img src="https://raw.githubusercontent.com/ej0cl6/pytorch-adversarial-examples/master/results/Clean/0.png" width="100%"> </td>
        <td align="center"> <img src="https://raw.githubusercontent.com/ej0cl6/pytorch-adversarial-examples/master/results/Clean/1.png" width="100%"> </td>
        <td align="center"> <img src="https://raw.githubusercontent.com/ej0cl6/pytorch-adversarial-examples/master/results/Clean/2.png" width="100%"> </td>
        <td align="center"> <img src="https://raw.githubusercontent.com/ej0cl6/pytorch-adversarial-examples/master/results/Clean/3.png" width="100%"> </td>
        <td align="center"> <img src="https://raw.githubusercontent.com/ej0cl6/pytorch-adversarial-examples/master/results/Clean/4.png" width="100%"> </td>
        <td align="center"> <img src="https://raw.githubusercontent.com/ej0cl6/pytorch-adversarial-examples/master/results/Clean/5.png" width="100%"> </td>
        <td align="center"> <img src="https://raw.githubusercontent.com/ej0cl6/pytorch-adversarial-examples/master/results/Clean/6.png" width="100%"> </td>
        <td align="center"> <img src="https://raw.githubusercontent.com/ej0cl6/pytorch-adversarial-examples/master/results/Clean/7.png" width="100%"> </td>
        <td align="center"> <img src="https://raw.githubusercontent.com/ej0cl6/pytorch-adversarial-examples/master/results/Clean/8.png" width="100%"> </td>
        <td align="center"> <img src="https://raw.githubusercontent.com/ej0cl6/pytorch-adversarial-examples/master/results/Clean/9.png" width="100%"> </td>
    </tr>
</table>

#### FGSM (eps=0.15)
<table border=0>
    <tr>		
        <td align="center">Label: 2 </td>
        <td align="center">Label: 8 </td>
        <td align="center">Label: 1 </td>
        <td align="center">Label: 2 </td>
        <td align="center">Label: 9 </td>
        <td align="center">Label: 3 </td>
        <td align="center">Label: 5 </td>
        <td align="center">Label: 2 </td>
        <td align="center">Label: 1 </td>
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
    <tr>
        <td align="center"> <img src="https://raw.githubusercontent.com/ej0cl6/pytorch-adversarial-examples/master/results/FGSM/0_diff.png" width="100%"> </td>
        <td align="center"> <img src="https://raw.githubusercontent.com/ej0cl6/pytorch-adversarial-examples/master/results/FGSM/1_diff.png" width="100%"> </td>
        <td align="center"> <img src="https://raw.githubusercontent.com/ej0cl6/pytorch-adversarial-examples/master/results/FGSM/2_diff.png" width="100%"> </td>
        <td align="center"> <img src="https://raw.githubusercontent.com/ej0cl6/pytorch-adversarial-examples/master/results/FGSM/3_diff.png" width="100%"> </td>
        <td align="center"> <img src="https://raw.githubusercontent.com/ej0cl6/pytorch-adversarial-examples/master/results/FGSM/4_diff.png" width="100%"> </td>
        <td align="center"> <img src="https://raw.githubusercontent.com/ej0cl6/pytorch-adversarial-examples/master/results/FGSM/5_diff.png" width="100%"> </td>
        <td align="center"> <img src="https://raw.githubusercontent.com/ej0cl6/pytorch-adversarial-examples/master/results/FGSM/6_diff.png" width="100%"> </td>
        <td align="center"> <img src="https://raw.githubusercontent.com/ej0cl6/pytorch-adversarial-examples/master/results/FGSM/7_diff.png" width="100%"> </td>
        <td align="center"> <img src="https://raw.githubusercontent.com/ej0cl6/pytorch-adversarial-examples/master/results/FGSM/8_diff.png" width="100%"> </td>
        <td align="center"> <img src="https://raw.githubusercontent.com/ej0cl6/pytorch-adversarial-examples/master/results/FGSM/9_diff.png" width="100%"> </td>
    </tr>
</table>

#### BIM (eps=0.15, eps_iter=0.01, n_iter=50)
<table border=0>
    <tr>		
        <td align="center">Label: 7 </td>
        <td align="center">Label: 8 </td>
        <td align="center">Label: 3 </td>
        <td align="center">Label: 2 </td>
        <td align="center">Label: 9 </td>
        <td align="center">Label: 3 </td>
        <td align="center">Label: 5 </td>
        <td align="center">Label: 2 </td>
        <td align="center">Label: 1 </td>
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
    <tr>
        <td align="center"> <img src="https://raw.githubusercontent.com/ej0cl6/pytorch-adversarial-examples/master/results/BIM/0_diff.png" width="100%"> </td>
        <td align="center"> <img src="https://raw.githubusercontent.com/ej0cl6/pytorch-adversarial-examples/master/results/BIM/1_diff.png" width="100%"> </td>
        <td align="center"> <img src="https://raw.githubusercontent.com/ej0cl6/pytorch-adversarial-examples/master/results/BIM/2_diff.png" width="100%"> </td>
        <td align="center"> <img src="https://raw.githubusercontent.com/ej0cl6/pytorch-adversarial-examples/master/results/BIM/3_diff.png" width="100%"> </td>
        <td align="center"> <img src="https://raw.githubusercontent.com/ej0cl6/pytorch-adversarial-examples/master/results/BIM/4_diff.png" width="100%"> </td>
        <td align="center"> <img src="https://raw.githubusercontent.com/ej0cl6/pytorch-adversarial-examples/master/results/BIM/5_diff.png" width="100%"> </td>
        <td align="center"> <img src="https://raw.githubusercontent.com/ej0cl6/pytorch-adversarial-examples/master/results/BIM/6_diff.png" width="100%"> </td>
        <td align="center"> <img src="https://raw.githubusercontent.com/ej0cl6/pytorch-adversarial-examples/master/results/BIM/7_diff.png" width="100%"> </td>
        <td align="center"> <img src="https://raw.githubusercontent.com/ej0cl6/pytorch-adversarial-examples/master/results/BIM/8_diff.png" width="100%"> </td>
        <td align="center"> <img src="https://raw.githubusercontent.com/ej0cl6/pytorch-adversarial-examples/master/results/BIM/9_diff.png" width="100%"> </td>
    </tr>
</table>

#### DeepFool (max_iter=50)
<table border=0>
    <tr>		
        <td align="center">Label: 9 </td>
        <td align="center">Label: 8 </td>
        <td align="center">Label: 3 </td>
        <td align="center">Label: 8 </td>
        <td align="center">Label: 9 </td>
        <td align="center">Label: 3 </td>
        <td align="center">Label: 5 </td>
        <td align="center">Label: 8 </td>
        <td align="center">Label: 3 </td>
        <td align="center">Label: 7 </td>
    </tr>
    <tr>
    	<td align="center"> <img src="https://raw.githubusercontent.com/ej0cl6/pytorch-adversarial-examples/master/results/DeepFool/0_adversarial.png" width="100%"> </td>
        <td align="center"> <img src="https://raw.githubusercontent.com/ej0cl6/pytorch-adversarial-examples/master/results/DeepFool/1_adversarial.png" width="100%"> </td>
        <td align="center"> <img src="https://raw.githubusercontent.com/ej0cl6/pytorch-adversarial-examples/master/results/DeepFool/2_adversarial.png" width="100%"> </td>
        <td align="center"> <img src="https://raw.githubusercontent.com/ej0cl6/pytorch-adversarial-examples/master/results/DeepFool/3_adversarial.png" width="100%"> </td>
        <td align="center"> <img src="https://raw.githubusercontent.com/ej0cl6/pytorch-adversarial-examples/master/results/DeepFool/4_adversarial.png" width="100%"> </td>
        <td align="center"> <img src="https://raw.githubusercontent.com/ej0cl6/pytorch-adversarial-examples/master/results/DeepFool/5_adversarial.png" width="100%"> </td>
        <td align="center"> <img src="https://raw.githubusercontent.com/ej0cl6/pytorch-adversarial-examples/master/results/DeepFool/6_adversarial.png" width="100%"> </td>
        <td align="center"> <img src="https://raw.githubusercontent.com/ej0cl6/pytorch-adversarial-examples/master/results/DeepFool/7_adversarial.png" width="100%"> </td>
        <td align="center"> <img src="https://raw.githubusercontent.com/ej0cl6/pytorch-adversarial-examples/master/results/DeepFool/8_adversarial.png" width="100%"> </td>
        <td align="center"> <img src="https://raw.githubusercontent.com/ej0cl6/pytorch-adversarial-examples/master/results/DeepFool/9_adversarial.png" width="100%"> </td>
    </tr>
    <tr>
        <td align="center"> <img src="https://raw.githubusercontent.com/ej0cl6/pytorch-adversarial-examples/master/results/DeepFool/0_diff.png" width="100%"> </td>
        <td align="center"> <img src="https://raw.githubusercontent.com/ej0cl6/pytorch-adversarial-examples/master/results/DeepFool/1_diff.png" width="100%"> </td>
        <td align="center"> <img src="https://raw.githubusercontent.com/ej0cl6/pytorch-adversarial-examples/master/results/DeepFool/2_diff.png" width="100%"> </td>
        <td align="center"> <img src="https://raw.githubusercontent.com/ej0cl6/pytorch-adversarial-examples/master/results/DeepFool/3_diff.png" width="100%"> </td>
        <td align="center"> <img src="https://raw.githubusercontent.com/ej0cl6/pytorch-adversarial-examples/master/results/DeepFool/4_diff.png" width="100%"> </td>
        <td align="center"> <img src="https://raw.githubusercontent.com/ej0cl6/pytorch-adversarial-examples/master/results/DeepFool/5_diff.png" width="100%"> </td>
        <td align="center"> <img src="https://raw.githubusercontent.com/ej0cl6/pytorch-adversarial-examples/master/results/DeepFool/6_diff.png" width="100%"> </td>
        <td align="center"> <img src="https://raw.githubusercontent.com/ej0cl6/pytorch-adversarial-examples/master/results/DeepFool/7_diff.png" width="100%"> </td>
        <td align="center"> <img src="https://raw.githubusercontent.com/ej0cl6/pytorch-adversarial-examples/master/results/DeepFool/8_diff.png" width="100%"> </td>
        <td align="center"> <img src="https://raw.githubusercontent.com/ej0cl6/pytorch-adversarial-examples/master/results/DeepFool/9_diff.png" width="100%"> </td>
    </tr>
</table>

### Reference

[1] Ian J. Goodfellow, Jonathon Shlens, Christian Szegedy. 
    Explaining and Harnessing Adversarial Examples.
    ICLR, 2015

[2] Alexey Kurakin, Ian J. Goodfellow, Samy Bengio.
    Adversarial Examples in the Physical World.
    arXiv, 2016

[3] Seyed-Mohsen Moosavi-Dezfooli, Alhussein Fawzi, Pascal Frossard.
    DeepFool: A Simple and Accurate Method to Fool Deep Neural Networks.
    CVPR, 2016

### Author

Kuan-Hao Huang / [@ej0cl6](http://ej0cl6.github.io/)
