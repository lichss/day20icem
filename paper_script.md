### 第二部分 intro

深度估计分为有两大分支，相对深度估计（RDE）和绝对深度估计(MDE)，
其中绝对深度估计是主流，因为其在规划、导航、物体识别、3D重建和图像编辑方向有抢进的实用价值。

深度估计工作的第二个分支是相对深度估计，相对深度估计通过弱化深度图(depth map
)中的尺度信息来处理环境中深度尺的的大幅度变化，每个像素的深度预测值仅先对于彼此一致。
一种已经被证实可以作为有效的提升单目深度估计准去率和可迁移性的方法是，将深度估计的回归任务在一定程度上转化为分类任务。


这篇文章将会提出一种two-stager架构，尝试由相对深度估计值得到绝对深度估计值。


结合相对深度估计的易获取的特点和绝对深度估计的尺度信息。

深度估计是不适定问题。

相对深度估计更容易获取，但应用场景受限
不如绝对深度估计。

绝对深度估计一直以来都面临xx挑战。

//绝对深度估计得任务可以简述如下

一种已经被证实可以作为有效的提升单目深度估计准去率和可迁移性的方法是，将深度估计的回归任务在一定程度上转化为分类任务。


### 第三部分 method

*相对深度估计可以转换为绝对深度估计。如果能做到在同一张图片中的深度层次相同*
*理论上来说就可通过一个单调函数映射到真实的深度值上。*
**

#### Backgroud
* chs

有很多传统的learning-based monocular 单目深度估计采用才有卷积层的encoder-decoder架构，最近还出现了许多采用diffusion 模块。深度估计通常被认为是逐像素的回归任务，然而有一系列研究认为可以将深度估计是做是回归和分类结合的任务。这类研究提出的模型通常具有一个模块用于将预测的深度范围划分为区间。我们提出的reflector在此基础上进行改进。
生成深度范围区间的同时使用相对深度估计和全局和的视野改善深度估计值。

我们使用MiDaS [1]的训练策略进行相对深度预测。MiDaS使用一种对尺度和偏移不变的损失函数。如果有多个数据集可用，则使用多任务损失函数以确保数据集之间的帕累托最优性。MiDaS的训练策略可以应用于许多不同的网络架构。我们使用DPT编码器-解码器架构作为基础模型[1]，但用更近的基于变换器的骨干网络[3]替换了编码器。在预训练MiDaS模型进行相对深度预测后，我们通过将提出的度量区间模块附加到解码器上，添加一个或多个用于度量深度预测的头部（见图2的整体架构）。度量区间模块输出度量深度，并遵循自适应区间划分原则，该原则最初在[1]中引入，并随后被[1, 1, 1, 1]修改。特别是，我们从像素级预测设计开始，如LocalBins [1]中所述，并提出进一步改进性能的修改。最后，我们端到端地微调整个架构。

在第一阶段，我们使用xxx的网络结构得到相对深度估计，相对深度估计网络占据模型的大部分参数。
在后续的过程中我们逐步把相对深度估计转换为绝对深度估计。在获得相对深度估计图的同时，backbone的bottle neck会作为输入传入下一个阶段。

#### Overview
* chs
我们使用backbone得到的 bottleneck block
reflector将深度值概率和相对深度结合，反应全局的深度值分布。组成一个连续的区间。这种映射方法是自适应的(随图像的变化而变化)。reflector的backbone采用编码器-解码器结构。最终的深度估计是通过预测区间上的像素级概率分布并计算预测的全局期望值来获得的。
我们在ZoeDepth的基本思想上进行构建，以估计深度分布。但我们改变了结构以引入两个新颖的想法。我们增加了两个特殊的结构使预测深度能更多的参考相对深度估计的信息。
-----

* eng：
Reflector integrates the probability of depth values with relative depth to mirror the overall distribution of depth values, composing a continuous range. This mapping approach is adaptive (it varies with changes in the image). Reflector's backbone utilizes an encoder-decoder architecture. The ultimate depth estimate is derived through predicting pixel-level probability distributions across the intervals and computing the predicted global expectation.

We build on the core ideas of ZoeDepth for estimating depth distributions but modify the structure to introduce two innovative elements. We incorporate two specific architectures to allow the predicted depths to better reference information from relative depth estimations.


#### Architecture Deatail
* chs
reflector同时获得相对深度估计的结果和backbone部分中的得到的feature blocks，
reflector 首先根据相对深度图对每一个像素生成一个的区间，并在后续的模块中不断地调整这个值。
调整的策略是，在区间中加入一些点，这些点会在后续的模块中被修改，直观的理解就是这些点在生成的深度区间上移动，
最后根据这些点所在的位置来预测得到单一像素点的绝对深度估计值。
这些点在调整的过程中会逐步具有更加广阔的视野，注意到周围像素的分布。



我们通过调整区间来实现多尺度细化，即在深度区间上将它们向左或向右移动。利用多尺度特征，我们预测深度区间上的一组点，这些点是区间中心被吸引的方向。

reflector在一个连续的区间中根据概率结合最终得到的深度估计值。
refloctor根据先验的深度信息和相对深度估计值得到绝对深度估计。

projector

attractor

reflext instead of split




#### 结构
我们使用在计算机视觉领域常见的backbone


#### related w

Learning-based MDE has witnessed tremendous progress in recent
 years. Saxena et al. [23] proposed a pioneering work that uses Markov Random Field to capture
 critical local- and global-image features for depth estimation. Later, Eigen et al. [1] introduced
 a convolutional neural network (CNN)-based architecture to attain multi-scale depth predictions.
 Since then, CNNs have been extensively studied in MDE. For instance, Laina et al. [24] utilized
 residual CNN [25] for better optimization. Recently, Transformer [26] has attracted widespread
 attention in the computer vision community [27–29].
_这段写得挺好的。我想炒_ 


In recent years, the field of learning-based monocular depth estimation (MDE) has witnessed significant advancements. 
In early research, Saxena et al. pioneered the use of Markov Random Fields to capture critical local and global image features essential for depth estimation. Following this, Eigen et al. introduced a convolutional neural network (CNN)-based architecture that achieved multi-scale depth predictions. Since then, research on CNNs within the MDE domain has deepened continuously. For example, Laina et al. adopted residual CNNs to achieve better optimization effects. Recently, the Transformer architecture has garnered considerable attention within the computer vision community and has gradually been applied in this field, further driving technological development [23-29].
自从

xxx 和 xxx 等人提出，将深度估计任务转化为分类任务可以是模型的性能提高，即把预测连续分布的深度值转换为对深度值所在的区间行分类。

此外， xxx xxx xxx等人进一步将深度估计任务重构为分类-回归混合任务。其中
AdaBins是先驱性的工作。

Some researchers have proposed that transforming the depth estimation task into a classification task can improve model performance. This is achieved by converting the prediction of continuously distributed depth values into the classification of the intervals in which these depth values fall. This approach can simplify the problem and potentially make the model easier to train.

Furthermore, other researchers have restructured the depth estimation task into a hybrid classification-regression task. Under this framework, AdaBins stands as pioneering work that demonstrates the potential of this method. The AdaBins method first groups depth values into different "bins" (intervals) as a discretization step, and then performs regression on each bin to obtain precise depth estimates. This hybrid strategy not only leverages the advantages of classification but also retains the capability of regression, thus achieving a good balance between accuracy and efficiency.

深度估计可以按照训练方式分为有监督，无监督，自监督。
无监督的深度估计不使用groud truth，这类模型通过3D reconstruccitn 学习估计深度。这些methods的典型方法是采用视频作为dataset。

我们的训练方法属于有监督单目深度估计

### Training Strategies
* chs
如前所述，我们提出的模型训练分为两个阶段。MiDas 骨干玩咯的相对深度训练和reflector绝对深度预测的深度微调。

#### Datasets
我们主要的数据集是NYU Depth v2 。
NYU Depth v2 数据集专注一室内深度估计。对于预训练的相对深度估计网络我们使用了[1]的混合数据集。


### 引用条目
- [1] Huan Fu, Mingming Gong, Chaohui Wang, Kayhan Batmanghelich, and Dacheng Tao. Deep
 ordinal regression network for monocular depth estimation. In Proceedings of the IEEE
 Conference on Computer Vision and Pattern Recognition, pages 2002–2011, 2018.

 - [1] Yuanzhouhan Cao, Zifeng Wu, and Chunhua Shen. Estimating depth from monocular images as
 classification using deep fully convolutional residual networks. IEEE Transactions on Circuits
 and Systems for Video Technology, 28(11):3174–3182, 2017.
 - [1] Ashutosh Saxena, Sung H Chung, Andrew Y Ng, et al. Learning depth from single monocular
 images. In Advances in Neural Information Processing Systems, volume 18, pages 1–8, 2005.
 - [1] Iro Laina, Christian Rupprecht, Vasileios Belagiannis, Federico Tombari, and Nassir Navab.
 Deeper depth prediction with fully convolutional residual networks. In 2016 Fourth International
 Conference on 3D Vision, pages 239–248. IEEE, 2016.
 -  [1] 
 -  [1] 
 -  [1] 
 -  [1] 
 -  [1] 
 -  [1] 
 -  [1] 
 -  [1] 
 -  [1] 
 -  [1] 
 -  [1] 
 -  [1] 
 -  [1] 
 -  [1] 
 -  