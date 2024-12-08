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
可以认为相对深度估计具有转换为绝对深度估计都潜力

目前有一些方法注重于得到深度在图片分布的概率，即每个像素点对应的深度值服从某种概率分布，将其分类为多种深度值，

在第一阶段，
我们使用xxx的网络结构得到相对深度估计，相对深度估计网络占据模型的大部分参数。
在后续的过程中我们逐步把相对深度估计转换为绝对深度估计。
在获得相对深度估计图的同时，backbone的bottle neck会作为输入传入下一个阶段。
reflector module 主要包含三个结构。
binsregressor
binrg从featureblock中提取初步信息，生成大致的可能性深度区间。
backbone产生的featureblock首先作为brg的输入，brg的输入

projector

attractor

reflext instead of split

我们使用backbone得到的 bottleneck block


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