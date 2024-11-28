# day20icem
just kidding. can't 

拼搏20天 我要发CCF-B

## idea
目前想法还是不去碰backbone,调整一下bins的部分，
现在感觉比较正经的路子就是把attractor部分修改一下。
base里面attractor核心就是1*1卷积过一遍。这里我加参数，看有没有效果。
之前提到的是相对深度估计的结果reletivedepth和attractor结合太晚，看能不能早点结合。
也不一定就要提生性能，能吹一吹也是好的。

或者，我们想个法子看能不能提升泛化性能。
（但是这样可能要面对多个数据集的问题，工作量挺大的。。）

## idea2
 之前改attractor 不好改
 主要是构成att 的核心是projector
 ，projector又主要是1*1的卷积。
 如果把这部分换了参数应该一下就上去了。但性能不保证

纠正一下 
也不能算是1*1卷积 
是1*1卷积 加 relu 
加1*1卷积
应该算是 mlp
合起来算是linear

我祈祷把linear 厚度double一下性能不下降。
之后试试能不能扯个甚么position吧


#### 我们主打一个 从相对深度到绝对深度
就是说 或许可以把Midas core冻结了
我可能提不了性能，但baseline在这种情况下肯定要下降吧

#### seed bin regressor可能是很重要的
或许我们只改这一点
## 11-26
沟吧的hlh 居然玩真的。叫建个overleaf项目。

## 试验记录

修改binregressor
可能是记得东西太多掉点很严重，
base 0.301
掉到 0.328
现在想法是减少点引入的参数再试试吧...有点沮丧