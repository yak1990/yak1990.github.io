# DDPM(Denoising Diffusion Probabilistic Models)


## 基础公式

### 前向过程，是一个逐步加入noise的过程

加入噪声的公式为：
$$\epsilon = N(0,I)$$
$$\begin{align}
    X_{t} &= N(\sqrt{1-\beta_{t}}*X_{t-1},\beta_{t}*I)\\
    &=\sqrt{1-\beta_{t}}*X_{t-1}+\sqrt{\beta_{t}}*I
\end{align}$$

假设输入的原始图片是$X_0$，引入新变量：

$$\begin{align}
    \alpha_t=1-\beta_t
\end{align}$$

$$\begin{align}
    \overline{\alpha_t}=\prod_{s=1}^t{\alpha_t}
\end{align}$$

逐步计算，可以推导得出

$$\begin{align}
    X_{t}&=N(\sqrt{\overline{\alpha_t}}*X_0,(1-\overline{\alpha_t})*I)\\
    &=\sqrt{\overline{\alpha_t}}*X_0+\sqrt{(1-\overline{\alpha_t})}*I
\end{align}$$


### 逆向过程，是一个逐步去噪过程
逆向问题的定义是，给定某一步的图片，我们该如何消除该图片中的噪声；即给定$x_{t}$，如何得出$x_{t-1}$

我们先考虑一个比较简单的问题，即如果$x_0$,$x_t$已知，那么$x_{t-1}$的分布是什么

$$\begin{align}
    q(x_{t-1}|x_0,x_t)&=q(x_{t-1},x_t,x_0)/q(x_0,x_t)\\
    &=\frac{q(x_t|x_{t-1})*q(x_{t-1}|x_0)*q(x_0)}{q(x_t|x_0)*q(x_0)}\\
    &=\frac{q(x_t|x_{t-1})*q(x_{t-1}|x_0)}{q(x_t|x_0)}
\end{align}$$

其中这三个分布，均为高斯分布，经过一系列的结算，可以得出（计算过程省略）：
>注意，高斯分布乘以高斯分布，或者高斯分布除以高斯分布，其结果，可能不一定为高斯分布

$$\begin{align}
    q(x_{t-1}|x_0,x_t)=N(\frac{\sqrt{\overline{\alpha}_{t-1}}*\beta_t}{1-\overline{\alpha}_t}*x_0+\frac{\sqrt{\alpha_t}*(1-\overline{\alpha}_{t-1})}{1-\overline{\alpha}_t}*x_t,\frac{1-\overline{\alpha}_{t-1}}{1-\overline{\alpha}_{t}}*I)
\end{align}$$

其中：

$x_t$与$x_0$的关系：

$$\begin{align}
x_t=\sqrt{\overline{\alpha_t}}*x_0+\sqrt{(1-\overline{\alpha_t})}*\epsilon
\end{align}$$

其中$\epsilon$是从标准高斯分布，随机取样得出来的。
我们可以使用一个网络$\epsilon_{\theta}(x_t,t)$，来预测，$x_t$加入的噪声。

此时：

$$\begin{align}
x_0=\frac{x_t-\sqrt{(1-\overline{\alpha_t})}*\epsilon_{\theta}(x_t,t)}{\sqrt{\overline{\alpha_t}}}
\end{align}$$

此时，将$x_0$ , $x_t$ , 代入公式(8)，计算得出$x_{t-1}$


<!-- 结合$x_t$与$x_0$的关系：

$$\begin{align}
x_t=\sqrt{\alpha}_t*x_0+\sqrt{(1-\overline{\alpha}_t)}*\epsilon
\end{align}$$

其中$\epsilon$是从标准高斯分布，随机取样得出来的，我们可以使用一个网络，来预测，给定$x_t$后，其加入的噪声；使用$\epsilon_{\theta}(x_t,t)$来表示，此时：

$$\begin{align}
x_0=\frac{x_t-\sqrt{(1-\overline{\alpha}_t)}*\epsilon_{\theta}(x_t,t)}{\sqrt{\alpha}_t}
\end{align}$$

此时，对于公式(8)，来说，其均值，可以化简得出

$$\begin{align}
Mean(x_{t-1}|x_t)&=\frac{x_t-\frac{\beta_t}{\sqrt{1-\overline{\alpha}_t}}*\epsilon_{\theta}(x_t,t)}{\sqrt{\alpha_t}}\\
&=\frac{x_t-\frac{1-\alpha_t}{\sqrt{1-\overline{\alpha}_t}}*\epsilon_{\theta}(x_t,t)}{\sqrt{\alpha_t}}
\end{align}$$ -->


## 实现细节

1. 超参数设置
    1. 设置$\beta$,这里采用最简单的配置，使用线性取值，获取所有的beta:
        $$\beta_{begin}=0.0001$$
        $$\beta_{end}=0.02$$
        中间的beta使用线性插值获取
        (也有一些其他取样方式，如cosin等)
    2. 计算得出$\alpha_t$，$\overline{\alpha}_t$
2. 训练网络(add noise 过程):
    1. 随机取样图片$x_0$
    2. 随机取样步数$t$（加了多少次噪声）
    3. 从标准高斯噪声取样$\epsilon$，计算$x_t$
        $$x_t=\sqrt{\overline{\alpha}_t}*x_0+\sqrt{1-\overline{\alpha}_t}*\epsilon$$
    4. 对网络进行训练，其loss function为
        $$|\epsilon - \epsilon_{\theta}(x_t,t)|^2$$
3. 生成图片(denoise 过程)
   1. 随机取样，得出T步骤图片 $x_t = N(0,I)$
   2. 从T、T-1、T-2、... 1 ,依次进行denoise
      1. init noise
        $$z = N(0,I) \quad if \quad t>1  \quad else  \quad z=0$$
      2. predict backward noise
        $$\epsilon_{\theta}(x_t,t)$$
      3. 由公式(12)、公式(10)推断，可以得出
        
        $$x_{t-1}=\frac{x_t-\frac{1-\alpha_t}{\sqrt{1-\overline{\alpha}_t}}*\epsilon_{\theta}(x_t,t)}{\sqrt{\alpha_t}}+\sigma_t*z$$
        
        $$其中：\sigma_t=\sqrt{\beta_t} \quad (假定，非真实方差)$$

      <!-- 3. compute $x_0$
        $$x_0=\frac{x_t-\sqrt{(1-\overline{\alpha_t})}*\epsilon_{\theta}(x_t,t)}{\sqrt{\overline{\alpha_t}}}
        $$
      4. compute $x_{t-1}$
        $$x_{t-1}=\frac{\sqrt{\overline{\alpha}_{t-1}}*\beta_t}{1-\overline{\alpha_t}}*x_0+\frac{\sqrt{\alpha_t}*(1-\overline{\alpha}_{t-1})}{1-\overline{\alpha}_t}*x_t+\sqrt{\beta_t}*z$$
        $$\overline{\alpha}_{t-1}=1 \quad when \quad t=1$$ -->
            
        

   3. 输出最终生成的图片$x_0$
   