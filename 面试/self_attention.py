import torch
import torch.nn as nn
import torch.functional as F

# 单头self attention 实现
class one_head_self_attention(nn.Module):
    def __init__(self, dim,*args, **kwargs):
        super().__init__(*args, **kwargs)

        self.k=nn.Linear(dim,dim)
        self.q=nn.Linear(dim,dim)
        self.v=nn.Linear(dim,dim)

        self.dim=dim
        self.scale_num=dim**0.5

    def forward(self,input_data):
        # input_data shape : B T D
        k=self.k(input_data)
        q=self.q(input_data)
        v=self.v(input_data)

        B,T,D=input_data.shape

        energy=torch.bmm(q,k.permute(0,2,1))
        energy=energy/self.scale_num
        energy=torch.softmax(energy,dim=-1)
        out=torch.bmm(energy,v)

        return out
    


class self_attention(nn.Module):
    def __init__(self,nhead, dim,*args, **kwargs):
        super().__init__(*args, **kwargs)

        self.k=nn.Linear(dim,dim)
        self.q=nn.Linear(dim,dim)
        self.v=nn.Linear(dim,dim)

        self.nhead=nhead
        self.dim=dim
        self.head_dim=dim/self.nhead
        self.scale_num=(self.head_dim)**0.5


    # 多头self attention 实现，方式一
    def forward1(self,input_data,mask=None):
        # input_data shape : B T D
        # mask : B T
        k=self.k(input_data)
        q=self.q(input_data)
        v=self.v(input_data)

        B,T,D=input_data.shape
        
        k=k.view(B,T,self.nhead,-1).permute(0,2,1,3)
        q=q.view(B,T,self.nhead,-1).permute(0,2,1,3)
        v=v.view(B,T,self.nhead,-1).permute(0,2,1,3)

        # shape Batch*nhead*T*D

        energy=torch.matmul(q,k.permute(0,1,3,2))
        energy=energy/self.scale_num # energy shape: B*Nhead*T*D
        if mask is not None:
            mask=mask.view(B,1,1,T)
            energy=energy.masked_fill(mask==0,float('-inf'))

        energy=torch.softmax(energy,dim=-1)
        out=torch.matmul(energy,v) # shape B*Nhead*Query*Key

        out=out.permute(0,2,1,3).reshape(B,T,-1)
        return out

    
    # 多头self attention 实现，方式二
    def forward2(self,input_data,mask=None):
        # input_data shape : B T D
        # mask : B T
        k=self.k(input_data)
        q=self.q(input_data)
        v=self.v(input_data)

        B,T,D=input_data.shape
        
        k=k.view(B,T,self.nhead,-1)
        q=q.view(B,T,self.nhead,-1)
        v=v.view(B,T,self.nhead,-1)

        energy=torch.einsum('bqnd,bknd->bnqk',q,k)
        energy=energy/self.scale_num # energy shape : B*Nhead*Query*Key
        if mask is not None:
            mask=mask.view(B,1,1,T)
            energy=energy.masked_fill(mask==0,float('-inf'))

        energy=torch.softmax(energy,dim=-1)
        out=torch.einsum('bnqk,bknd->bqnd',energy,v)

        out=out.reshape(B,T,-1)
        return out
    
def main():
    # a=one_head_self_attention(3)
    # input_data=torch.randn((4,10,3))
    # out=a(input_data)
    # print(out.shape)

    batch_size=20
    T=10

    a=self_attention(4,12)
    input_data=torch.randn(batch_size,T,12)

    out1=a.forward1(input_data)
    out2=a.forward2(input_data)

    diff=out1-out2
    print(diff.max(),diff.min())

    mask = torch.randint(0, 2, (batch_size, T))  # Random mask with shape (batch_size, 1, 1, seq_length)

    out1=a.forward1(input_data,mask)
    out2=a.forward2(input_data,mask)
    diff=out1-out2
    print('with mask',diff.max(),diff.min())

if __name__ == "__main__":
    main()