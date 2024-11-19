# 二叉树

## 101. Symmetric Tree

众所周知，常见的树的遍历方式有：
1. 前序
2. 中序
3. 后序

这个题目个人的第一反应是中序遍历，然后检查输出的列表是否是一个对称链表，这个解法会存在corner case，如[1,2,2,2,null,2]

然而，还存在一种遍历方式，是树的层次遍历，用于该题正合适

因此，这里的解法应该为层次遍历该树，对于空节点，也应该保留，然后检查每一层的遍历结果是否是对称的，如果不是，则该树也不是对称的，如果是，那么去除空节点，继续层析遍历

代码：
```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def isSymmetric(self, root: Optional[TreeNode]) -> bool:
        now_list=[root]
        next_list=[]
        while len(now_list)>0:
            next_list=[]
            for i in now_list:
                if i is None:
                    continue
                next_list.append(i.left)
                next_list.append(i.right)
            l=0
            r=len(next_list)-1
            while l<r:
                if next_list[l] is None and next_list[r] is None:
                    pass
                elif next_list[l] is None:
                    return False
                elif next_list[r] is None:
                    return False
                elif next_list[l].val!=next_list[r].val:
                        return False
                l+=1
                r-=1
            now_list=[i for i in next_list if i is not None]
        return True
```


## 105. Construct Binary Tree from Preorder and Inorder Traversal

这道题比较经典，考验的是通过前序遍历、中序遍历，重建二叉树

这里有几个核心点：

1. 前序遍历，root元素第一个被访问，所以前序遍历的第一个节点就是root节点
2. 中序遍历，左子树遍历完成后，才会遍历右子树。所以，有了root节点，我们可以将中序遍历，划分为 [左子树的中序遍历] + root + [右子树的中序遍历]
3. 还有一个隐含的知识点，即一个树的 前序遍历、中序遍历、后序遍历，其节点数量是一致的
4. 前序遍历，是由 root + [左子树的前序遍历] + [右子树的前序遍历] 组成，其中在2中，我们可以知道左子树的节点个数，可以由3直接得出左子树的前序遍历，与右子树的前序遍历
5. 递归将树进行重建

代码：
```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def buildTree(self, preorder: List[int], inorder: List[int]) -> Optional[TreeNode]:
        # print(preorder,inorder)
        if len(preorder)==0:
            return None
        
        root_val=preorder[0]
        left_val=[]
        right_val=[]
        left_stu=True
        for i in inorder:
            if i==root_val:
                left_stu=False
            elif left_stu:
                left_val.append(i)
            else:
                right_val.append(i)
        pre_left_val=preorder[1:1+len(left_val)]
        pre_right_val=preorder[1+len(left_val):]
        
        
        out=TreeNode(root_val)
        out.left=self.buildTree(pre_left_val,left_val)
        out.right=self.buildTree(pre_right_val,right_val)
        return out
```