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


## 117. Populating Next Right Pointers in Each Node II


这道题，其实是一道相当trick的题目，注意follow up，要求o(1)空间复杂度。

如果不关注follow up，用传统的二叉树框架，这里可能使用层次遍历，如果使用前序、中序、后序的框架，可能前序遍历，比较适用（未验证）。对于某个节点，其左子结点的next，是右节点（如果不为空），或者该节点next的第一个不为空的子节点（**这里还有个trick的地方，就是如果该节点的next没有子节点，则需要到next的next继续找第一个不为空的子节点**）。


如果关注follow up，这意味着，我们套用前序、中序、后序的框架，这里不好用了，因为这里不能递归调用。其次，对于层次遍历，由于queue的空间复杂度为o(n)，其实这里也不适用。

因为这道题目，并不是一个完全的二叉树题目，其与链表有一定的交集。

针对每个节点进行分析，我们会发现，如果第n层已经完成next指针的设置了，那么对于n+1层，我们要做的是，从第一个不为空的指针，以此向后找其next，然后一直遍历，直到最右边。特别注意的是，在遍历的时候，不仅仅子节点向前走，父节点，也需要向前走

```python
"""
# Definition for a Node.
class Node:
    def __init__(self, val: int = 0, left: 'Node' = None, right: 'Node' = None, next: 'Node' = None):
        self.val = val
        self.left = left
        self.right = right
        self.next = next
"""

class Solution:
    def connect(self, root: 'Node') -> 'Node':
        if root is None:
            return None
        
        last_a=root

        def get_next(a,b):
            while True:
                if a is None:
                    return None,None
                if a.left:
                    if a.left is not b:
                        return a,a.left
                if a.right:
                    if a.right is not b:
                        return a.next,a.right # 这里需要特别注意，父节点，也要向右走
                a=a.next
        
        while last_a:
            a=last_a
            b=None
            first_b=None
            while True:
                next_a,next_b=get_next(a,b)
                if first_b is None:
                    first_b=next_b
                if next_b is None:
                    break
                if b:
                    b.next=next_b
                b=next_b
                a=next_a
                
            last_a=first_b
        return root
            
```


## 114. Flatten Binary Tree to Linked List

这也是一道链表与二叉树结合的题目

这道题的解法有两种：
1. 空间换时间，使用递归
2. 时间换空间，多次遍历

首先说一下空间换时间，这个的思路就非常直观，对于每一个子树都进行flatten操作，返回flatten的头结点与尾结点，然后再与父节点进行拼接，代码如下：

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def flatten(self, root: Optional[TreeNode]) -> None:
        """
        Do not return anything, modify root in-place instead.
        """
        self.my_flatten(root)

    
    def my_flatten(self,root):
        if root is None:
            return None,None
        
        l_front,l_end=self.my_flatten(root.left)
        r_front,r_end=self.my_flatten(root.right)

        root.left=None
        root.right=None

        if l_front is None and r_front is None:
            return root,root
        elif l_front is not None:
            root.right=l_front
            if r_front:
                l_end.right=r_front
                return root,r_end
            else:
                return root,l_end
        else:
            root.right=r_front
            return root,r_end

        
```


对于空间换时间，其原理如下：
1. 找到最左边的拥有左子树的节点
2. 对该节点进行处理，将其flatten
3. 回到1，继续，直到找不到符合条件的节点
思想其实不难，但是细节较多

代码如下：
```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def flatten(self, root: Optional[TreeNode]) -> None:
        """
        Do not return anything, modify root in-place instead.
        """
        if root is None:
            return None
        run_stu=True
        while run_stu:
            last_node=root

            now_node=root
            run_stu=False
            while True:
                if now_node.left is not None:
                    run_stu=True
                    last_node=now_node
                    now_node=now_node.left
                elif now_node.right is not None:
                    now_node=now_node.right
                else:
                    break
            
            # print(f'{last_node.val} , {now_node.val}')

            if run_stu:
                now_node.right=last_node.right
                last_node.right=last_node.left
                last_node.left=None

```

## 124. Binary Tree Maximum Path Sum

这道题的标签是hard，但是其思路是非常直观地，简单来说，就是找到左子树的一个path，右子树的一个path，然后与父节点进行结合，来找出最长的路径。

根据上面的思路，首先这是一道后序遍历的题目。其次，这里还有一个细节需要注意，我们在返回的时候，只能选择左子树返回，或者右子树返回，或者根节点单独返回，**不能左右子树结合后返回**。


```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def maxPathSum(self, root: Optional[TreeNode]) -> int:
        self.out=None
        self.post_order(root)
        return self.out
    
    def post_order(self,root):
        if root is None:
            return 0

        l=self.post_order(root.left)
        r=self.post_order(root.right)

        out=root.val
        out=max(out,root.val+l)
        out=max(out,root.val+r)
        fin_out=max(out,root.val+l+r)

        # print(f'{root.val} , {out} , {self.out}')

        if self.out is None or self.out<fin_out:
            self.out=fin_out
        return out
```

# 一些总结

1. 有一些二叉树与链表结合的题目，还是比较trick的