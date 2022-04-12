class BST:
    "a BST node"

    def __init__(self, data, left=None, right=None):
        self.data = data
        self.left = left
        self.right = right

    def __str__(self):
        if not self.left and not self.right:  # leaf node
            return str(self.data)
        else:
            left_str = str(self.left) if self.left else "_"
            right_str = str(self.right) if self.right else "_"
            return "(" + left_str + " " + str(self.data) + " " + right_str + ")"

    # Inserts a new leaf node containing x as its data into this BST
    # Note: If the BST already contains x, do nothing
    def insert(self, x):
        # ****************
        # YOUR CODE HERE
        ## note: if value is already in tree..do nothing....return "Present"
        ## base case
        ## case when the value is found
        ##  case when there is a way to insert the node..create a BST instance and put it aas right or left child

        ## check with the root value
        if x == self.data:
            return None
        elif x < self.data:
            ## navigate to the left node
            if self.left is not None:
                self.left.insert(x)
            else:
                ## insert x
                self.left = BST(x)
        else:
            ## naviagate to the right child
            if self.right is not None:
                self.right.insert(x)
            else:
                self.right = BST(x)
        # ****************
        # pass


test_bst = BST(
    8, BST(3, BST(1), BST(6, BST(4), BST(7))), BST(10, None, BST(14, BST(13), None))
)

print(f"test_bst = {str(test_bst)}")
# test_bst = ((1 3 (4 6 7)) 8 (_ 10 (13 14 _)))

test_bst.insert(9)
print(f"test_bst after --- insert(9) = {str(test_bst)}")
# test_bst after insert(9) = ((1 3 (4 6 7)) 8 (9 10 (13 14 _)))

test_bst.insert(6)
print(f"test_bst after --- insert(6) = {str(test_bst)}")
# test_bst after insert(6) = ((1 3 (4 6 7)) 8 (9 10 (13 14 _)))

test_bst.insert(0)
print(f"test_bst after --- insert(0) = {str(test_bst)}")
# test_bst after insert(0) = (((0 1 _) 3 (4 6 7)) 8 (9 10 (13 14 _)))
