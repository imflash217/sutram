class BST:
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

    # Returns the nth smallest element of the binary search tree,
    # where n=0 returns the smallest, n=1 returns the second smallest, etc
    # Returns None if there are not n elements in the tree
    def nth_smallest(self, n):
        # ****************
        # YOUR CODE HERE
        (c, v) = self.inorder(n + 1)
        if isinstance(v, int):
            return v

        # ****************
        # pass

    def inorder(self, n):
        result = (0, None)
        if self.left:
            # print(self.left.data)
            lt = self.left.inorder(n)
            if lt[0] == n:
                ## counter reached.
                result = (n, lt[1])
                return result
            result = lt

        result = (result[0] + 1, self.data)

        # print(result[0], "--", n, "--", self.data)
        if result[0] == n:
            ## found the root as result
            return result

        if self.right:
            # print(self.right.data)
            # print(result[0], "------")
            rt = self.right.inorder(n - result[0])
            if result[0] + rt[0] == n:
                # print(n, result[0], rt[0], rt[1])
                result = (n, rt[1])
                return result
            result = (result[0] + rt[0], None)

        return result

    def inorder2(self, n, counter):
        ## visit the left node first
        ## then root and then right
        # result = []
        if self.left:
            result += self.left.inorder(n, counter)
            if len(result) == n:
                ## break this
                return result[-1]
        result.append(self.data)
        if self.right:
            result += self.right.inorder(n)

        return result  # will always be a list


test_bst = BST(
    8, BST(3, BST(1), BST(6, BST(4), BST(7))), BST(10, None, BST(14, BST(13), None))
)

print(f"test_bst = {str(test_bst)}")
# test_bst = ((1 3 (4 6 7)) 8 (_ 10 (13 14 _)))

for n in range(0, 10):
    print(f"{str(n)}th smallest: {str(test_bst.nth_smallest(n))}")

# 0th smallest: 1
# 1th smallest: 3
# 2th smallest: 4
# 3th smallest: 6
# ...
# 8th smallest: 14
# 9th smallest: None
