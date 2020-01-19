class Node:
    def __init__(self, value=None, default=None, is_leaf=None, index=None):
        """
        Initialization of node class
        :param value: the value of the node
        :param default: the default value
        :param is_leaf: is a leaf
        :param index: index as a feature
        """
        self.childrens = {}
        self.value = value
        self.default = default
        self.leaf = is_leaf
        self.index = index
        self.is_root = False

    def set_is_root(self):
        """
        Set the node to be root
        :return:
        """
        self.is_root = True

    def get_is_root(self):
        """
        :return: True if root is node, else False
        """
        return self.is_root

    def add_child(self, value, child):
        """
        Add a child to a node
        :param value: the value of child
        :param child: the name of the child
        :return:
        """
        self.childrens[value] = child

    def get_default(self):
        """
        :return: the default value
        """
        return self.default

    def is_leaf(self):
        """
        :return: Is the node a leaf
        """
        return self.leaf

    def get_value(self):
        """
        :return: Return the value
        """
        return self.value

    def get_index(self):
        """
        :return: Return the index
        """
        return self.index

    def get_childrens(self):
        """
        :return: Return all the childrens
        """
        return self.childrens
