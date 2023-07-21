import math

def gaussian_density(x, mu, sigma):
    result = math.exp(-0.5 * ( (x-mu)/sigma )** 2)
    result /= sigma * (2 * math.pi) ** 0.5
    return result

class Action:
    def __init__(self, name):
        self.name = name

    def get_value(self) -> float:
        """Return the value of the action under current conditions"""
        pass

class ConstantAction(Action):
    def __init__(self, name, value):
        super().__init__(name)
        self.value = value

    def get_value(self, *args):
        return self.value

class CongestedConstantAction(Action):
    def __init__(self, name, capacity, below_capacity_value, above_capacity_value):
        super().__init__(name)
        self.capacity = capacity
        self.below_capacity_value = below_capacity_value
        self.above_capacity_value = above_capacity_value

    def get_value(self, *args):
        if args[0] <= self.capacity:
            return self.below_capacity_value
        else:
            return self.above_capacity_value

class GaussianCongestedAction(Action):
    def __init__(self, name, capacity, left_tail_sd, right_tail_sd):
        super().__init__(name)
        self.capacity = capacity
        self.left_tail_sd = left_tail_sd
        self.right_tail_sd = right_tail_sd

    def get_value(self, *args):
        if args[0] <= self.capacity:
            #in-capacity case
            return gaussian_density(args[0], self.capacity, self.left_tail_sd)
        else:
            return gaussian_density(args[0], self.capacity, self.right_tail_sd)

class RightGaussianCongestedAction(Action):
    """Action where the right-tail uses gaussian density but the left is constant"""
    def __init__(self, name, capacity, left_tail_constant, right_tail_sd):
        super().__init__(name)
        self.capacity = capacity
        self.left_tail_constant = left_tail_constant
        self.right_tail_sd = right_tail_sd

    def get_value(self, *args):
        if args[0] <= self.capacity:
            #in-capacity case
            return self.left_tail_constant
        else:
            return gaussian_density(args[0], self.capacity, self.right_tail_sd)