def layer(op):
    # def layer_decorated(self):
    #     print('the  inner')
    # op('123')
    def layer_decorated(self):
        op(self)
        print(op.__name__)
        print('The inner')
        return self

    return layer_decorated


class Test(object):
    def __init__(self):
        print("the initial method of the class")

    @layer
    def test(self):
        print('The outer')

    @layer
    def test2(self):
        print('The test2 outer')

    def __name__(self):
        print('')


if __name__=='__main__':
    print('The first')
    # instance = Test()
    # instance.test()
    # instance.test2()
    # print(instance.__name__)
    # message = [1,2,3]
    # print(message[:])
    # message = []

    if True:
        message=9
    for i in range(10):
        pass

    print(i)

    print(message)