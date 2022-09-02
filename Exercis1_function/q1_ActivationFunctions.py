import numpy as np
import tensorflow as fl
def softmax(x):

    if x.ndim == 1:
        shiftx = x - np.max(x)
        exps = np.exp(shiftx)
        return exps / np.sum(exps)

    else:
        x = x - np.max(x,axis=1, keepdims=True)
        x = np.exp(x)
        x /= np.sum(x,axis=1, keepdims=True)
        return x


def ReLU(x):
    ### YOUR CODE HERE
    x=np.maximum(x, 0)

    ### END YOUR CODE

    return x


def Leaky_ReLU(x):
    ### YOUR CODE HERE
    x = np.maximum(x, x*0.01)

    ### END YOUR CODE

    return x


def ELU(x):
    ### YOUR CODE HERE
    # tafrigh max as tamam onhai ke az x>=0 hastan barai jologiri az overflow dar np.exp
    x1 = np.where(x >= 0, x - np.max(x), x)
    # anjam function ELU
    x = np.where(x <= 0, 2 * (np.exp(x1) - 1), x)
    return x


def Tanh(x):
    ### YOUR CODE HERE
    x= np.tanh(x)

    ### END YOUR CODE

    return x

def test_softmax_basic():

    print ("Running basic tests Soft_Max...")
    print("Running basic tests Soft_Max Example1 [1,2]")
    x=np.array([[1,2]])

    test1 = softmax(x)
    print(test1)
    print("Running basic tests Soft_Max Example2 [[1001, 1002], [3, 4]]")
    test2 = softmax(np.array([[1001,1002],[3,4]]))
    print (test2)
    print("Running basic tests Soft_Max Example3 [[-1001, -1002]]")
    test3 = softmax(np.array([[-1001, -1002]]))
    print (test3)

    print("End Soft_Max...")


def test_ReLU():
    print("Running basic tests ReLU...")
    print("Running basic tests ReLU Example1 ")
    x = np.array([-1, 3])

    test1 = ReLU(x)
    print(test1)
    print("Running basic tests ReLU Example2 [[1001, 1002], [3, 4]]")
    test2 = ReLU(np.array([[-1001, -1002], [3, 4]]))
    print(test2)
    print("Running basic tests ReLU Example3 [[-1001, -1002]]")
    test3 = ReLU(np.array([[-1001, -1002]]))
    print(test3)

    print("End ReLU...")


def test_Leaky_ReLU():
    print("Running basic tests Leaky_ReLU...")
    print("Running basic tests Leaky_ReLU Example1 [1,2]")
    x = np.array([-1, 3])

    test1 = Leaky_ReLU(x)
    print(test1)
    print("Running basic tests Leaky_ReLU Example2 [[-1001, 1002], [-3, 4]]")
    test2 = Leaky_ReLU(np.array([[-1001, 1002], [-3, 4]]))
    print(test2)
    print("Running basic tests Leaky_ReLU Example3 [[-1001, -1002]]")
    test3 = Leaky_ReLU(np.array([[-1001, -1002]]))
    print(test3)

    print("End Leaky_ReLU...")


def test_ELU():
    print("Running basic tests ELU...")
    print("Running basic tests ELU Example1 [1,3]")
    f= np.array([[1, 3]])

    test1 = ELU(f)
    print(test1)
    print("Running basic tests ELU Example2 [[-1, 2], [-1000, 4]]")
    f = np.array([[-1, 2], [-1000, 4]])
    test2 = ELU(np.array(f))
    print(test2)
    print("Running basic tests ELU Example3 [[-1001, -1002]]")
    test3 = ELU(np.array([[-10000, 10000]]))
    print(test3)

    print("End ELU...")


def test_Tanh():
    print("Running basic tests Tanh...")
    print("Running basic tests Tanh Example1 [1,2]")
    x = np.array([1, 3])

    test1 = Tanh(x)
    print(test1)
    print("Running basic tests Tanh Example2 [[1001, 1002], [3, 4]]")
    test2 = Tanh(np.array([[1001, 1002], [3, 4]]))
    print(test2)
    print("Running basic tests Tanh Example3 [[-1001, -1002]]")
    test3 = Tanh(np.array([[-1001, -1002]]))
    print(test3)

    print("End Tanh...")

if __name__ == "__main__":
      test_softmax_basic()
      test_ReLU()
      test_Leaky_ReLU()
      test_ELU()
      test_Tanh()
