import nn

class PerceptronModel(object):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        self.w = nn.Parameter(1, dimensions)

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """
        return nn.DotProduct(x,self.w)

    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        ans = nn.as_scalar(self.run(x))
        if ans >= 0 : return 1
        else : return -1

    def train(self, dataset):
        """
        Train the perceptron until convergence.
        """
        accuracy = 0
        batch_size = 1
        while accuracy != 1:
            accuracy = 1
            for x,y in dataset.iterate_once(batch_size):
                predict_ans = self.get_prediction(x)
                actual_ans = nn.as_scalar(y)
                if predict_ans != actual_ans :
                    accuracy = 0
                    direction = x
                    multiplier = actual_ans
                    self.w.update(direction, multiplier) 

class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self):
        # Initialize your model parameters here
        self.w1 = nn.Parameter(1,50)
        self.b1 = nn.Parameter(1,50)
        self.w2 = nn.Parameter(50,100)
        self.b2 = nn.Parameter(1,100)
        self.w3 = nn.Parameter(100,1)
        self.b3 = nn.Parameter(1,1)

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        self.l1 = nn.AddBias(nn.Linear(x,self.w1),self.b1)
        self.n1 = nn.ReLU(self.l1)
        self.l2 = nn.AddBias(nn.Linear(self.n1,self.w2),self.b2)
        self.n2 = nn.ReLU(self.l2)
        self.l3 = nn.AddBias(nn.Linear(self.n2,self.w3),self.b3)
        return self.l3

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        predict_ans = self.run(x)
        return nn.SquareLoss(predict_ans,y)

    def train(self, dataset):
        """
        Trains the model.
        """
        average_loss = float('inf')
        batch_size = 10
        while average_loss >= 0.02 :
            average_loss = 0
            for x,y in dataset.iterate_once(batch_size) :
                loss = self.get_loss(x, y)
                average_loss += nn.as_scalar(loss)
                grad_w1, grad_w2, grad_w3, grad_b1, grad_b2, grad_b3 = nn.gradients(loss, [self.w1,self.w2, self.w3, self.b1, self.b2, self.b3])
                multiplier = -0.005
                self.w1.update(grad_w1, multiplier)
                self.w2.update(grad_w2, multiplier)
                self.w3.update(grad_w3, multiplier)
                self.b1.update(grad_b1, multiplier)
                self.b2.update(grad_b2, multiplier)
                self.b3.update(grad_b3, multiplier)

class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Initialize your model parameters here
        self.w1 = nn.Parameter(784,200)
        self.b1 = nn.Parameter(1,200)
        self.w2 = nn.Parameter(200,100)
        self.b2 = nn.Parameter(1,100)
        self.w3 = nn.Parameter(100,100)
        self.b3 = nn.Parameter(1,100)
        self.w4 = nn.Parameter(100,100)
        self.b4 = nn.Parameter(1,100)
        self.w5 = nn.Parameter(100,10)
        self.b5 = nn.Parameter(1,10)

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        self.l1 = nn.AddBias(nn.Linear(x,self.w1),self.b1)
        self.n1 = nn.ReLU(self.l1)
        self.l2 = nn.AddBias(nn.Linear(self.n1,self.w2),self.b2)
        self.n2 = nn.ReLU(self.l2)
        self.l3 = nn.AddBias(nn.Linear(self.n2,self.w3),self.b3)
        self.n3 = nn.ReLU(self.l3)
        self.l4 = nn.AddBias(nn.Linear(self.n3,self.w4),self.b4)
        self.n4 = nn.ReLU(self.l4)
        self.l5 = nn.AddBias(nn.Linear(self.n4,self.w5),self.b5)
        return self.l5


    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        predict_ans = self.run(x)
        return nn.SoftmaxLoss(predict_ans, y)

    def train(self, dataset):
        """
        Trains the model.
        """
        batch_size = 10
        multiplier = -0.005
        for _ in range(10) :
            for x,y in dataset.iterate_once(batch_size) :
                loss = self.get_loss(x, y)
                grad_w1, grad_w2, grad_w3, grad_w4, grad_w5, grad_b1, grad_b2, grad_b3, grad_b4, grad_b5 =\
                     nn.gradients(loss, [self.w1,self.w2, self.w3, self.w4, self.w5, self.b1, self.b2, self.b3, self.b4, self.b5])
                self.w1.update(grad_w1, multiplier)
                self.w2.update(grad_w2, multiplier)
                self.w3.update(grad_w3, multiplier)
                self.w4.update(grad_w4, multiplier)
                self.w5.update(grad_w5, multiplier)
                self.b1.update(grad_b1, multiplier)
                self.b2.update(grad_b2, multiplier)
                self.b3.update(grad_b3, multiplier)
                self.b4.update(grad_b4, multiplier)
                self.b5.update(grad_b5, multiplier)
            accuracy = dataset.get_validation_accuracy()
            if accuracy > 0.976 : break

class LanguageIDModel(object):
    """
    A model for language identification at a single-word granularity.

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Our dataset contains words from five different languages, and the
        # combined alphabets of the five languages contain a total of 47 unique
        # characters.
        # You can refer to self.num_chars or len(self.languages) in your code
        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]

        # Initialize your model parameters here
        self.d = 100
        self.multiplier = -0.005
        self.batch_size = 10
        self.wx = nn.Parameter(self.num_chars, self.d)
        self.whidden = nn.Parameter(self.d, self.d)
        self.wout1 = nn.Parameter(self.d, 50)
        self.bout1 = nn.Parameter(1,50)
        self.wout2 = nn.Parameter(50,5)
        self.bout2 = nn.Parameter(1,5)

    def run(self, xs):
        """
        Runs the model for a batch of examples.

        Although words have different lengths, our data processing guarantees
        that within a single batch, all words will be of the same length (L).

        Here `xs` will be a list of length L. Each element of `xs` will be a
        node with shape (batch_size x self.num_chars), where every row in the
        array is a one-hot vector encoding of a character. For example, if we
        have a batch of 8 three-letter words where the last word is "cat", then
        xs[1] will be a node that contains a 1 at position (7, 0). Here the
        index 7 reflects the fact that "cat" is the last word in the batch, and
        the index 0 reflects the fact that the letter "a" is the inital (0th)
        letter of our combined alphabet for this task.

        Your model should use a Recurrent Neural Network to summarize the list
        `xs` into a single node of shape (batch_size x hidden_size), for your
        choice of hidden_size. It should then calculate a node of shape
        (batch_size x 5) containing scores, where higher scores correspond to
        greater probability of the word originating from a particular language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
        Returns:
            A node with shape (batch_size x 5) containing predicted scores
                (also called logits)
        """
        for i in range(len(xs)) :
            if i == 0 :
                hi = nn.Linear(xs[i], self.wx)
                continue
            z = nn.Add(nn.Linear(xs[i], self.wx), nn.Linear(hi, self.whidden))
            hi = nn.ReLU(z)
        self.l1 = nn.AddBias(nn.Linear(hi,self.wout1),self.bout1)
        self.n1 = nn.ReLU(self.l1)
        self.l2 = nn.AddBias(nn.Linear(self.n1,self.wout2),self.bout2)
        return self.l2
        

    def get_loss(self, xs, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 5). Each row is a one-hot vector encoding the correct
        language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
            y: a node with shape (batch_size x 5)
        Returns: a loss node
        """
        predice_ans = self.run(xs)
        return nn.SoftmaxLoss(predice_ans, y)

    def train(self, dataset):
        """
        Trains the model.
        """
        for _ in range(15) :
            for x,y in dataset.iterate_once(self.batch_size) :
                loss = self.get_loss(x, y)
                grad_wx, grad_wh, grad_w1, grad_w2, grad_b1, grad_b2 = \
                     nn.gradients(loss, [self.wx, self.whidden, self.wout1, self.wout2, self.bout1, self.bout2])
                self.wx.update(grad_wx, self.multiplier)
                self.whidden.update(grad_wh, self.multiplier)
                self.wout1.update(grad_w1, self.multiplier)
                self.wout2.update(grad_w2, self.multiplier)
                self.bout1.update(grad_b1, self.multiplier)
                self.bout2.update(grad_b2, self.multiplier)
            