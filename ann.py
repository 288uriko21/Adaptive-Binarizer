import numpy as np


def sigmoid(x):
    # Функция активации sigmoid:: f(x) = 1 / (1 + e^(-x))
    return 1 / (1 + np.exp(-x))


def deriv_sigmoid(x):
    # Производная от sigmoid: f'(x) = f(x) * (1 - f(x))
    fx = sigmoid(x)
    return fx * (1 - fx)


def mse_loss(y_true, y_pred):
    return ((y_true - y_pred) ** 2).mean()


class NeuralNetwork:

    def __init__(self):
        # Вес
        self.w1 = np.random.normal()
        self.w2 = np.random.normal()
        self.w3 = np.random.normal()
        self.w4 = np.random.normal()
        self.w5 = np.random.normal()
        self.w6 = np.random.normal()
        self.w7 = np.random.normal()
        self.w8 = np.random.normal()
        self.w9 = np.random.normal()
        self.w10 = np.random.normal()
        self.w11 = np.random.normal()
        self.w12 = np.random.normal()
        self.w13 = np.random.normal()
        self.w14 = np.random.normal()
        self.w15 = np.random.normal()
        self.w16 = np.random.normal()
        self.w17 = np.random.normal()
        self.w18 = np.random.normal()
        self.w19 = np.random.normal()
        self.w20 = np.random.normal()
        self.w21 = np.random.normal()
        self.w22 = np.random.normal()
        self.w23 = np.random.normal()
        self.w24 = np.random.normal()
        self.w25 = np.random.normal()
        self.w26 = np.random.normal()
        self.w27 = np.random.normal()
        self.w28 = np.random.normal()
        self.w29 = np.random.normal()
        self.w30 = np.random.normal()
        self.w31 = np.random.normal()
        self.w32 = np.random.normal()
        self.w33 = np.random.normal()
        self.w34 = np.random.normal()
        self.w35 = np.random.normal()
        self.w36 = np.random.normal()
        self.w37 = np.random.normal()
        self.w38 = np.random.normal()
        self.w39 = np.random.normal()
        self.w40 = np.random.normal()
        self.w41 = np.random.normal()
        self.w42 = np.random.normal()

        # Смещения
        self.b1 = np.random.normal()
        self.b2 = np.random.normal()
        self.b3 = np.random.normal()
        self.b4 = np.random.normal()
        self.b5 = np.random.normal()
        self.b6 = np.random.normal()
        self.b7 = np.random.normal()
        self.b8 = np.random.normal()

    def feedforward(self, x):
        # x является массивом numpy с двумя элементами
        h1 = sigmoid(self.w1 * x[0] + self.w2 * x[1] + self.w3 * x[2] + self.w4 * x[3] + self.w5 * x[4] + self.b1)
        h2 = sigmoid(self.w6 * x[0] + self.w7 * x[1] + self.w8 * x[2] + self.w9 * x[3] + self.w10 * x[4] + self.b2)
        h3 = sigmoid(self.w11 * x[0] + self.w12 * x[1] + self.w13 * x[2] + self.w14 * x[3] + self.w15 * x[4] + self.b3)
        h4 = sigmoid(self.w16 * x[0] + self.w17 * x[1] + self.w18 * x[2] + self.w19 * x[3] + self.w20 * x[4] + self.b4)
        h5 = sigmoid(self.w21 * x[0] + self.w22 * x[1] + self.w23 * x[2] + self.w24 * x[3] + self.w25 * x[4] + self.b5)
        h6 = sigmoid(self.w26 * x[0] + self.w27 * x[1] + self.w28 * x[2] + self.w29 * x[3] + self.w30 * x[4] + self.b6)
        h7 = sigmoid(self.w31 * x[0] + self.w32 * x[1] + self.w33 * x[2] + self.w34 * x[3] + self.w35 * x[4] + self.b7)
        o1 = sigmoid(self.w36 * h1 + self.w37 * h2 + self.w38 * h3 + self.w39 * h4 + self.w40 * h5 + self.w41 * h6 + self.w42 * h7 + self.b8)
        return o1

    def train(self, data, all_y_trues):
        learn_rate = 0.7
        epochs = 1000  # количество циклов во всём наборе данных

        for epoch in range(epochs):
            for x, y_true in zip(data, all_y_trues):
                # --- Выполняем обратную связь (нам понадобятся эти значения в дальнейшем)
                sum_h1 = self.w1 * x[0] + self.w2 * x[1] + self.w3 * x[2] + self.w4 * x[3] + self.w5 * x[4] + self.b1
                h1 = sigmoid(sum_h1)

                sum_h2 = self.w6 * x[0] + self.w7 * x[1] + self.w8 * x[2] + self.w9 * x[3] + self.w10 * x[4] + self.b2
                h2 = sigmoid(sum_h2)

                sum_h3 = self.w11 * x[0] + self.w12 * x[1] + self.w13 * x[2] + self.w14 * x[3] + self.w15 * x[4] + self.b3
                h3 = sigmoid(sum_h3)

                sum_h4 = self.w16 * x[0] + self.w17 * x[1] + self.w18 * x[2] + self.w19 * x[3] + self.w20 * x[4] + self.b4
                h4 = sigmoid(sum_h4)

                sum_h5 = self.w21 * x[0] + self.w22 * x[1] + self.w23 * x[2] + self.w24 * x[3] + self.w25 * x[4] + self.b5
                h5 = sigmoid(sum_h5)

                sum_h6 = self.w26 * x[0] + self.w27 * x[1] + self.w28 * x[2] + self.w29 * x[3] + self.w30 * x[4] + self.b6
                h6 = sigmoid(sum_h6)

                sum_h7 = self.w31 * x[0] + self.w32 * x[1] + self.w33 * x[2] + self.w34 * x[3] + self.w35 * x[4] + self.b7
                h7 = sigmoid(sum_h7)

                sum_o1 = self.w36 * h1 + self.w37 * h2 + self.w38 * h3 + self.w39 * h4 + self.w40 * h5 + self.w41 * h6 + self.w42 * h7 + self.b8
                o1 = sigmoid(sum_o1)
                y_pred = o1

                # --- Подсчет частных производных
                d_L_d_ypred = -2 * (y_true - y_pred)

                # Нейрон o1
                d_ypred_d_w36 = h1 * deriv_sigmoid(sum_o1)
                d_ypred_d_w37 = h2 * deriv_sigmoid(sum_o1)
                d_ypred_d_w38 = h3 * deriv_sigmoid(sum_o1)
                d_ypred_d_w39 = h4 * deriv_sigmoid(sum_o1)
                d_ypred_d_w40 = h5 * deriv_sigmoid(sum_o1)
                d_ypred_d_w41 = h6 * deriv_sigmoid(sum_o1)
                d_ypred_d_w42 = h7 * deriv_sigmoid(sum_o1)
                d_ypred_d_b8 = deriv_sigmoid(sum_o1)

                d_ypred_d_h1 = self.w36 * deriv_sigmoid(sum_o1)
                d_ypred_d_h2 = self.w37 * deriv_sigmoid(sum_o1)
                d_ypred_d_h3 = self.w38 * deriv_sigmoid(sum_o1)
                d_ypred_d_h4 = self.w39 * deriv_sigmoid(sum_o1)
                d_ypred_d_h5 = self.w40 * deriv_sigmoid(sum_o1)
                d_ypred_d_h6 = self.w41 * deriv_sigmoid(sum_o1)
                d_ypred_d_h7 = self.w42 * deriv_sigmoid(sum_o1)

                # Нейрон h1
                d_h1_d_w1 = x[0] * deriv_sigmoid(sum_h1)
                d_h1_d_w2 = x[1] * deriv_sigmoid(sum_h1)
                d_h1_d_w3 = x[2] * deriv_sigmoid(sum_h1)
                d_h1_d_w4 = x[3] * deriv_sigmoid(sum_h1)
                d_h1_d_w5 = x[4] * deriv_sigmoid(sum_h1)
                d_h1_d_b1 = deriv_sigmoid(sum_h1)

                # Нейрон h2
                d_h2_d_w6 = x[0] * deriv_sigmoid(sum_h2)
                d_h2_d_w7 = x[1] * deriv_sigmoid(sum_h2)
                d_h2_d_w8 = x[2] * deriv_sigmoid(sum_h2)
                d_h2_d_w9 = x[3] * deriv_sigmoid(sum_h2)
                d_h2_d_w10 = x[4] * deriv_sigmoid(sum_h2)
                d_h2_d_b2 = deriv_sigmoid(sum_h2)

                # Нейрон h3
                d_h3_d_w11 = x[0] * deriv_sigmoid(sum_h3)
                d_h3_d_w12 = x[1] * deriv_sigmoid(sum_h3)
                d_h3_d_w13 = x[2] * deriv_sigmoid(sum_h3)
                d_h3_d_w14 = x[3] * deriv_sigmoid(sum_h3)
                d_h3_d_w15 = x[4] * deriv_sigmoid(sum_h3)
                d_h3_d_b3 = deriv_sigmoid(sum_h3)

                # Нейрон h4
                d_h4_d_w16 = x[0] * deriv_sigmoid(sum_h4)
                d_h4_d_w17 = x[1] * deriv_sigmoid(sum_h4)
                d_h4_d_w18 = x[2] * deriv_sigmoid(sum_h4)
                d_h4_d_w19 = x[3] * deriv_sigmoid(sum_h4)
                d_h4_d_w20 = x[4] * deriv_sigmoid(sum_h4)
                d_h4_d_b4 = deriv_sigmoid(sum_h4)

                # Нейрон h5
                d_h5_d_w21 = x[0] * deriv_sigmoid(sum_h5)
                d_h5_d_w22 = x[1] * deriv_sigmoid(sum_h5)
                d_h5_d_w23 = x[2] * deriv_sigmoid(sum_h5)
                d_h5_d_w24 = x[3] * deriv_sigmoid(sum_h5)
                d_h5_d_w25 = x[4] * deriv_sigmoid(sum_h5)
                d_h5_d_b5 = deriv_sigmoid(sum_h5)

                # Нейрон h6
                d_h6_d_w26 = x[0] * deriv_sigmoid(sum_h6)
                d_h6_d_w27 = x[1] * deriv_sigmoid(sum_h6)
                d_h6_d_w28 = x[2] * deriv_sigmoid(sum_h6)
                d_h6_d_w29 = x[3] * deriv_sigmoid(sum_h6)
                d_h6_d_w30 = x[4] * deriv_sigmoid(sum_h6)
                d_h6_d_b6 = deriv_sigmoid(sum_h6)

                # Нейрон h7
                d_h7_d_w31 = x[0] * deriv_sigmoid(sum_h7)
                d_h7_d_w32 = x[1] * deriv_sigmoid(sum_h7)
                d_h7_d_w33 = x[2] * deriv_sigmoid(sum_h7)
                d_h7_d_w34 = x[3] * deriv_sigmoid(sum_h7)
                d_h7_d_w35 = x[4] * deriv_sigmoid(sum_h7)
                d_h7_d_b7 = deriv_sigmoid(sum_h7)

                ###### Обновляем вес и смещения
                # Нейрон h1
                self.w1 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w1
                self.w2 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w2
                self.w3 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w3
                self.w4 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w4
                self.w5 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w5
                self.b1 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_b1

                # Нейрон h2
                self.w6 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w6
                self.w7 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w7
                self.w8 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w8
                self.w9 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w9
                self.w10 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w10
                self.b2 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_b2

                # Нейрон h3
                self.w11 -= learn_rate * d_L_d_ypred * d_ypred_d_h3 * d_h3_d_w11
                self.w12 -= learn_rate * d_L_d_ypred * d_ypred_d_h3 * d_h3_d_w12
                self.w13 -= learn_rate * d_L_d_ypred * d_ypred_d_h3 * d_h3_d_w13
                self.w14 -= learn_rate * d_L_d_ypred * d_ypred_d_h3 * d_h3_d_w14
                self.w15 -= learn_rate * d_L_d_ypred * d_ypred_d_h3 * d_h3_d_w15
                self.b3 -= learn_rate * d_L_d_ypred * d_ypred_d_h3 * d_h3_d_b3

                # Нейрон h4
                self.w16 -= learn_rate * d_L_d_ypred * d_ypred_d_h4 * d_h4_d_w16
                self.w17 -= learn_rate * d_L_d_ypred * d_ypred_d_h4 * d_h4_d_w17
                self.w18 -= learn_rate * d_L_d_ypred * d_ypred_d_h4 * d_h4_d_w18
                self.w19 -= learn_rate * d_L_d_ypred * d_ypred_d_h4 * d_h4_d_w19
                self.w20 -= learn_rate * d_L_d_ypred * d_ypred_d_h4 * d_h4_d_w20
                self.b4 -= learn_rate * d_L_d_ypred * d_ypred_d_h4 * d_h4_d_b4

                # Нейрон h5
                self.w21 -= learn_rate * d_L_d_ypred * d_ypred_d_h5 * d_h5_d_w21
                self.w22 -= learn_rate * d_L_d_ypred * d_ypred_d_h5 * d_h5_d_w22
                self.w23 -= learn_rate * d_L_d_ypred * d_ypred_d_h5 * d_h5_d_w23
                self.w24 -= learn_rate * d_L_d_ypred * d_ypred_d_h5 * d_h5_d_w24
                self.w25 -= learn_rate * d_L_d_ypred * d_ypred_d_h5 * d_h5_d_w25
                self.b5 -= learn_rate * d_L_d_ypred * d_ypred_d_h5 * d_h5_d_b5

                # Нейрон h6
                self.w26 -= learn_rate * d_L_d_ypred * d_ypred_d_h6 * d_h6_d_w26
                self.w27 -= learn_rate * d_L_d_ypred * d_ypred_d_h6 * d_h6_d_w27
                self.w28 -= learn_rate * d_L_d_ypred * d_ypred_d_h6 * d_h6_d_w28
                self.w29 -= learn_rate * d_L_d_ypred * d_ypred_d_h6 * d_h6_d_w29
                self.w30 -= learn_rate * d_L_d_ypred * d_ypred_d_h6 * d_h6_d_w30
                self.b6 -= learn_rate * d_L_d_ypred * d_ypred_d_h6 * d_h6_d_b6

                # Нейрон h7
                self.w31 -= learn_rate * d_L_d_ypred * d_ypred_d_h7 * d_h7_d_w31
                self.w32 -= learn_rate * d_L_d_ypred * d_ypred_d_h7 * d_h7_d_w32
                self.w33 -= learn_rate * d_L_d_ypred * d_ypred_d_h7 * d_h7_d_w33
                self.w34 -= learn_rate * d_L_d_ypred * d_ypred_d_h7 * d_h7_d_w34
                self.w35 -= learn_rate * d_L_d_ypred * d_ypred_d_h7 * d_h7_d_w35
                self.b7 -= learn_rate * d_L_d_ypred * d_ypred_d_h7 * d_h7_d_b7

                # Нейрон o1
                self.w36 -= learn_rate * d_L_d_ypred * d_ypred_d_w36
                self.w37 -= learn_rate * d_L_d_ypred * d_ypred_d_w37
                self.w38 -= learn_rate * d_L_d_ypred * d_ypred_d_w38
                self.w39 -= learn_rate * d_L_d_ypred * d_ypred_d_w39
                self.w40 -= learn_rate * d_L_d_ypred * d_ypred_d_w40
                self.w41 -= learn_rate * d_L_d_ypred * d_ypred_d_w41
                self.w42 -= learn_rate * d_L_d_ypred * d_ypred_d_w42
                self.b8 -= learn_rate * d_L_d_ypred * d_ypred_d_b8

            # Подсчитываем общую потерю в конце каждой фазы
            y_preds = np.apply_along_axis(self.feedforward, 1, data)
            loss = mse_loss(all_y_trues, y_preds)
            print("Epoch %d loss: %.3f" % (epoch, loss))


f = open('traindata/2.txt', 'r')
text = f.read()
f.close()

lines = text.split('\n')

data = []
all_y_trues = []

for line in lines:
    lst = list(line.split(' '))
    if lst == ['']:
        break
    #print(lst)
    lst.remove('')
    lst.remove('')
    #print(lst)
    n, b, g, r, delta, accuracy, med, cropped = map(float, lst)
    data.append([b, g, r, med, cropped])
    all_y_trues.append(delta)

print(data)
print(all_y_trues)

network = NeuralNetwork()
network.train(data, all_y_trues)

print('w1 =', network.w1, '\n')
print('w2 =', network.w2, '\n')
print('w3 =', network.w3, '\n')
print('w4 =', network.w4, '\n')
print('w5 =', network.w5, '\n')
print('w6 =', network.w6, '\n')
print('w7 =', network.w7, '\n')
print('w8 =', network.w8, '\n')
print('w9 =', network.w9, '\n')
print('w10 =', network.w10, '\n')
print('w11 =', network.w11, '\n')
print('w12 =', network.w12, '\n')
print('w13 =', network.w13, '\n')
print('w14 =', network.w14, '\n')
print('w15 =', network.w15, '\n')
print('w16 =', network.w16, '\n')
print('w17 =', network.w17, '\n')
print('w18 =', network.w18, '\n')
print('w19 =', network.w19, '\n')
print('w20 =', network.w20, '\n')
print('w21 =', network.w21, '\n')
print('w22 =', network.w22, '\n')
print('w23 =', network.w23, '\n')
print('w24 =', network.w24, '\n')
print('w25 =', network.w25, '\n')
print('w26 =', network.w26, '\n')
print('w27 =', network.w27, '\n')
print('w28 =', network.w28, '\n')
print('w29 =', network.w29, '\n')
print('w30 =', network.w30, '\n')
print('w31 =', network.w31, '\n')
print('w32 =', network.w32, '\n')
print('w33 =', network.w33, '\n')
print('w34 =', network.w34, '\n')
print('w35 =', network.w35, '\n')
print('w36 =', network.w36, '\n')
print('w37 =', network.w37, '\n')
print('w38 =', network.w38, '\n')
print('w39 =', network.w39, '\n')
print('w40 =', network.w40, '\n')
print('w41 =', network.w41, '\n')
print('w42 =', network.w42, '\n')

print('b1 =', network.b1, '\n')
print('b2 =', network.b2, '\n')
print('b3 =', network.b3, '\n')
print('b4 =', network.b4, '\n')
print('b5 =', network.b5, '\n')
print('b6 =', network.b6, '\n')
print('b7 =', network.b7, '\n')
print('b8 =', network.b8, '\n')
