import lstm
import random
import numpy as np

x_train, y_train, x_test, y_test = lstm.load_data('sp500.csv', 50, True)


model = lstm.build_model([1, 25, 50, 1])

model.fit(
    x_train,
    y_train,
    batch_size=512,
    epochs=5,
    validation_split=0.05
)

model.save("My_Model.h5")
def arrayToInput(array):
    test = np.array(array)
    test = test.reshape(1, test.shape[0], 1)
    return test

buff = []
for x in range(400):
    buff.append(y_test[x])


for x in range(30):
    temp = arrayToInput(buff)
    ans = model.predict(temp)
    print(str(ans))
    buff.append(ans[0][0])


lstm.plot(buff)






# plen = 50
# predictions = lstm.predict_sequences_multiple(model, x_test, plen, plen)
# lstm.plot_results_multiple(predictions, y_test, plen)
# hi = 1
