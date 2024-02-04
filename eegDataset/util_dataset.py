from sklearn.preprocessing import StandardScaler

def standardize_data_batch(x_train, x_test, n_channel):
    # 在batch维度
    b,c,t = x_train.shape
    for i in range(b):
        scaler = StandardScaler()
        scaler.fit(x_train[i, :, :])
        x_train[i, :, :] = scaler.transform(x_train[i, :, :])
        x_test[i, :, :] = scaler.transform(x_test[i, :, :])
    return x_train, x_test

def standardize_data(x_train, x_test, n_channel):
    # 在通道维度进行归一化
    x_train = x_train.copy()
    x_test = x_test.copy()
    b,c,t = x_train.shape
    for j in range(n_channel):
        scaler = StandardScaler()
        scaler.fit(x_train[:, j, :])
        x_train[:, j, :] = scaler.transform(x_train[:, j, :])
        x_test[:, j, :] = scaler.transform(x_test[:, j, :])
    return x_train, x_test