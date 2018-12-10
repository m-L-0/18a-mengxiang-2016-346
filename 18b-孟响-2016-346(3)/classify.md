第三次作业：

作业要求：9个类别的高光谱数据分类

9个类别数据集参见压缩包

每个样本由200个波段组成(即200个光谱特征)



因为一个类别有200个特征值，特征比较多。

有9个类别，所以是多分类问题。

我就想到了用keras来解决这个问题



```
#建造keras模型
from keras.callbacks import EarlyStopping
earlystop = EarlyStopping(monitor='val_acc', patience=5, verbose=0, mode='auto')

def baseline_model():
    model = Sequential()
    model.add(Dense(output_dim=128, input_dim=200, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(output_dim=256, input_dim=128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(output_dim=9, input_dim=256, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

```

为了提高模型的准确率，我才用了交叉迭代的方式

```
kfold = KFold(n_splits=8, shuffle=True, random_state=seed)
results = cross_val_score(estimator, X, dummy_y, cv=kfold)
```

