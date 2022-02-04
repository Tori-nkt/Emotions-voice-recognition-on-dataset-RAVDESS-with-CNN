import time
import tensorflow as tf

gpu_device_name = tf.test.gpu_device_name()
print(gpu_device_name)
print(tf.test.is_gpu_available())

def performanceTest(device_name,size):
    with tf.device(device_name): # Выберите конкретное устройство
        W = tf.random_normal([size, size],name='W') # Создание матрицы W путем случайной генерации значений
        X = tf.random_normal([size, size],name='X') # Создать матрицу X путем случайного генерирования значений
        mul = tf.matmul(W, X,name='mul')
        sum_result = tf.reduce_sum(mul,name='sum') # Для суммирования значений в матрице mul

    startTime = time.time() # Время начала записи
    tfconfig=tf.ConfigProto(log_device_placement=True) # Представляет информацию об устройстве отображения
    with tf.Session(config=tfconfig) as sess:
        result = sess.run(sum_result)
    takeTimes=time.time()  - startTime
    print(device_name," size=",size,"Time:",takeTimes )
    return takeTimes # Возврат во время работы